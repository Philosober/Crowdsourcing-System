import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from Network import DQN_Net as Net
import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.utils.data import random_split
from Crowdsource import Basic_model

# hyper parameters
TASK_FEATURE = 76
EPSILON = 0.9
GAMMA = 0.9
LR = 0.00001
EPOCH = 20
TEST_ITERATION = 30
BATCH_SIZE = 128
SAVE_EPOCH = 1


class DQN(Basic_model):
    def __init__(self, batch_size, task_feature, train_mode=True, file_path='./train/worker_7945.csv'):
        super(DQN, self).__init__(batch_size, task_feature)
        self.eval_net, self.target_net = Net(TASK_FEATURE * 2), Net(TASK_FEATURE * 2)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_net.to(device=self.device)
        self.target_net.to(device=self.device)

        self.optimizer = optim.Adam(self.eval_net.parameters(), LR)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.5)
        self.loss = nn.MSELoss()

        if train_mode:
            self.load_data(file_path)

    def get_batch(self, state, action, reward, next_state):
        batch_state = []  # B x T x D
        batch_state_Seq = []  # list
        batch_action = []  # B x 1
        batch_next_state = []  # B x T x D
        batch_next_state_Seq = []  # list
        batch_reward = torch.unsqueeze(reward, dim=1)  # B x 1

        # state变成fs且可以batch
        # action对应到序列中第几个vector
        for k, s in enumerate(state):
            fs = self.state_feature(s)
            T, _ = fs.shape
            batch_state.append(fs)
            batch_state_Seq.append(T)  # 记录每个state的序列长度，方便后面取argmaxQ(s, a)来进行决策
            batch_action.append(self.action_index(s, action[k]))  # 记录action对应序列中的哪个位置
        batch_state = pad_sequence(batch_state, batch_first=True)  # B x T x D
        # next_state变成fs且可以batch
        for s in next_state:
            fs = self.state_feature(s)
            T, _ = fs.shape
            batch_next_state_Seq.append(T)  # 记录每个state的序列长度，方便后面取argmaxQ(s, a)    相当于T x 1取前t个值中的最大值
            batch_next_state.append(fs)
        batch_next_state = pad_sequence(batch_next_state, batch_first=True)  # B x T x D
        batch_action = torch.tensor(batch_action).unsqueeze(dim=1)  # B x 1

        batch_state = batch_state.to(self.device)
        batch_reward = batch_reward.to(self.device)
        batch_next_state = batch_next_state.to(self.device)
        batch_action = batch_action.to(self.device)

        return batch_state, batch_state_Seq, batch_action, batch_reward, batch_next_state, batch_next_state_Seq

    def random_action(self, state_Seq, k=1):
        """
        随机选择action
        :param state_Seq: [56, 90, 20, 40, ...]
        :return: action B x k
        """
        action = torch.zeros((len(state_Seq), k))
        for i, T in enumerate(state_Seq):
            action[i] = torch.randint(low=0, high=T, size=(k,))
        return action.to(self.device)

    def choose_action(self, qsa_eval, state_Seq, k=1):
        """
        按照DQN进行决策，选择使得Q(s, a)最大的action
        :param q_eval: B x T
        :param state_Seq: [100, 200, 43, 54, ....]   len = B
        :return: action B x topk
        """
        action = torch.zeros((len(state_Seq), k))
        for i, T in enumerate(state_Seq):
            res = qsa_eval[i, : T].topk(k, dim=0)[1]  # k
            action[i] = res
        return action.to(self.device)

    @torch.no_grad()
    def eval(self, state, action, reward, next_state):
        # 每个epoch训练完以后，评估一下DQN什么情况，reward和loss，顺便和random策略比较
        # state, action, reward, next_state = next(iter(self.dataloader))

        batch_state, batch_state_Seq, batch_action, \
        batch_reward, batch_next_state, batch_next_state_Seq \
            = self.get_batch(state, action, reward, next_state)

        qsa_eval = self.eval_net(batch_state).squeeze(dim=2)  # B x T
        q_eval = qsa_eval.gather(1, batch_action)  # B x 1

        q_next = self.target_net(batch_next_state).detach()  # B x T x 1
        q_next = torch.squeeze(q_next, dim=2)  # B x T
        q_next_max = torch.zeros(q_next.shape[0])
        for i, q in enumerate(q_next):
            q_next_max[i] = max(q_next[i, : batch_next_state_Seq[i]])
        q_next_max = torch.unsqueeze(q_next_max, 1)  # B x 1
        q_next_max = q_next_max.to(self.device)
        q_target = batch_reward + GAMMA * q_next_max

        loss = self.loss(q_eval, q_target)  # Q-learning的objective function

        # 计算random_action的reward
        random_policy = self.random_action(batch_state_Seq)
        # 计算choose_action的reward
        optimal_policy = self.choose_action(qsa_eval, batch_state_Seq)

        # batch_action是ground-truth action
        random_reward = torch.sum(random_policy == batch_action).item()
        DQN_reward = torch.sum(optimal_policy == batch_action).item()

        # print("loss: {}, random_reward: {}, DQN_reward: {}".format(loss, random_reward, DQN_reward))

        return random_reward, DQN_reward, loss

    def learn(self, bar):
        for iteration, batch_data in enumerate(bar):
            self.target_net.load_state_dict(self.eval_net.state_dict())  # target_net参数更新
            state, action, reward, next_state = batch_data
            if iteration % TEST_ITERATION == 0:
                random_reward_eval, DQN_reward_eval, loss_eval = self.eval(state, action, reward, next_state)
            else:
                bar.set_postfix(loss_eval=loss_eval.item(), random_reward=random_reward_eval,
                                DQN_reward=DQN_reward_eval)

            batch_state, _, batch_action, \
            batch_reward, batch_next_state, batch_next_state_Seq \
                = self.get_batch(state, action, reward, next_state)

            q_eval = self.eval_net(batch_state).squeeze(dim=2)  # B x T
            q_eval = q_eval.gather(1, batch_action)  # B x 1

            q_next = self.target_net(batch_next_state).detach()  # B x T x 1
            q_next = torch.squeeze(q_next, dim=2)  # B x T
            q_next_max = torch.zeros(q_next.shape[0])
            for i, q in enumerate(q_next):
                q_next_max[i] = max(q_next[i, : batch_next_state_Seq[i]])
            q_next_max = torch.unsqueeze(q_next_max, 1)  # B x 1
            q_next_max = q_next_max.to(self.device)
            q_target = batch_reward + GAMMA * q_next_max

            loss = self.loss(q_eval, q_target)  # Q-learning的objective function
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            bar.set_postfix(loss=loss.item())

            # print("batch loss:", loss.detach())

    def learn_1(self):
        for worker_id, worker_dataloader in self.worker_dataloader.items():
            print(worker_id)
            for batch_data in worker_dataloader:
                self.target_net.load_state_dict(self.eval_net.state_dict())  # target_net参数更新
                state, action, reward, next_state = batch_data

                batch_state, _, batch_action, \
                batch_reward, batch_next_state, batch_next_state_Seq \
                    = self.get_batch(state, action, reward, next_state)

                q_eval = self.eval_net(batch_state).squeeze(dim=2)  # B x T
                batch_action = torch.tensor(batch_action).unsqueeze(dim=1)  # B x 1
                q_eval = q_eval.gather(1, batch_action)  # B x 1

                q_next = self.target_net(batch_next_state).detach()  # B x T x 1
                q_next = torch.squeeze(q_next, dim=2)  # B x T
                q_next_max = torch.zeros(q_next.shape[0])
                for i, q in enumerate(q_next):
                    q_next_max[i] = max(q_next[i, : batch_next_state_Seq[i]])
                q_next_max = torch.unsqueeze(q_next_max, 1)  # B x 1
                q_target = batch_reward + GAMMA * q_next_max

                loss = self.loss(q_eval, q_target)  # Q-learning的objective function
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print("batch loss:", loss.detach())

    def load_DQN(self, load_path):
        state_dict = torch.load(load_path)
        self.eval_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)


def train(model, epoch=EPOCH, every_save=SAVE_EPOCH, save_model_path="./checkpoint"):
    for e in range(epoch):
        model.eval_net.train()
        train_bar = tqdm(model.train_dataloader, desc="Training", total=len(model.train_dataloader))
        train_bar.set_description(f'Training Epoch [{e}/{epoch}]')
        model.learn(train_bar)
        model.scheduler.step()  # learning rate scheduler

        model.eval_net.eval()
        total_random_reward, total_DQN_reward, total_loss = 0, 0, 0
        test_bar = tqdm(model.val_dataloader, desc="Testing", total=len(model.val_dataloader))
        test_bar.set_description(f'Testing Epoch [{e}/{epoch}]')
        for batch_data in test_bar:
            state, action, reward, next_state = batch_data
            random_reward, DQN_reward, loss = model.eval(state, action, reward, next_state)
            total_random_reward += random_reward
            total_DQN_reward += DQN_reward
            total_loss += loss

            test_bar.set_postfix(loss=loss.item(), total_random_reward=total_random_reward,
                                 total_DQN_reward=total_DQN_reward)

        if e % every_save == 0:
            torch.save(model.eval_net.state_dict(), save_model_path + "/DQN_%d.pkl" % e)


def main():
    # train
    model = DQN(BATCH_SIZE, TASK_FEATURE, train_mode=True, file_path="./train")
    train(model)
    # test
    # model = DQN(train_mode=False)
    # model.load_DQN("./checkpoint/DQN_5.pkl")


if __name__ == '__main__':
    main()
    # data = Sample_Data('./train')
