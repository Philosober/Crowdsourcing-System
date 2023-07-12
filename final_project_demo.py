"""
to-do list:
1. attention network，满足seq-seq的DQN
2. 通过Project_id找到任务的embedding，然后构造序列(T x D)，D就是[f_T, f_W]的维度
3. 把on-policy改成off-policy
4. 数据扩充（选做）
"""

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from Network import Net
import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# hyper parameters
TASK_FEATURE = 76
EPSILON = 0.9
GAMMA = 0.9
LR = 0.001
EPOCH = 20
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 64


class Sample_Data(Dataset):
    def __init__(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
        except:
            worker_list = os.listdir(file_path)
            worker_data = []
            for worker in worker_list:
                worker_data.append(pd.read_csv(file_path + '/' + worker))
            self.data = pd.concat(worker_data, axis=0)
        finally:
            self.data = np.array(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return list(self.data[index])


class DQN():
    def __init__(self, file_path):
        self.eval_net, self.target_net = Net(TASK_FEATURE * 2), Net(TASK_FEATURE * 2)
        self.optimizer = optim.Adam(self.eval_net.parameters(), LR)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.loss = nn.MSELoss()

        self.load_data(file_path)

    def load_data(self, file_path):
        dataset = Sample_Data(file_path)
        self.dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.project_info = pd.read_csv("project_info.csv",
                                        usecols=['project_id', 'project_feature'],
                                        index_col='project_id')

    def load_data_1(self, file_path):
        worker_list = os.listdir(file_path)
        self.worker_dataloader = {}
        for worker in worker_list:
            try:
                dataset = Sample_Data(file_path + '/' + worker)
                dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
                self.worker_dataloader[worker] = dataloader
            except:
                continue
        self.project_info = pd.read_csv("project_info.csv",
                                        usecols=['project_id', 'project_feature'],
                                        index_col='project_id')

    def get_batch(self, state, action, reward, next_state):
        batch_state = []
        batch_state_Seq = []  # list
        batch_action = []  # B x 1
        batch_next_state = []
        batch_next_state_Seq = []  # list
        batch_reward = torch.unsqueeze(reward, dim=1)

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
        return action

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
        return action

    @torch.no_grad()
    def eval(self, state, action, reward, next_state):
        # 每个epoch训练完以后，评估一下DQN什么情况，reward和loss，顺便和random策略比较
        # state, action, reward, next_state = next(iter(self.dataloader))

        batch_state, batch_state_Seq, batch_action, \
        batch_reward, batch_next_state, batch_next_state_Seq \
            = self.get_batch(state, action, reward, next_state)

        qsa_eval = self.eval_net(batch_state).squeeze(dim=2)  # B x T
        batch_action = torch.tensor(batch_action).unsqueeze(dim=1)
        q_eval = qsa_eval.gather(1, batch_action)  # B x 1

        q_next = self.target_net(batch_next_state).detach()  # B x T x 1
        q_next = torch.squeeze(q_next, dim=2)  # B x T
        q_next_max = torch.zeros(q_next.shape[0])
        for i, q in enumerate(q_next):
            q_next_max[i] = max(q_next[i, : batch_next_state_Seq[i]])
        q_next_max = torch.unsqueeze(q_next_max, 1)  # B x 1
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
        for i, batch_data in enumerate(bar):
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

    def task_feature(self, task_id):
        res = self.project_info.loc[task_id]
        res = np.array(eval(res[0]))
        return torch.from_numpy(res)

    def worker_feature(self, completed_work):
        res = torch.zeros(TASK_FEATURE)
        for task in completed_work:
            res += self.task_feature(task)
        if completed_work:
            res /= len(completed_work)
        return res  # 76

    def state_feature(self, state):
        """

        :param state: [[100, 101], [102, 103, 105]]
        :return:state_feature
        """
        state = list(eval(state))
        task_pool = state[1]
        completed_task = state[0]
        T = len(state[1])  # 任务池中任务的个数
        D = TASK_FEATURE * 2  # [ft, fw]
        res = torch.zeros((T, D))
        # 构造fw(通过state[0])
        fw = self.worker_feature(completed_task)  # 76
        res[:, : TASK_FEATURE] = fw
        # 构造ft(通过state[1][j])
        for j, task in enumerate(task_pool):
            ft = self.task_feature(task)
            res[j, TASK_FEATURE:] = ft

        return res  # T x D

    def action_index(self, state, action):
        """

        :param state: [[100, 101], [102, 103, 105]]
        :param action: 105
        :return: 2
        """
        state = list(eval(state))
        task_pool = state[1]
        return task_pool.index(action)


def main():
    model = DQN('./train')
    TEST_BATCH = 100
    # q_eval = torch.randn((10, 10, 1))
    # state_Seq = [4, 5, 8, 9, 3, 5, 5, 5, 6, 8]
    # print(model.choose_action(q_eval, state_Seq, 3).shape)
    # print(model.random_action(state_Seq, 3).shape)

    for e in range(EPOCH):
        train_bar = tqdm(model.dataloader, desc="Training", total=len(model.dataloader))
        train_bar.set_description(f'Training Epoch [{e}/{EPOCH}]')
        model.learn(train_bar)
        # model.scheduler.step()
        # state, action, reward, next_state = next(iter(model.dataloader))
        # random_reward, DQN_reward, loss = model.eval(state, action, reward, next_state)
        # print("random: {}, QDN: {}, loss: {}".format(random_reward, DQN_reward, loss))

        total_random_reward, total_DQN_reward, total_loss = 0, 0, 0
        test_bar = tqdm(model.dataloader, desc="Testing", total=TEST_BATCH)
        test_bar.set_description(f'Testing Epoch [{e}/{EPOCH}]')
        for batch_data in test_bar:
            state, action, reward, next_state = batch_data
            random_reward, DQN_reward, loss = model.eval(state, action, reward, next_state)
            total_random_reward += random_reward
            total_DQN_reward += DQN_reward
            total_loss += loss

            test_bar.set_postfix(loss=loss.item(), total_random_reward=total_random_reward, total_DQN_reward=total_DQN_reward)
        # print("random: {}, QDN: {}, loss: {}".format(total_random_reward, total_DQN_reward, total_loss))


if __name__ == '__main__':
    main()
    # data = Sample_Data('./train')
