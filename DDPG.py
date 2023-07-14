import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split
# import gym
import time
import pandas as pd
import os
from tqdm import tqdm
#####################  hyper parameters  ####################
EPOCH = 10
TASK_FEATURE = 76
EPISODES = 200
EP_STEPS = 200
LR_ACTOR = 0.001
LR_CRITIC = 0.002
GAMMA = 0.9
TAU = 0.01
BATCH_SIZE = 32


RENDER = False


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

########################## DDPG Framework ######################


class ActorNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, a_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        return x


class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        # print(s.shape)
        # print(a.shape)
        x = self.fcs(s)
        y = self.fca(a)
        actions_value = self.out(F.relu(x + y))
        return actions_value


class DDPG(object):
    def __init__(self, a_dim, s_dim,train_mode=True, file_path='train'):
        self.a_dim, self.s_dim = a_dim, s_dim

        self.actor_eval = ActorNet(s_dim, a_dim)
        self.actor_target = ActorNet(s_dim, a_dim)
        self.critic_eval = CriticNet(s_dim, a_dim)
        self.critic_target = CriticNet(s_dim, a_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_CRITIC)

        self.loss_func = nn.MSELoss()

        if train_mode:
            self.load_data(file_path)


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

    # def load_data(self):
    #     dataset = Sample_Data('worker_7945.csv')
    #     self.dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    #     self.project_info = pd.read_csv("project_info.csv",
    #                                     usecols=['project_id', 'project_feature'],
    #                                     index_col='project_id')
    def load_data(self, file_path, train_ratio=0.8):
        dataset = Sample_Data(file_path)
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset=dataset,
            lengths=[train_size, test_size],
            generator=torch.Generator().manual_seed(0)
        )
        self.train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
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

        padding = 90 - batch_state.size(1)
        batch_state = F.pad(batch_state, (0, 0, 0, padding))

        # next_state变成fs且可以batch
        for s in next_state:
            fs = self.state_feature(s)
            T, _ = fs.shape
            batch_next_state_Seq.append(T)  # 记录每个state的序列长度，方便后面取argmaxQ(s, a)    相当于T x 1取前t个值中的最大值
            batch_next_state.append(fs)
        batch_next_state = pad_sequence(batch_next_state, batch_first=True)  # B x T x D
        padding = 90 - batch_next_state.size(1)
        batch_next_state = F.pad(batch_next_state, (0, 0, 0, padding))

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

    def choose_action(self, s):
        # print(s)
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.actor_eval(s)[0].detach()

    # def choose_action(self, qsa_eval, state_Seq, k=1):
    #     """
    #     按照DQN进行决策，选择使得Q(s, a)最大的action
    #     :param q_eval: B x T
    #     :param state_Seq: [100, 200, 43, 54, ....]   len = B
    #     :return: action B x topk
    #     """
    #     action = torch.zeros((len(state_Seq), k))
    #     for i, T in enumerate(state_Seq):
    #         res = qsa_eval[i, : T].topk(k, dim=0)[1]  # k
    #         action[i] = res
    #     return action.to(self.device)

    @torch.no_grad()
    def eval(self, state, action, reward, next_state):
        # 每个epoch训练完以后，评估一下DQN什么情况，reward和loss，顺便和random策略比较
        # state, action, reward, next_state = next(iter(self.dataloader))

        batch_state, _, batch_action, \
        batch_reward, batch_next_state, batch_next_state_Seq \
            = self.get_batch(state, action, reward, next_state)
        batch_s = torch.FloatTensor(batch_state)
        batch_a = torch.FloatTensor(batch_state[:, :, :76])
        batch_r = torch.FloatTensor(batch_reward.float())
        batch_s_ = torch.FloatTensor(batch_next_state)
        a = self.actor_eval(batch_s)
        q = self.critic_eval(batch_s, a)
        actor_loss = -torch.mean(q)

        # compute the target Q value using the information of next state
        a_target = self.actor_target(batch_s_)
        q_tmp = self.critic_target(batch_s_, a_target)
        q_target = (batch_r + GAMMA * q_tmp.transpose(0, 1)).transpose(0, 1)
        # compute the current q value and the loss
        q_eval = self.critic_eval(batch_s, batch_a)
        td_error = self.loss_func(q_target, q_eval)

        # # 计算random_action的reward
        # random_policy = self.random_action(batch_state_Seq)
        # # 计算choose_action的reward
        # optimal_policy = self.choose_action(qsa_eval, batch_state_Seq)
        #
        # # batch_action是ground-truth action
        # random_reward = torch.sum(random_policy == batch_action).item()
        # DQN_reward = torch.sum(optimal_policy == batch_action).item()

        # print("loss: {}, random_reward: {}, DQN_reward: {}".format(loss, random_reward, DQN_reward))

        # return random_reward, DQN_reward, loss
        return td_error, actor_loss

    def learn(self, bar):
        # softly update the target networks
        # for x in self.actor_target.state_dict().keys():
        #     eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        # for x in self.critic_target.state_dict().keys():
        #     eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
        #     eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')
        # # sample from buffer a mini-batch data
        # for batch_data in self.dataloader:
        #     state, action, reward, next_state = batch_data
        #     batch_state, _, batch_action, \
        #     batch_reward, batch_next_state, batch_next_state_Seq \
        #         = self.get_batch(state, action, reward, next_state)
        for iteration, batch_data in enumerate(bar):
            for x in self.actor_target.state_dict().keys():
                eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
                eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
            for x in self.critic_target.state_dict().keys():
                eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
                eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')

            state, action, reward, next_state = batch_data
            if iteration % 10 == 0:
                td_error_eval, actor_loss_eval = self.eval(state, action, reward, next_state)
            else:
                bar.set_postfix(td_error_eval=td_error_eval.item(), actor_loss_eval=actor_loss_eval.item())


            # if iteration % 100 == 0:
            #     random_reward_eval, DQN_reward_eval, loss_eval = self.eval(state, action, reward, next_state)
            # else:
            #     bar.set_postfix(loss_eval=loss_eval.item(), random_reward=random_reward_eval,
            #                     DQN_reward=DQN_reward_eval)

            batch_state, _, batch_action, \
            batch_reward, batch_next_state, batch_next_state_Seq \
                = self.get_batch(state, action, reward, next_state)
            # extract data from mini-batch of transitions including s, a, r, s_
            batch_s = torch.FloatTensor(batch_state)
            # batch_a = torch.FloatTensor(batch_action)
            batch_a = torch.FloatTensor(batch_state[:,: ,:76])
            batch_r = torch.FloatTensor(batch_reward.float())
            batch_s_ = torch.FloatTensor(batch_next_state)
            # print(batch_s.shape)
            # print(batch_s_.shape)
            # batch_s = torch.FloatTensor(batch_trans[:, :self.s_dim])
            # batch_a = torch.FloatTensor(batch_trans[:, self.s_dim:self.s_dim + self.a_dim])
            # batch_r = torch.FloatTensor(batch_trans[:, -self.s_dim - 1: -self.s_dim])
            # batch_s_ = torch.FloatTensor(batch_trans[:, -self.s_dim:])
            # make action and evaluate its action values
            a = self.actor_eval(batch_s)
            q = self.critic_eval(batch_s, a)
            actor_loss = -torch.mean(q)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # print("actor loss:", actor_loss.detach())

            # compute the target Q value using the information of next state

            a_target = self.actor_target(batch_s_)
            q_tmp = self.critic_target(batch_s_, a_target)
            q_target = (batch_r + GAMMA * q_tmp.transpose(0,1)).transpose(0,1)
            q_eval = self.critic_eval(batch_s, batch_a)
            td_error = self.loss_func(q_target, q_eval)
            self.critic_optimizer.zero_grad()
            td_error.backward()
            self.critic_optimizer.step()
            # print("critic loss:", td_error.detach())

            bar.set_postfix(td_error=td_error.item(), actor_loss=actor_loss.item())


############################### Training ######################################
# Define the env in gym
# env = gym.make(ENV_NAME)
# env = env.unwrapped
# env.seed(1)
# s_dim = 152
# a_dim = 76
# a_bound = env.action_space.high
# a_low_bound = env.action_space.low
s_dim = 152
a_dim = 76
model = DDPG(a_dim, s_dim)
# t1 = time.time()
every_save=1
save_model_path='checkpoint'
for e in range(EPOCH):

    train_bar = tqdm(model.train_dataloader, desc="Training", total=len(model.train_dataloader))
    train_bar.set_description(f'Training Epoch [{e}/{EPOCH}]\n')
    model.learn(train_bar)
    if e % every_save == 0:
        torch.save(model.actor_eval.state_dict(), save_model_path + "/actor_eval_%d.pkl" % e)
        torch.save(model.actor_target.state_dict(), save_model_path + "/actor_target_%d.pkl" % e)
        torch.save(model.critic_eval.state_dict(), save_model_path + "/critic_eval_%d.pkl" % e)
        torch.save(model.critic_target.state_dict(), save_model_path + "/critic_target_%d.pkl" % e)
    # model.eval()
# print('Running time: ', time.time() - t1)


# var = 3  # the controller of exploration which will decay during training process
#
# for i in range(EPISODES):
#     s = env.reset()
#     ep_r = 0
#     for j in range(EP_STEPS):
#         # if RENDER: env.render()
#         # # add explorative noise to action
#         # a = ddpg.choose_action(s)
#         # a = np.clip(np.random.normal(a, var), a_low_bound, a_bound)
#         # s_, r, done, info = env.step(a)
#         # ddpg.store_transition(s, a, r / 10, s_)  # store the transition to memory
#
#         if ddpg.pointer > MEMORY_CAPACITY:
#             var *= 0.9995  # decay the exploration controller factor
#             ddpg.learn()
#
#         s = s_
#         ep_r += r
#         if j == EP_STEPS - 1:
#             print('Episode: ', i, ' Reward: %i' % (ep_r), 'Explore: %.2f' % var)
#             if ep_r > -300: RENDER = True
#             break



