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

# hyper parameters
TASK_FEATURE = 76
EPSILON = 0.9
GAMMA = 0.9
LR = 0.01
EPOCH = 5
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
BATCH_SIZE = 32

EPISODES = 40


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
    def __init__(self):
        self.eval_net, self.target_net = Net(TASK_FEATURE * 2), Net(TASK_FEATURE * 2)
        self.optimizer = optim.Adam(self.eval_net.parameters(), LR)
        self.loss = nn.MSELoss()

        self.load_data()

    def load_data(self):
        dataset = Sample_Data('./train/worker_7945.csv')
        self.dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
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

    def random_action(self, state_Seq):
        """
        随机选择action
        :param state_Seq: [56, 90, 20, 40, ...]
        :return: action B x 1
        """
        action = torch.zeros(len(state_Seq))
        for i, T in enumerate(state_Seq):
            action[i] = int(np.random.randint(0, T, 1))
        return torch.unsqueeze(action, dim=1)

    def choose_action(self, qsa_eval, state_Seq):
        """
        按照DQN进行决策，选择使得Q(s, a)最大的action
        :param q_eval: B x T
        :param state_Seq: [100, 200, 43, 54, ....]   len = B
        :return: action B x 1
        """
        action = torch.zeros(len(state_Seq))
        for i, T in enumerate(state_Seq):
            action[i] = torch.argmax(qsa_eval[0, : T])
        return torch.unsqueeze(action, dim=1)

    @torch.no_grad()
    def eval(self):
        # 每个epoch训练完以后，评估一下DQN什么情况，reward和loss，顺便和random策略比较
        state, action, reward, next_state = next(iter(self.dataloader))

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
        random_reward = torch.sum(optimal_policy == batch_action).item()
        DQN_reward = torch.sum(random_policy == batch_action).item()

        print("loss: {}, random_reward: {}, DQN_reward: {}".format(loss, random_reward, DQN_reward))

    def learn(self):
        for batch_data in self.dataloader:
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
    model = DQN()
    for e in range(EPOCH):
        print("-------%d-------" % e)
        model.learn()
        model.eval()


if __name__ == '__main__':
    main()
    # data = Sample_Data('./train')