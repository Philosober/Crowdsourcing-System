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
from utils import Sample_Data

class Basic_model:
    def __init__(self, batch_size, task_feature):
        self.batch_size = batch_size
        self.task_f = task_feature

    def load_data(self, file_path, train_ratio=0.8, worker_num=None):
        dataset = Sample_Data(file_path, worker_num)
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset=dataset,
            lengths=[train_size, test_size],
            generator=torch.Generator().manual_seed(0)
        )
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        self.project_info = pd.read_csv("data/project_info.csv",
                                        usecols=['project_id', 'project_feature'],
                                        index_col='project_id')

    def load_data_1(self, file_path):
        worker_list = os.listdir(file_path)
        self.worker_dataloader = {}
        for worker in worker_list:
            try:
                dataset = Sample_Data(file_path + '/' + worker)
                dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                self.worker_dataloader[worker] = dataloader
            except:
                continue
        self.project_info = pd.read_csv("data/project_info.csv",
                                        usecols=['project_id', 'project_feature'],
                                        index_col='project_id')

    def task_feature(self, task_id):
        res = self.project_info.loc[task_id]
        res = np.array(eval(res[0]))
        return torch.from_numpy(res)

    def worker_feature(self, completed_work):
        res = torch.zeros(self.task_f)
        for task in completed_work:
            res += self.task_feature(task)
        if completed_work:
            res /= len(completed_work)
        return res  # 76

    def state_feature(self, state):
        """

        :param state: "[[100, 101], [102, 103, 105]]"
        :return:state_feature
        """
        if type(state) is not list:
            state = list(eval(state))
        task_pool = state[1]
        completed_task = state[0]
        T = len(state[1])  # 任务池中任务的个数
        D = self.task_f * 2  # [ft, fw]
        res = torch.zeros((T, D))
        # 构造fw(通过state[0])
        fw = self.worker_feature(completed_task)  # 76
        res[:, : self.task_f] = fw
        # 构造ft(通过state[1][j])
        for j, task in enumerate(task_pool):
            ft = self.task_feature(task)
            res[j, self.task_f:] = ft

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


