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

class Sample_Data(Dataset):
    def __init__(self, file_path, worker_num=None):
        try:
            self.data = pd.read_csv(file_path)
        except:
            if not worker_num:
                worker_list = os.listdir(file_path)
            else:
                worker_list = np.random.choice(os.listdir(file_path), size=worker_num)
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