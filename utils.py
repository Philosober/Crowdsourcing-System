import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os

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