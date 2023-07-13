import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from Network import REINFORCE_Net as Net
import os
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.utils.data import random_split
from Crowdsource import Basic_model
from torch.distributions import Categorical
from Agent import Worker_Agent
from matplotlib import pyplot as plt

# hyper parameters
TASK_FEATURE = 76
GAMMA = 1
LR = 0.000001
EPISODE = 100
TEST_EPISODE = 10
# SAVE_EPISODE = 1
eps = np.finfo(np.float32).eps.item()


class REINFORCE(Basic_model):
    def __init__(self, batch_size, task_feature, train_mode=True, file_path='./data/train/worker_7945.csv'):
        super(REINFORCE, self).__init__(batch_size, task_feature)
        self.policy_net = Net(TASK_FEATURE * 2)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_net.to(device=self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), LR)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.5)
        self.loss = nn.MSELoss()

        if train_mode:
            self.load_data(file_path)

    def load_model(self, load_path):
        state_dict = torch.load(load_path)
        self.policy_net.load_state_dict(state_dict)

    def select_action(self, state):
        task_pool = state[1]
        state = self.state_feature(state)  # T x D
        state = torch.unsqueeze(state, dim=0)  # 1 x T x D
        probs = self.policy_net(state)  # 1 x T x 1
        probs = torch.squeeze(probs)
        c = Categorical(probs)
        action = c.sample()

        self.policy_net.saved_log_probs.append(c.log_prob(action))
        action = action.item()
        return task_pool[action]

    def finish_episode(self):
        R = 0
        policy_loss = []
        rewards = []

        for r in self.policy_net.rewards[::-1]:
            # 从前往后计算
            R = r + self.policy_net.gamma * R
            rewards.insert(0, R)

        # Formalize reward
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

        # get loss
        for reward, log_prob in zip(rewards, self.policy_net.saved_log_probs):
            policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()
        policy_loss = sum(policy_loss)
        policy_loss.backward()
        self.optimizer.step()

        del self.policy_net.rewards[:]
        del self.policy_net.saved_log_probs[:]


def plot(episode_return):
    plt.plot(episode_return)
    plt.show()



def train(episodes):
    agent = Worker_Agent("./data/train", 7945, gamma=GAMMA)

    policy_model = REINFORCE(32, 76)
    episode_return = []
    max_return = 0
    for e in range(episodes):
        agent.restart()
        return_value = agent.policy_sample(policy_model)
        policy_model.finish_episode()
        episode_return.append(return_value)
        if return_value > max_return:
            torch.save(policy_model.policy_net, "./checkpoint/best_policy.pkl")
            max_return = return_value
        print(return_value)

    plot(episode_return)



def main():
    # train
    train(EPISODE)
    # test
    # test()


if __name__ == '__main__':
    main()
    # data = Sample_Data('./train')
