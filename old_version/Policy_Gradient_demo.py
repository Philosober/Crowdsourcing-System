import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import functional as F
from Agent import Worker_Agent
from torch.optim import adam
from torch.distributions import Categorical
from Network import REINFORCE_Net as Net

torch.manual_seed(1)

#Hyperparameters
learning_rate = 0.02
gamma = 0.995
episodes = 1000
TASK_FEATURE = 76

eps = np.finfo(np.float32).eps.item()

policy = Net(TASK_FEATURE)
optimizer = adam.Adam(policy.parameters(), lr=learning_rate)


def selct_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    c = Categorical(probs)
    action = c.sample()


    policy.saved_log_probs.append(c.log_prob(action))
    action = action.item()
    return action

def finish_episode():
    R = 0
    policy_loss = []
    rewards = []

    for r in policy.rewards[::-1]:
        # 从前往后计算
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Formalize reward
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean())/(rewards.std() + eps)

    # get loss
    for reward, log_prob in zip(rewards, policy.saved_log_probs):
        policy_loss.append(-log_prob * reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()



    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    agent = Worker_Agent("./train", 7945)
    # running_reward = 0
    # steps = []
    for i in range(1000):
        # 采样1000轮
        agent.restart()

        while agent.cur_step <= agent.max_step:
            # 利用policy_net选择action
            action = selct_action(agent.state)
            # 根据action去sample data
            reward = agent.sample_step(action)
            # next_state, reward, terminated, truncated, info = env.step(action)
            policy.rewards.append(reward)    # 记录奖励值

            # if terminated or truncated:
            #     print("Episode {}, live time = {}".format(episode, t))
            #     steps.append(t)
            #     plot(steps)
            #     break
        if i % 50 == 0:
            torch.save(policy, './checkpoint/policyNet.pkl')

        # running_reward = running_reward * policy.gamma - t*0.01
        finish_episode()

if __name__ == '__main__':
    main()
