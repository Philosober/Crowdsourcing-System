import pandas as pd
import numpy as np
from ast import literal_eval
import torch


class Worker_Agent:
    def __init__(self, folder, worker, gamma=1, save_sample=False):
        self.real_world = pd.read_csv(folder + '/worker_{}.csv'.format(worker))
        # self.real_world = pd.read_csv('./test.csv')

        self.real_world['S'] = self.real_world['S'].apply(literal_eval)
        self.real_world["S'"] = self.real_world["S'"].apply(literal_eval)
        self.gamma = gamma
        self.save_sample = save_sample

    def restart(self):
        self.max_step = len(self.real_world) - 1  # 记录最大回合数
        self.cur_step = 0  # 记录当前是第几个回合
        self.EPISODE_MEMORY = []  # 保存采样结果
        self.G = 0  # episode返回值
        # 初始状态
        try:
            self.state = self.real_world['S'].iloc[0]  # [[w_i], [task_pool]]
            self.optimal_action = self.real_world['A'].iloc[0]  # 当前应该选择的动作
            return True
        except:
            return False

    def sample_step(self, action):
        step_data = [self.state, action]

        if action == self.optimal_action:
            # 更新S
            self.state = self.update_state(True)
            self.optimal_action = self.update_optimal_action(True)
            reward = 1
        else:
            self.state = self.update_state(False)
            self.optimal_action = self.update_optimal_action(False)
            reward = 0

        step_data.append(reward)
        step_data.append(self.state)

        if not self.optimal_action:
            try:
                self.optimal_action = self.real_world["A"].iloc[self.cur_step + 1]
            except:
                pass

        # 计算return值
        self.G += np.power(self.gamma, self.cur_step) * reward
        # 保存sample数据
        if self.save_sample:
            self.EPISODE_MEMORY.append(step_data)

        self.cur_step += 1

        return reward

    def update_optimal_action(self, done):
        real_completed_task = self.real_world["S'"].iloc[self.cur_step][0]
        cur = real_completed_task.index(self.optimal_action)
        if done:  # 如果做了optimal_action，则需要从下一个判断
            cur += 1
        if cur >= len(real_completed_task):
            # 此时选择self.cur_step + 1时的action为optimal action
            return None
        task_pool = self.real_world["S'"].iloc[self.cur_step][1]
        # 在real_completed_task中找到第一个在task_pool中的task
        while real_completed_task[cur] not in task_pool:
            cur += 1
            if cur >= len(real_completed_task):
                # 此时选择self.cur_step + 1时的action为optimal action
                return None
        return real_completed_task[cur]

    def update_state(self, done):
        state = []
        task_pool = self.real_world["S'"].iloc[self.cur_step][1]  # 任务池随着时间推进
        completed_task = self.state[0]
        if done:
            completed_task.append(self.optimal_action)
            state.append(completed_task)
            state.append(task_pool)
        else:
            state.append(completed_task)
            state.append(task_pool)

        return state

    def random_sample(self):
        # 每次从任务池中随机选择一个任务
        while self.cur_step <= self.max_step:
            action = np.random.choice(self.state[1])
            self.sample_step(action)
        print(self.G)

    def optimal_sample(self):
        while self.cur_step <= self.max_step:
            action = self.optimal_action
            self.sample_step(action)
        print(self.G)

    def dqn_sample(self, value_model):
        """

        :param model: instance of DQN class
        :return:
        """
        while self.cur_step <= self.max_step:
            # 通过DQN获取action
            task_pool = self.state[1]
            state = value_model.state_feature(self.state)  # T x D
            state = torch.unsqueeze(state, dim=0)
            qsa = value_model.eval_net(state)
            qsa = torch.squeeze(qsa)
            action = task_pool[torch.argmax(qsa)]
            self.sample_step(action)
        print(self.G)

    def policy_sample(self, policy_model):
        while self.cur_step <= self.max_step:
            # 利用policy_net选择action
            action = policy_model.select_action(self.state)
            # 根据action去sample data
            reward = self.sample_step(action)
            policy_model.policy_net.rewards.append(reward)    # 记录奖励值
        return self.G

if __name__ == "__main__":
    from Value_based import DQN
    from Policy_based import REINFORCE
    # agent1 = Worker_Agent(2459641)
    # if agent1.restart():
    #     agent1.random_sample()
    # else:
    #     print("没有数据，无法采样")

    agent2 = Worker_Agent("./test", 7945)
    agent2.restart()
    agent2.optimal_sample()
    agent2.restart()
    agent2.random_sample()

    # dqn_model = DQN(32, 76)
    # dqn_model.load_DQN("./checkpoint/DQN_5.pkl")
    # agent2.restart()
    # agent2.dqn_sample(dqn_model)


    # agent.sample_step(100)
    # agent.sample_step(101)
    # agent.sample_step(102)
