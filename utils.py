import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def get_batch(state, action, reward, next_state):
    batch_state = []
    batch_state_Seq = []  # list
    batch_action = []  # B x 1
    batch_next_state = []
    batch_next_state_Seq = []  # list
    batch_reward = torch.unsqueeze(reward, dim=1)

    # state变成fs且可以batch
    # action对应到序列中第几个vector
    for k, s in enumerate(state):
        fs = state_feature(s)
        T, _ = fs.shape
        batch_state.append(fs)
        batch_state_Seq.append(T)  # 记录每个state的序列长度，方便后面取argmaxQ(s, a)来进行决策
        batch_action.append(action_index(s, action[k]))  # 记录action对应序列中的哪个位置
    batch_state = pad_sequence(batch_state, batch_first=True)  # B x T x D
    # next_state变成fs且可以batch
    for s in next_state:
        fs = state_feature(s)
        T, _ = fs.shape
        batch_next_state_Seq.append(T)  # 记录每个state的序列长度，方便后面取argmaxQ(s, a)    相当于T x 1取前t个值中的最大值
        batch_next_state.append(fs)
    batch_next_state = pad_sequence(batch_next_state, batch_first=True)  # B x T x D

    return batch_state, batch_state_Seq, batch_action, batch_reward, batch_next_state, batch_next_state_Seq

def task_feature(task_id):
    res = project_info.loc[task_id]
    res = np.array(eval(res[0]))
    return torch.from_numpy(res)


def worker_feature(completed_work, TASK_FEATURE=76):
    res = torch.zeros(TASK_FEATURE)
    for task in completed_work:
        res += task_feature(task)
    if completed_work:
        res /= len(completed_work)
    return res  # 76

def action_index(state, action):
    """

    :param state: [[100, 101], [102, 103, 105]]
    :param action: 105
    :return: 2
    """
    state = list(eval(state))
    task_pool = state[1]
    return task_pool.index(action)

def state_feature(state, TASK_FEATURE=76):
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
    fw = worker_feature(completed_task)  # 76
    res[:, : TASK_FEATURE] = fw
    # 构造ft(通过state[1][j])
    for j, task in enumerate(task_pool):
        ft = task_feature(task)
        res[j, TASK_FEATURE:] = ft

    return res  # T x D