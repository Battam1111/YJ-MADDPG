import numpy as np
from torch import Tensor
from torch.autograd import Variable
import torch

class ReplayBuffer(object):
    """
    多智能体强化学习的经验重放缓冲区，用于存储和采样经验。
    """

    def __init__(self, max_steps, num_agents, obs_dims, ac_dims, agent_types):
        """
        初始化ReplayBuffer类。

        参数:
            max_steps (int): 缓冲区中可存储的最大时间步数。
            num_agents (int): 环境中智能体的数量。
            obs_dims (list of ints): 每个智能体的观察维度。
            ac_dims (list of ints): 每个智能体的动作维度。
        """
        self.max_steps = max_steps  # 缓冲区的最大容量
        self.num_agents = num_agents  # 智能体数量
        self.agent_types = agent_types

        # 初始化各个缓冲区，用于存储智能体的观测、动作、奖励、下一步观测和结束标志
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        

        # 为每个智能体分别创建对应的缓冲区
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float32))  # 存储观测
            self.ac_buffs.append(np.zeros((max_steps, adim), dtype=np.float32))   # 存储动作
            self.rew_buffs.append(np.zeros(max_steps, dtype=np.float32))          # 存储奖励
            self.next_obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float32))  # 存储下一步观测
            self.done_buffs.append(np.zeros(max_steps, dtype=np.uint8))           # 存储是否终止

        self.filled_i = 0  # 缓冲区中当前存储的步数（当缓冲区已满时为最大值）
        self.curr_i = 0    # 当前写入数据的位置索引（当缓冲区满时会覆盖最旧的数据）

    def __len__(self):
        """
        返回缓冲区中存储的有效步数。
        """
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        """
        将新的经验推入缓冲区。

        参数:
            observations (np.array): 当前的观测，形状为 (n_agents, obs_dim)。
            actions (np.array): 执行的动作，形状为 (n_agents, ac_dim)。
            rewards (np.array): 收到的奖励，形状为 (n_agents,)。
            next_observations (np.array): 下一步的观测，形状为 (n_agents, obs_dim)。
            dones (np.array): 是否结束，形状为 (n_agents,)。
        """
        nentries = observations.shape[0]  # 当前推入的经验数（通常为并行环境的数量）

        # 如果缓冲区即将溢出，则将旧数据滚动以释放空间
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i  # 需要滚动的步数
            for agent_i in range(self.num_agents):
                # 将旧数据滚动
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i], rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i], rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i], rollover)
                self.next_obs_buffs[agent_i] = np.roll(self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i], rollover)
            self.curr_i = 0  # 将当前索引重置为0（开始覆盖旧数据）
            self.filled_i = self.max_steps  # 缓冲区已满

        # 将新经验推入缓冲区
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                observations[:, agent_i])
            # 动作数据已经按智能体批处理，因此索引方式不同
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[agent_i]  # 使用一维索引
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(
                next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[agent_i]  # 使用一维索引

        # 更新索引和存储的步数
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=True):
        """
        从缓冲区中随机采样一批经验，并按智能体类型分割。

        参数:
            N (int): 要采样的经验数量。
            to_gpu (bool): 是否将采样的数据移动到GPU。
            norm_rews (bool): 是否对奖励进行标准化。
            
        返回:
            dict: 以智能体类型为键，值为对应智能体的数据元组 (obs, acs, rews, next_obs, dones)。
        """
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=True)  # 随机选择索引

        # 根据是否移动到GPU，定义数据类型转换函数
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        
        # 采样后的数据按智能体类型组织
        sampled_data = {}
        
        for agent_i in range(self.num_agents):
            obs = cast(self.obs_buffs[agent_i][inds])
            acs = cast(self.ac_buffs[agent_i][inds])
            rews = cast(self.rew_buffs[agent_i][inds])
            next_obs = cast(self.next_obs_buffs[agent_i][inds])
            dones = cast(self.done_buffs[agent_i][inds])
            
            if norm_rews:
                rews = (rews - self.rew_buffs[agent_i][:self.filled_i].mean()) / self.rew_buffs[agent_i][:self.filled_i].std()

            # 确保采样的数据按智能体类型分组
            agent_type = self.agent_types[agent_i]
            if agent_type not in sampled_data:
                sampled_data[agent_type] = ([], [], [], [], [])
            sampled_data[agent_type][0].append(obs)
            sampled_data[agent_type][1].append(acs)
            sampled_data[agent_type][2].append(rews)
            sampled_data[agent_type][3].append(next_obs)
            sampled_data[agent_type][4].append(dones)
        
        # 将列表转为Tensor
        for agent_type in sampled_data:
            sampled_data[agent_type] = tuple(map(lambda x: torch.stack(x), sampled_data[agent_type]))

        return sampled_data

    def get_average_rewards(self, N):
        """
        获取缓冲区中最后N个时间步的平均奖励。

        参数:
            N (int): 时间步数量。

        返回:
            list: 每个智能体的平均奖励。
        """
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # 允许负索引以处理滚动后的数据
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)

        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]
