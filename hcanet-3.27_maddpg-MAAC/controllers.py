import random
from abc import ABC
from datetime import datetime
from enum import Enum
from os.path import isfile
from os import path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
from torch import optim
from torch.cuda import device as torch_device
from torch_geometric.data import Batch, Data, DataLoader

from rl.memory import ReplayMemory
# from nn.nets import HeteroMAGNet
from nn.modules.action import HeteroMAGNetActionLayer
from nn.modules.graph import GATModule
from rl.epsilon_schedules import DecayThenFlatSchedule
from rl.replay import ReplayBuffer, Transition
from training import TrainingConfig
from nn.nets import MADDPGAgent
from env.ou_noise import OUNoise
from utils.utils import get_load_path
from itertools import chain

class BaseController(ABC):
    """
    基类，用于控制器的基础架构
    """
    def __init__(self, node_types, agent_types, features_by_node_type, actions_by_node_type, device: torch_device):
        """
        初始化控制器基本属性

        参数:
          node_types: 节点类型列表
          agent_types: 智能体类型列表
          features_by_node_type: 每种节点类型的特征数量
          actions_by_node_type: 每种节点类型的动作数量
          device: 设备（如GPU）
        """
        self.num_unit_types = len(set(node_types))
        assert self.num_unit_types == len(features_by_node_type) == len(actions_by_node_type)
        assert all([at in node_types for at in agent_types])

        self.device = device
        self.agent_types = torch.tensor(agent_types, dtype=torch.long, device=self.device)
        self.n_agents = sum(node_types.count(agent_type) for agent_type in agent_types)
        self.n_actions_agents = [actions_by_node_type[nt] for nt in node_types if nt in agent_types]

    def act(self, *args) -> torch.tensor:
        """
        抽象方法，必须在子类中实现
        """
        raise NotImplementedError

    @staticmethod
    def _action_lists_to_tensors(actions: List[list]):
        """
        将动作列表转换为张量

        参数:
            actions: 一个包含多个动作列表的列表

        返回:
            将每个动作列表转换为 PyTorch 张量
        """
        for i in range(len(actions)):
            actions[i] = torch.tensor(actions[i], dtype=torch.bool)
            while len(actions[i].size()) < 1:
                actions[i].unsqueeze_(-1)

    @staticmethod
    def random_policy(valid_actions: list, device: torch_device) -> torch.tensor:
        """
        随机策略选择函数，从有效动作中随机选择一个

        参数:
            valid_actions: 每个智能体有效动作的布尔掩码列表
            device: 设备（如GPU）

        返回:
            actions: 每个智能体选择的动作索引组成的一维张量
        """
        n_agents = len(valid_actions)
        actions = torch.empty(n_agents, dtype=torch.int, device=device)
        for agent_id in range(n_agents):
            av_act = valid_actions[agent_id].nonzero().squeeze()
            if len(av_act.shape) == 0:
                actions[agent_id] = av_act
            else:
                chosen_action = torch.randint(av_act.shape[0], (1, ))
                actions[agent_id] = av_act[chosen_action]
        return actions

class MADDPGController(ABC):
    def __init__(self,
                 checkpoint_file: str,
                 checkpoint_dir: str,
                 optimizer: str,
                 critic_lr: float,
                 actor_lr: float,
                 weight_decay: float,
                 rmsprop_alpha: float,
                 rmsprop_eps: float,
                 num_UAVAgents,
                 num_chargerAgents,
                 node_types,
                 dim_obs_list: list, 
                 dim_act_list: list,
                 gamma,
                 tau,
                 device,
                 resume_run,
                 memory_size):
        """
        初始化多智能体深度确定性策略梯度（MADDPG）控制器

        参数:
            checkpoint_file: 检查点根目录路径
            checkpoint_dir: 检查点目录（保存检查点时使用）
            optimizer: 优化器类型（例如 'rmsprop' 或 'adam'）
            critic_lr: 评价网络学习率
            actor_lr: 策略网络学习率
            weight_decay: 权重衰减系数
            rmsprop_alpha: RMSprop 优化器 alpha 参数
            rmsprop_eps: RMSprop 优化器 epsilon 参数
            num_UAVAgents: UAV 智能体数量
            num_chargerAgents: 充电器智能体数量
            node_types: 节点类型列表
            dim_obs_list: 每种智能体的状态维度列表
            dim_act_list: 每种智能体的动作维度列表
            gamma: 折扣因子
            tau: 软更新系数
            device: 设备（例如 GPU）
            resume_run: 是否从检查点恢复
            memory_size: 经验回放缓冲区大小
        """
        super().__init__()
        self.checkpoint = checkpoint_dir
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.tau = tau
        self.weight_decay = weight_decay
        self.num_UAVAgents = num_UAVAgents
        self.num_chargerAgents = num_chargerAgents
        self.dim_UAVactions = dim_act_list[0]
        self.dim_UAVobs = dim_obs_list[0]
        self.dim_chargerobs = dim_obs_list[1]
        self.device = device
        self.OUNoise = OUNoise(self.dim_UAVactions)
        self.var = [1. for _ in range(num_UAVAgents + num_chargerAgents)]

        # 计算所有智能体状态与动作维度总和（供 critic 使用）
        self.dim_all_obs = self.dim_UAVobs * self.num_UAVAgents + self.dim_chargerobs * self.num_chargerAgents
        dim_all_act = self.dim_UAVactions * (self.num_UAVAgents + self.num_chargerAgents)

        # 初始化 UAV 和 Charger 智能体
        self.UAVAgent = MADDPGAgent(0, 
                                    dim_obs_list, 
                                    dim_act_list,
                                    num_UAVAgents,
                                    self.dim_all_obs,
                                    dim_all_act,
                                    device)
        self.chargerAgent = MADDPGAgent(1,
                                        dim_obs_list,
                                        dim_act_list,
                                        num_chargerAgents,
                                        self.dim_all_obs,
                                        dim_all_act,
                                        device)
      
        # 初始化经验回放缓冲区
        self.memory = ReplayMemory(memory_size, self.num_UAVAgents + self.num_chargerAgents, True, self.dim_UAVobs, self.dim_UAVactions, True, 3, 0.95)

        # 根据优化器类型初始化优化器
        if optimizer == 'rmsprop':
            one_critic_param = []
            for i in range(self.num_UAVAgents):
                one_critic_param += list(self.UAVAgent.critic[i].parameters())
            self.critic_optimizer = optim.RMSprop(one_critic_param,
                                                  lr=self.critic_lr,
                                                  alpha=rmsprop_alpha,
                                                  eps=rmsprop_eps,
                                                  weight_decay=self.weight_decay)
            one_actor_param = []
            for i in range(self.num_UAVAgents):
                one_actor_param += list(self.UAVAgent.actor[i].parameters())
            self.actor_optimizer = optim.RMSprop(one_actor_param,
                                                 lr=self.actor_lr,
                                                 alpha=rmsprop_alpha,
                                                 eps=rmsprop_eps,
                                                 weight_decay=self.weight_decay)
        elif optimizer == 'adam':
            one = []
            for i in range(self.num_UAVAgents):
                one += list(self.UAVAgent.critic[i].parameters())
            self.UAV_critic_optimizer = optim.Adam(one, lr=self.critic_lr)
            para = []
            for i in range(self.num_chargerAgents):
                para += list(self.chargerAgent.critic[i].parameters())
            self.charger_critic_optimizer = optim.Adam(para, lr=self.critic_lr)
            one_actor_param = []
            for i in range(self.num_UAVAgents):
                one_actor_param += list(self.UAVAgent.actor[i].parameters())
            self.UAV_actor_optimizer = optim.Adam(one_actor_param, lr=self.actor_lr)
            param = []
            for i in range(self.num_chargerAgents):
                param += list(self.chargerAgent.actor[i].parameters())
            self.charger_actor_optimizer = optim.Adam(param, lr=self.actor_lr)
        else:
            raise ValueError('Invalid optimizer "{}"'.format(optimizer))

        # 如果恢复训练，则加载检查点
        if resume_run:
            load_path = get_load_path(checkpoint_file)
            print('Loading from checkpoint file:', load_path)
            checkpoint = torch.load(load_path, map_location=self.device)
            # 尝试加载 UAVAgent.actor[0] 的参数
            if 'actor1_state_dict' in checkpoint:
                self.UAVAgent.actor[0].load_state_dict(checkpoint['actor1_state_dict'])
            else:
                print("Warning: 'actor1_state_dict' not found in checkpoint; skipping load for UAVAgent.actor[0]")
            # 尝试加载 UAVAgent.actor[1] 的参数
            if 'actor2_state_dict' in checkpoint:
                self.UAVAgent.actor[1].load_state_dict(checkpoint['actor2_state_dict'])
            else:
                print("Warning: 'actor2_state_dict' not found in checkpoint; skipping load for UAVAgent.actor[1]")
            # 尝试加载 chargerAgent.actor[0] 的参数
            if 'charger_actor1_state_dict' in checkpoint:
                self.chargerAgent.actor[0].load_state_dict(checkpoint['charger_actor1_state_dict'])
            else:
                print("Warning: 'charger_actor1_state_dict' not found in checkpoint; skipping load for chargerAgent.actor[0]")

    def get_net_state_dicts(self):
        """
        获取当前网络状态字典

        返回:
            dict: 包含 UAV 与 Charger 各 actor 的状态字典
        """
        return {
            'actor1_state_dict': self.UAVAgent.actor[0].state_dict(),
            'actor2_state_dict': self.UAVAgent.actor[1].state_dict(),
            'charger_actor1_state_dict': self.chargerAgent.actor[0].state_dict()
        }
    
    def save_checkpoint(self, step_num, n_episodes):
        """
        保存检查点

        参数:
            step_num: 当前步数
            n_episodes: 当前 episode 编号
        """
        print('Saving checkpoint...')
        net_weights = self.get_net_state_dicts()
        variables = {**net_weights}
        cur_checkpointpath = path.join(self.checkpoint, 'model_{}.pt'.format(n_episodes))
        torch.save(variables, cur_checkpointpath)

    def act(self, state_batch, episode_num, episode_before_train, if_noise):
        """
        根据当前状态选择动作

        参数:
            state_batch: 当前状态张量，维度 [batch_size, n_agents, dim_obs]
            episode_num: 当前 episode 编号
            episode_before_train: 训练前的 episode 数量
            if_noise: 是否加入探索噪声

        返回:
            actions: 选择的动作张量，维度 [n_agents, dim_act]
        """
        actions = torch.zeros((self.num_UAVAgents+self.num_chargerAgents, self.dim_UAVactions), dtype=torch.float32, device=self.device)
        if state_batch.ndim != 3:
            state_batch = state_batch.unsqueeze(0)
        if if_noise:
            for i in range(self.num_UAVAgents):
                curr_act = self.UAVAgent.actor[i](state_batch[:, i, :])
                curr_act += (self.OUNoise.noise() * self.var[i]).type(torch.float32).to(self.device)
                if episode_num > episode_before_train and self.var[i] > 0.05:
                    self.var[i] *= 0.9999997
                actions[i] = torch.clamp(curr_act, -1.0, 1.0)
            for i in range(self.num_chargerAgents):
                curr_act = self.chargerAgent.actor[i](state_batch[:, i+self.num_UAVAgents, :self.dim_chargerobs])
                curr_act += (self.OUNoise.noise() * self.var[i+self.num_UAVAgents]).type(torch.float32).to(self.device)
                if episode_num > episode_before_train and self.var[i+self.num_UAVAgents] > 0.05:
                    self.var[i+self.num_UAVAgents] *= 0.9999997
                actions[i+self.num_UAVAgents] = torch.clamp(curr_act, -1.0, 1.0)
        else:
            for i in range(self.num_UAVAgents):
                curr_act = self.UAVAgent.actor[i](state_batch[:, i, :])
                actions[i] = torch.clamp(curr_act, -1.0, 1.0)
            for i in range(self.num_chargerAgents):
                curr_act = self.chargerAgent.actor[i](state_batch[:, i+self.num_UAVAgents, :self.dim_chargerobs])
                actions[i+self.num_UAVAgents] = torch.clamp(curr_act, -1.0, 1.0)
        return actions

    def shared_parameters(self):
        """
        获取所有共享参数
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        缩放共享参数的梯度，使得共享参数的梯度更新更平衡
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.num_UAVAgents)

    def choose_neighbor_for_actor(self, state_batch, adj_batch, index):
        """
        为 actor 选择邻居

        参数:
            state_batch: 状态张量，维度 [batch_size, n_agents, dim_obs]
            adj_batch: 邻接矩阵张量
            index: 当前智能体的索引

        返回:
            合并当前智能体状态与其邻居状态后的张量
        """
        if state_batch.ndim != 3:
            state_batch = state_batch.unsqueeze(0)
        if adj_batch.ndim != 2:
            adj_batch = adj_batch.unsqueeze(0)
        bs = state_batch.shape[0]
        neighbor = adj_batch.shape[1]
        x = state_batch[:, index, :].unsqueeze(1)
        y = torch.zeros((bs, neighbor, state_batch.shape[2]), dtype=torch.float32, device=self.device)
        for i in range(bs):
            y[i] = state_batch[i, adj_batch[i].squeeze(), :]
        return torch.cat((x, y), dim=1)
      
    def update(self, i_step, param_dict: dict):
        """
        更新策略与评价网络

        参数:
            i_step: 当前步数
            param_dict: 参数字典

        返回:
            q_loss, policy_loss: 当前步的 critic 与 actor 损失
        """
        for i in range(self.num_UAVAgents):
            self.UAVAgent.actor[i].train()
            self.UAVAgent.critic[i].train()
        for i in range(self.num_chargerAgents):
            self.chargerAgent.actor[i].train()
            self.chargerAgent.critic[i].train()

        self.curr_index_agent = i_step % (self.num_UAVAgents + self.num_chargerAgents)

        (state_batch, action_batch, next_state_batch, 
         reward_batch, done_batch, is_weight_batch,
         n_step_batch) = self.memory.sample_TO(param_dict["BATCH_SIZE"], self.curr_index_agent)

        state_batch = state_batch.cuda()
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()
        done_batch = done_batch.cuda()
        is_weight_batch = is_weight_batch.cuda()
        n_step_batch = n_step_batch.cuda()
        next_state_batch = next_state_batch.cuda()

        target_next_actions_batch = torch.zeros((param_dict["BATCH_SIZE"], self.num_UAVAgents+self.num_chargerAgents, self.dim_UAVactions), device=self.device)
      
        q_loss = [0.]*2
        critic_loss = []
        abs_error = []
        beg = -self.memory.max_len
        end = (self.memory.now_len - self.memory.max_len) if (self.memory.now_len < self.memory.max_len) else None
      
        self.UAV_critic_optimizer.zero_grad()
        # Critic 网络更新 - UAV部分
        for i in range(self.num_UAVAgents):
            target_next_actions_batch[:, i, :] = self.UAVAgent.target_actor[i](next_state_batch[:, i, :]).detach()

        # Critic 网络更新 - Charger部分
        for i in range(self.num_chargerAgents):
            target_next_actions_batch[:, i+self.num_UAVAgents, :] = self.chargerAgent.target_actor[i](next_state_batch[:, i+self.num_UAVAgents, :self.dim_chargerobs]).detach()

        state_batch_flat = state_batch.view(state_batch.size()[0], -1)
        state_batch_flat = state_batch_flat[:, :self.dim_all_obs]
        for i in range(self.num_UAVAgents):
            predicted_q_values_batch = self.UAVAgent.critic[i](state_batch_flat, action_batch.view(action_batch.size()[0], -1))
            target_next_q_values_batch = self.UAVAgent.target_critic[i](next_state_batch.view(next_state_batch.size()[0], -1)[:, :self.dim_all_obs], target_next_actions_batch.view(target_next_actions_batch.size()[0], -1))
            target_q_values = reward_batch[:, i].unsqueeze(-1) + (1 - done_batch[:, i]).unsqueeze(-1) * pow(self.gamma, n_step_batch.unsqueeze(-1)) * target_next_q_values_batch
            if i == self.curr_index_agent:
                curr_q_loss = torch.mean(is_weight_batch * nn.MSELoss()(predicted_q_values_batch, target_q_values.detach()))
            else:
                self.memory.per_tree[i].indices = self.memory.per_tree[self.curr_index_agent].indices
                curr_q_loss = torch.mean(torch.tensor(self.memory.per_tree[i].get_is_weight_TO(beg, end), dtype=torch.float32, device=self.device) * nn.MSELoss()(predicted_q_values_batch, target_q_values.detach()))
            abs_error.append(torch.abs(predicted_q_values_batch.detach() - target_q_values.detach()))
            critic_loss.append(curr_q_loss)
            q_loss[0] += curr_q_loss
      
        q_loss[0].backward()
        torch.nn.utils.clip_grad_norm_(self.UAVAgent.critic[0].parameters(), 10 * self.num_UAVAgents)
        torch.nn.utils.clip_grad_norm_(self.UAVAgent.critic[1].parameters(), 10 * self.num_UAVAgents)
        self.UAV_critic_optimizer.step()
      
        self.charger_critic_optimizer.zero_grad()
        for i in range(self.num_chargerAgents):
            predicted_q_values_batch = self.chargerAgent.critic[i](state_batch_flat, action_batch.view(action_batch.size()[0], -1))
            target_next_q_values_batch = self.chargerAgent.target_critic[i](next_state_batch.view(next_state_batch.size()[0], -1)[:, :self.dim_all_obs], target_next_actions_batch.view(target_next_actions_batch.size()[0], -1))
            target_q_values = reward_batch[:, i+self.num_UAVAgents].unsqueeze(-1) + (1 - done_batch[:, i+self.num_UAVAgents]).unsqueeze(-1) * pow(self.gamma, n_step_batch.unsqueeze(-1)) * target_next_q_values_batch
            if i+self.num_UAVAgents == self.curr_index_agent:
                curr_q_loss = torch.mean(is_weight_batch * nn.MSELoss()(predicted_q_values_batch, target_q_values.detach()))
            else:
                self.memory.per_tree[i+self.num_UAVAgents].indices = self.memory.per_tree[self.curr_index_agent].indices
                curr_q_loss = torch.mean(torch.tensor(self.memory.per_tree[i+self.num_UAVAgents].get_is_weight_TO(beg, end), dtype=torch.float32, device=self.device) * nn.MSELoss()(predicted_q_values_batch, target_q_values.detach()))
            abs_error.append(torch.abs(predicted_q_values_batch.detach() - target_q_values.detach()))
            critic_loss.append(curr_q_loss)
            q_loss[1] += curr_q_loss
      
        q_loss[1].backward()
        torch.nn.utils.clip_grad_norm_(self.chargerAgent.critic[0].parameters(), 10 * self.num_chargerAgents)
        self.charger_critic_optimizer.step()

        policy_loss = [0.]*2
        self.UAV_actor_optimizer.zero_grad()
        self.charger_actor_optimizer.zero_grad()
        # Actor 网络更新 - UAV部分
        for i in range(self.num_UAVAgents):
            ac = action_batch.clone()
            action_i = self.UAVAgent.actor[i](state_batch[:, i, :])
            ac[:, i, :] = action_i
            curr_a_loss = -self.UAVAgent.critic[i](state_batch_flat, ac.view(ac.size()[0], -1)).mean()
            policy_loss[0] += curr_a_loss
        policy_loss[0].backward()
        torch.nn.utils.clip_grad_norm_(self.UAVAgent.actor[0].parameters(), 10 * self.num_UAVAgents)
        torch.nn.utils.clip_grad_norm_(self.UAVAgent.actor[1].parameters(), 10 * self.num_UAVAgents)
        self.UAV_actor_optimizer.step()

        # Actor 网络更新 - Charger部分
        for i in range(self.num_chargerAgents):
            ac = action_batch.clone()
            action_i = self.chargerAgent.actor[i](state_batch[:, i+self.num_UAVAgents, :self.dim_chargerobs])
            ac[:, i+self.num_UAVAgents, :] = action_i
            curr_a_loss = -self.chargerAgent.critic[i](state_batch_flat, ac.view(ac.size()[0], -1)).mean()
            policy_loss[1] += curr_a_loss
        policy_loss[1].backward()
        torch.nn.utils.clip_grad_norm_(self.chargerAgent.actor[0].parameters(), 10 * self.num_chargerAgents)
        self.charger_actor_optimizer.step()

        self.memory.td_error_update_TO(abs_error, self.curr_index_agent)
        return q_loss, policy_loss
   
    def update_target_net(self):
        """
        对目标网络进行软更新
        """
        for i in range(self.num_UAVAgents):
            self._soft_update(self.UAVAgent.target_actor[i], self.UAVAgent.actor[i], self.tau)
            self._soft_update(self.UAVAgent.target_critic[i], self.UAVAgent.critic[i], self.tau)
        for i in range(self.num_chargerAgents):
            self._soft_update(self.chargerAgent.target_actor[i], self.chargerAgent.actor[i], self.tau)
            self._soft_update(self.chargerAgent.target_critic[i], self.chargerAgent.critic[i], self.tau)

    def _soft_update(self, target_net, source_net, tau):
        """
        执行软更新操作：target = tau * source + (1 - tau) * target

        参数:
            target_net: 目标网络
            source_net: 源网络
            tau: 软更新系数
        """
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)
