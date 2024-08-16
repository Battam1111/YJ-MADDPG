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
from nn import HeteroMAGNet
from nn.modules.action import HeteroMAGNetActionLayer
from nn.modules.graph import GATModule
from rl.epsilon_schedules import DecayThenFlatSchedule
from rl.replay import ReplayBuffer, Transition
from training import TrainingConfig
from nn.nets import MADDPGAgent
from env.ou_noise import OUNoise
from utils import get_load_path
from itertools import chain

class BaseController(ABC):

   def __init__(self,
                node_types,
                agent_types,
                features_by_node_type,
                actions_by_node_type,
                device: torch_device):
      self.num_unit_types = len(set(node_types))

      assert self.num_unit_types == len(features_by_node_type) == len(actions_by_node_type)
      assert all([at in node_types for at in agent_types])

      self.device = device
      self.agent_types = torch.tensor(agent_types, dtype=torch.long, device=self.device)
      self.n_agents = sum(node_types.count(agent_type) for agent_type in agent_types)
      self.n_actions_agents = [actions_by_node_type[nt] for nt in node_types if nt in agent_types]

   def act(self, *args) -> torch.tensor:
      raise NotImplementedError

   @staticmethod
   def _action_lists_to_tensors(actions: List[list]):
      for i in range(len(actions)):
         actions[i] = torch.tensor(actions[i], dtype=torch.bool)
         while len(actions[i].size()) < 1:
            actions[i].unsqueeze_(-1)

   @staticmethod
   def random_policy(valid_actions: list, device: torch_device) -> torch.tensor:
      """Randomly selects an action for each agent, among the valid ones.

      :param valid_actions: list of lists containing boolean masks of valid actions for each agent, with `True` meaning the action is valid for selection and `False` meaning the action will be ignored during selection.
      :type valid_actions: list
      :return: 1D tensor containing the index of the selected action for each agent. Selected actions must be valid for application in the current state.
      :rtype: torch.tensor
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
                encoding_output_size: list,
                graph_hidden_size: list,
                action_hidden_size: list,
                share_encoding: bool,
                act_encoding: str,
                act_comms: str,
                act_action: str,
                gamma,
                tau,
                device,
                resume_run,
                memory_size,
                full_receptive_field: bool = True,
                gat_n_heads: int = 1,
                gat_average_last: bool = False,
                dropout: int = 0,
                add_self_loops: bool = True):
               #  mixing=None):
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
      self.device = device
      self.OUNoise = OUNoise(self.dim_UAVactions) #, scale=0.2, sigma= 0.2
      self.var = [1. for _ in range(num_UAVAgents+num_chargerAgents)]
      # self.epsilon = 1.0
      # self.epsilon_decay = 0.9993
      # self.epsilon_min = 0.01

      self.UAVAgent = MADDPGAgent(0, 
                                  node_types,
                                  dim_obs_list, 
                                  dim_act_list,
                                  num_UAVAgents,
                                  encoding_output_size,
                                  graph_hidden_size,
                                  action_hidden_size,
                                  share_encoding,
                                  act_encoding,
                                  act_comms,
                                  act_action,
                                  device,
                                  full_receptive_field,
                                  gat_n_heads,
                                  gat_average_last,
                                  dropout,
                                  add_self_loops)
      self.chargerAgent = MADDPGAgent(1,
                                      node_types,
                                      dim_obs_list,
                                      dim_act_list,
                                      num_chargerAgents,
                                      encoding_output_size,
                                      graph_hidden_size,
                                      action_hidden_size,
                                      share_encoding,
                                      act_encoding,
                                      act_comms,
                                      act_action,
                                      device,
                                      full_receptive_field,
                                      gat_n_heads,
                                      gat_average_last,
                                      dropout,
                                      add_self_loops)
      
      self.shared_modules = [self.UAVAgent.critic]
      self.memory = ReplayMemory(memory_size, self.num_UAVAgents+num_chargerAgents, True, self.dim_UAVobs, self.dim_UAVactions, True, 3, 0.95)

      if optimizer == 'rmsprop':
         one_critic_param = []
         for i in range(self.num_UAVAgents):
            one_critic_param = one_critic_param + list(self.UAVAgent.critic[i].parameters())
         self.critic_optimizer = optim.RMSprop(one_critic_param,
                                        lr=self.critic_lr,
                                        alpha=rmsprop_alpha,
                                        eps=rmsprop_eps,
                                        weight_decay=self.weight_decay)
         one_actor_param = []
         for i in range(self.num_UAVAgents):
            one_actor_param = one_actor_param + list(self.UAVAgent.actor[i].parameters())
         self.actor_optimizer = optim.RMSprop(one_actor_param,
                                       lr=self.actor_lr,
                                       alpha=rmsprop_alpha,
                                       eps=rmsprop_eps,
                                       weight_decay=self.weight_decay)
         
      elif optimizer == 'adam':
         self.UAV_critic_optimizer = optim.Adam(self.UAVAgent.critic.parameters(),
                                                lr=self.critic_lr)
         
         self.charger_critic_optimizer = optim.Adam(self.chargerAgent.critic.parameters(),
                                                    lr=self.critic_lr)
         
         # one_actor_param = []
         # for i in range(self.num_UAVAgents):
         #    one_actor_param = one_actor_param + list(self.UAVAgent.actor[i].parameters())
         self.UAV_actor_optimizer = optim.Adam(self.UAVAgent.actor.parameters(),
                                           lr=self.actor_lr)
         
         # param = []
         # for i in range(self.num_chargerAgents):
         #    param += list(self.chargerAgent.actor[i].parameters())
         self.charger_actor_optimizer = optim.Adam(self.chargerAgent.actor.parameters(),
                                                   lr=self.actor_lr)
      else:
         raise ValueError('Invalid optimizer "{}"'.format(optimizer))

      if resume_run:
         load_path = get_load_path(checkpoint_file)
         print('Loading from checkpoint...')
         checkpoint = torch.load(load_path, map_location=self.device )
         self.UAVAgent.actor.load_state_dict(checkpoint['actor_state_dict'])
         # self.UAVAgent.actor[1].load_state_dict(checkpoint['actor2_state_dict'])
         # self.UAVAgent.actor[2].load_state_dict(checkpoint['actor3_state_dict'])
         self.chargerAgent.actor.load_state_dict(checkpoint['charger_actor_state_dict'])
         # self.UAVAgent.target_actor[2].load_state_dict(checkpoint['target_actor3_state_dict'])
         # self.UAVAgent.critic.load_state_dict(checkpoint['critic_state_dict'])
         # self.UAVAgent.critic[1].load_state_dict(checkpoint['critic2_state_dict'])
         # self.UAVAgent.critic[2].load_state_dict(checkpoint['critic3_state_dict'])
         # self.UAVAgent.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
         # self.UAVAgent.target_critic[1].load_state_dict(checkpoint['target_critic2_state_dict'])
         # self.UAVAgent.target_critic[2].load_state_dict(checkpoint['target_critic3_state_dict'])
         # self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
         # self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
         # self.total_steps = checkpoint['total_steps']
         # self.last_grad_log = checkpoint['last_log_step']
         # self.n_episodes = checkpoint['n_episodes']

   def get_net_state_dicts(self):
      return {
          'actor_state_dict': self.UAVAgent.actor.state_dict(),
         #  'target_actor1_state_dict': self.UAVAgent.target_actor[0].state_dict(),
         #  'actor2_state_dict': self.UAVAgent.actor[1].state_dict(),
         #  'target_actor2_state_dict': self.UAVAgent.target_actor[1].state_dict(),
         #  'actor3_state_dict': self.UAVAgent.actor[2].state_dict(),
          'charger_actor_state_dict': self.chargerAgent.actor.state_dict()}
         #  'target_actor3_state_dict': self.UAVAgent.target_actor[2].state_dict(),
         #  'critic_state_dict': self.UAVAgent.critic.state_dict(),
         #  'target_critic_state_dict': self.UAVAgent.target_critic.state_dict()}
         #  'critic2_state_dict': self.UAVAgent.critic[1].state_dict(),
         #  'target_critic2_state_dict': self.UAVAgent.target_critic[0].state_dict(),
         #  'critic3_state_dict': self.UAVAgent.critic[2].state_dict(),
         #  'target_critic3_state_dict': self.UAVAgent.target_critic[2].state_dict()}
   
   def save_checkpoint(self, step_num, n_episodes):
      print('Saving checkpoint...')

      net_weights = self.get_net_state_dicts()

      # common_variables = {
         #  'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
         #  'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
         #  'total_steps': step_num,
         #  'last_log_step': self.last_grad_log,
         #  'n_episodes': n_episodes}  **common_variables

      variables = {**net_weights}
      cur_checkpointpath = path.join(self.checkpoint, 'model_{}.pt'.format(n_episodes))
      torch.save(variables, cur_checkpointpath)

   def act(self, state_batch, adj, episode_num, episode_before_train, if_noise):
      # data_list = DataLoader.uncollate(state_batch)
      actions = torch.zeros((self.num_UAVAgents+self.num_chargerAgents, self.dim_UAVactions), dtype=torch.float32, device = self.device)
      if(if_noise):
         for i in range(self.num_UAVAgents):
            # if np.random.rand() < self.epsilon:
            #    curr_act = torch.tensor(2*np.random.rand(self.dim_UAVactions)-1).to(self.device)
            # else:
            curr_act = self.UAVAgent.actor(self.choose_neighbor_for_actor(state_batch, adj[i], i))
            curr_act += (self.OUNoise.noise() * self.var[i]).type(torch.float32).to(self.device)
            if episode_num > episode_before_train and self.var[i] > 0.05:
               self.var[i] *= 0.9999997
            actions[i] = torch.clamp(curr_act, -1.0, 1.0)
         # self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
         for i in range(self.num_chargerAgents):
            curr_act = self.chargerAgent.actor(self.choose_neighbor_for_actor(state_batch, adj[i+self.num_UAVAgents], i+self.num_UAVAgents))
            curr_act += (self.OUNoise.noise() * self.var[i+self.num_UAVAgents]).type(torch.float32).to(self.device)
            if episode_num > episode_before_train and self.var[i+self.num_UAVAgents] > 0.05:
               self.var[i+self.num_UAVAgents] *= 0.9999997
            actions[i+self.num_UAVAgents] = torch.clamp(curr_act, -1.0, 1.0)
      else:
         for i in range(self.num_UAVAgents):
            curr_act = self.UAVAgent.actor(self.choose_neighbor_for_actor(state_batch, adj[i], i))
            actions[i] = torch.clamp(curr_act, -1.0, 1.0)
         for i in range(self.num_chargerAgents):
            curr_act = self.chargerAgent.actor(self.choose_neighbor_for_actor(state_batch, adj[i+self.num_UAVAgents], i+self.num_UAVAgents))
            actions[i+self.num_UAVAgents] = torch.clamp(curr_act, -1.0, 1.0)
      # print(actions)
      return actions
   
   def batch_to_data_list(self, batch):
      num_graphs = batch.num_graphs
      node_counts = batch.batch.bincount().tolist()
      edge_counts = batch.edge_index[0].bincount().tolist()
    
      data_list = []
      node_start = 0
      edge_start = 0
    
      for i in range(num_graphs):
         num_nodes_i = node_counts[i]
         num_edges_i = edge_counts[i]
        
         x_i = batch.x[node_start:node_start + num_nodes_i]
         edge_index_i = batch.edge_index[:, edge_start:edge_start + num_edges_i] - node_start
         y_i = batch.y[node_start:node_start + num_nodes_i]
        
         data_i = Data(x=x_i, edge_index=edge_index_i, y=y_i)
         data_list.append(data_i)
        
         node_start += num_nodes_i
         edge_start += num_edges_i
        
      return data_list
   
   def concat_obs_act(self, state_data_list, action_batch):
      data_list = []
      for index, state in enumerate(state_data_list):
         # print(state[-1].x)
         statandact_batch_x = torch.cat((state[-1].x, action_batch[index]), 1)
         statandact_batch_data = Data(x=statandact_batch_x, edge_index=state[-1].edge_index)
         data_list.append(statandact_batch_data)
      return Batch.from_data_list(data_list)

   def choose_full_graph(self, state_data_list):
      data_list = []
      for index, state in enumerate(state_data_list):
         # print(state[-1].x)
         stat_batch_x = state[-1].x
         stat_batch_data = Data(x=stat_batch_x, edge_index=state[-1].edge_index)
         data_list.append(stat_batch_data)
      return Batch.from_data_list(data_list)
   
   def sample_batch(self, episodes, param_dict: dict):
      state_batch = []
      action_batch = torch.zeros((len(episodes), param_dict["NUM_DRONE"], param_dict["DIMENSION_ACTION"][0]), dtype = torch.float32, device=self.device)
      next_state_batch = []
      reward_batch = []
      done_batch = []
      for i, episode in enumerate(episodes):
         state_batch.append(episode.state)
         action_batch[i] = episode.action.to(self.device)
         next_state_batch.append(episode.next_state)
         reward_batch.append(episode.reward)
         done_batch.append(episode.done)
      return state_batch, action_batch, next_state_batch, torch.tensor(reward_batch, dtype=torch.float32, device=self.device), torch.tensor(done_batch, device=self.device)
   
   def shared_parameters(self):
      """
      Parameters shared across agents and reward heads
      """
      return chain(*[m.parameters() for m in self.shared_modules])

   def scale_shared_grads(self):
      """
      Scale gradients for parameters that are shared since they accumulate
      gradients from the critic loss function multiple times
      """
      for p in self.shared_parameters():
         p.grad.data.mul_(1. / self.num_UAVAgents)

   def choose_neighbor_for_actor(self, state_batch, adj_batch, index):
      if state_batch.ndim != 3: #[bs, n_agents, dim_obs]
         state_batch = state_batch.unsqueeze(0)
      if adj_batch.ndim != 2: #[bs, 2/1]
         adj_batch = adj_batch.unsqueeze(0)
      bs = state_batch.shape[0]
      neighbor = adj_batch.shape[1]

      x = state_batch[:, index, :].unsqueeze(1)
      y = torch.zeros((bs, neighbor, state_batch.shape[2]), dtype=torch.float32, device=self.device)
      for i in range(bs):
         y[i] = state_batch[i, adj_batch[i].squeeze(), :]
      return torch.cat((x,y), dim=1)
      
   def update(self, i_step, param_dict: dict):
      
      # for i in range(self.num_UAVAgents):
      self.UAVAgent.actor.train()
      self.UAVAgent.critic.train()
      # for i in range(self.num_chargerAgents):
      self.chargerAgent.actor.train()
      self.chargerAgent.critic.train()

      # episodes = memory.sample(param_dict["BATCH_SIZE"]) if not memory.is_prioritized else memory.sample(
      #    param_dict["BATCH_SIZE"], param_dict["REPLAY_BUFFER_BETA"])
      self.curr_index_agent = i_step % (self.num_UAVAgents+self.num_chargerAgents)

      (state_batch, action_batch, next_state_batch, 
        reward_batch, done_batch, is_weight_batch,
        n_step_batch) = self.memory.sample_TO(param_dict["BATCH_SIZE"], self.curr_index_agent)

      state_batch = state_batch.cuda()
      state_batch, adj_batch = torch.split(state_batch, self.dim_UAVobs, dim=2)
      action_batch = action_batch.cuda()
      reward_batch = reward_batch.cuda()
      done_batch = done_batch.cuda()
      is_weight_batch = is_weight_batch.cuda()
      n_step_batch = n_step_batch.cuda()
      next_state_batch = next_state_batch.cuda()
      next_state_batch, next_adj_batch = torch.split(next_state_batch, self.dim_UAVobs, dim=2)
      adj_batch = adj_batch.type(torch.int64)
      next_adj_batch = next_adj_batch.type(torch.int64)

      target_next_actions_batch = torch.zeros((param_dict["BATCH_SIZE"], self.num_UAVAgents+self.num_chargerAgents, self.dim_UAVactions), device = self.device)
      
      q_loss = [0.]*2
      critic_loss = []
      abs_error = []
      beg = -self.memory.max_len
      end = (self.memory.now_len - self.memory.max_len) if (self.memory.now_len < self.memory.max_len) else None
      
      self.UAV_critic_optimizer.zero_grad()
      # Critic 网络更新
      for i in range(self.num_UAVAgents):
         target_next_actions_batch[:, i, :] = self.UAVAgent.target_actor(self.choose_neighbor_for_actor(next_state_batch, next_adj_batch[:, i, :], i)).detach()

      for i in range(self.num_chargerAgents):
         target_next_actions_batch[:, i+self.num_UAVAgents, :] = self.chargerAgent.target_actor(self.choose_neighbor_for_actor(next_state_batch, next_adj_batch[:, i+self.num_UAVAgents, :], i+self.num_UAVAgents)).detach()

      for i in range(self.num_UAVAgents):
         predicted_q_values_batch = self.UAVAgent.critic(state_batch, action_batch, i)
         target_next_q_values_batch = self.UAVAgent.target_critic(next_state_batch, target_next_actions_batch, i)
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
      # self.UAVAgent.critic.scale_shared_grads()
      torch.nn.utils.clip_grad_norm_(
         self.UAVAgent.critic.parameters(), 10 * self.num_UAVAgents)
      self.UAV_critic_optimizer.step()
      
      self.charger_critic_optimizer.zero_grad()
      for i in range(self.num_chargerAgents):
         predicted_q_values_batch = self.chargerAgent.critic(state_batch, action_batch, i+self.num_UAVAgents)
         target_next_q_values_batch = self.chargerAgent.target_critic(next_state_batch, target_next_actions_batch, i+self.num_UAVAgents)
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
      torch.nn.utils.clip_grad_norm_(
         self.chargerAgent.critic.parameters(), 10 * self.num_chargerAgents)
      self.charger_critic_optimizer.step()


      policy_loss = [0.]*2
      self.UAV_actor_optimizer.zero_grad()
      self.charger_actor_optimizer.zero_grad()
      # if i_step % 2 == 0:
      # Actor 网络更新
      for i in range(self.num_UAVAgents):
         ac = action_batch.clone()
         action_i = self.UAVAgent.actor(self.choose_neighbor_for_actor(state_batch, adj_batch[:, i, :], i))
         ac[:, i, :] = action_i
         curr_a_loss = -self.UAVAgent.critic(state_batch, ac, i).mean()
         policy_loss[0] += curr_a_loss
         
      policy_loss[0].backward()
      torch.nn.utils.clip_grad_norm_(
            self.UAVAgent.actor.parameters(), 10 * self.num_UAVAgents)
      # torch.nn.utils.clip_grad_norm_(
      #       self.UAVAgent.actor[1].parameters(), 10 * self.num_UAVAgents)
      # torch.nn.utils.clip_grad_norm_(
      #       self.UAVAgent.actor[2].parameters(), 10 * self.num_UAVAgents)
      self.UAV_actor_optimizer.step()

      for i in range(self.num_chargerAgents):
         ac = action_batch.clone()
         action_i = self.chargerAgent.actor(self.choose_neighbor_for_actor(state_batch, adj_batch[:, i+self.num_UAVAgents, :], i+self.num_UAVAgents))
         ac[:, i+self.num_UAVAgents, :] = action_i
         curr_a_loss = -self.chargerAgent.critic(state_batch, ac, i+self.num_UAVAgents).mean()
         policy_loss[1] += curr_a_loss

      policy_loss[1].backward()
      torch.nn.utils.clip_grad_norm_(
         self.chargerAgent.actor.parameters(), 10 * self.num_chargerAgents)
      self.charger_actor_optimizer.step()

      # else:
      #    policy_loss = None
      self.memory.td_error_update_TO(abs_error, self.curr_index_agent)
      return q_loss, policy_loss
   
   def update_target_net(self):
      # for i in range(self.num_UAVAgents):
      self._soft_update(self.UAVAgent.target_actor, self.UAVAgent.actor, self.tau)
      self._soft_update(self.UAVAgent.target_critic, self.UAVAgent.critic, self.tau)
      # for i in range(self.num_chargerAgents):
      self._soft_update(self.chargerAgent.target_actor, self.chargerAgent.actor, self.tau)
      self._soft_update(self.chargerAgent.target_critic, self.chargerAgent.critic, self.tau)

   def _soft_update(self, target_net, source_net, tau):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)