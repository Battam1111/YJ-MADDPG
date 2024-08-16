from utils import action_normalize
import torch
import torch.nn.functional as F

class MADDPGAgent(torch.nn.Module):
   def __init__(self, 
                node_type:int, 
                dim_obs_list:list, 
                dim_act_list:list,
                n_agents:int,
                dim_all_obs,
                dim_all_act,
                device):
      super().__init__()
      if node_type==0:
         self.actor = [Actor_uav(dim_obs_list[node_type],dim_act_list[node_type],device) for _ in range(n_agents)]
         self.target_actor = [Actor_uav(dim_obs_list[node_type],dim_act_list[node_type],device) for _ in range(n_agents)]
      else:
         self.actor = [Actor_charger(dim_obs_list[node_type],dim_act_list[node_type],device) for _ in range(n_agents)]
         self.target_actor = [Actor_charger(dim_obs_list[node_type],dim_act_list[node_type],device) for _ in range(n_agents)]
      self.critic = [Critic(dim_all_obs, dim_all_act, device)for _ in range(n_agents)]
      self.target_critic = [Critic(dim_all_obs, dim_all_act, device)for _ in range(n_agents)]
      for i in range(n_agents):
         self.target_actor[i].load_state_dict(self.actor[i].state_dict())
         self.target_critic[i].load_state_dict(self.critic[i].state_dict())

class Actor_uav(torch.nn.Module):
    def __init__(self, dim_observation, dim_action, device, activation_fc=F.leaky_relu):
        super(Actor_uav, self).__init__()
        self.device = device
        self.activation_fc = activation_fc
        self.FC1 = torch.nn.Linear(dim_observation, 128).to(self.device)
        self.FC2 = torch.nn.Linear(128, 64).to(self.device)
        self.FC3 = torch.nn.Linear(64, dim_action).to(self.device)

    def forward(self, obs):
        result = self.activation_fc(self.FC1(obs))
        result = self.activation_fc(self.FC2(result))
        result = action_normalize(torch.tanh(self.FC3(result)))
        return result

class Actor_charger(torch.nn.Module):
    def __init__(self, dim_observation, dim_action, device, activation_fc=F.leaky_relu):
        super(Actor_charger, self).__init__()
        self.device = device
        self.activation_fc = activation_fc
        self.FC1 = torch.nn.Linear(dim_observation, 64).to(self.device)
        self.FC2 = torch.nn.Linear(64, dim_action).to(self.device)

    def forward(self, obs):
        result = self.activation_fc(self.FC1(obs))
        result = action_normalize(torch.tanh(self.FC2(result)))
        return result

class Critic(torch.nn.Module):
    def __init__(self, dim_observation, dim_action, device, activation_fc=F.leaky_relu):
        super(Critic, self).__init__()
        self.device = device
        self.activation_fc = activation_fc
        self.FC1 = torch.nn.Linear(dim_observation, 256).to(self.device)
        self.FC2 = torch.nn.Linear(256+dim_action, 128).to(self.device)
        self.FC3 = torch.nn.Linear(128, 64).to(self.device)
        self.FC4 = torch.nn.Linear(64, 1).to(self.device)

    def forward(self, obs, acts):
        embedding = self.activation_fc(self.FC1(obs))
        result = torch.cat((embedding,acts), 1)
        result = self.activation_fc(self.FC2(result))
        result = self.activation_fc(self.FC3(result))
        result = self.FC4(result)
        return result