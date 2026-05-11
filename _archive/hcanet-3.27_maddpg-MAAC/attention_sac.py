import torch
import torch.nn as nn
import torch.optim as optim
from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from utils.agents import AttentionAgent
from utils.critics import AttentionCritic

class AttentionSAC:
    """
    使用集中注意力的SAC算法的多智能体强化学习类
    """

    def __init__(self, agent_init_params, agent_types, sa_size, gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.0, pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4, device='cpu'):
        """
        初始化AttentionSAC类

        参数:
            agent_init_params (list of dict): 初始化每个代理的参数字典列表
            sa_size (list of (int, int)): 每个代理的状态和动作空间大小
            gamma (float): 折扣因子
            tau (float): 目标网络更新速率
            pi_lr (float): 策略网络学习率
            q_lr (float): 评论家网络学习率
            reward_scale (float): 奖励缩放因子
            pol_hidden_dim (int): 策略网络的隐藏层维度
            critic_hidden_dim (int): 评论家网络的隐藏层维度
            attend_heads (int): 注意力头的数量
            device (str): 设备类型 ('cpu' 或 'gpu')
        """
        self.nagents = len(sa_size)  # 智能体的数量
        self.device = device
        self.agent_types = agent_types  # 存储智能体的类型

        # 初始化每个智能体的策略网络
        self.agents = [AttentionAgent(lr=pi_lr, hidden_dim=pol_hidden_dim, **params)
                       for params in agent_init_params]
        
        # 初始化评论家网络和目标评论家网络
        self.critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim, attend_heads=attend_heads).to(self.device)
        self.target_critic = AttentionCritic(sa_size, hidden_dim=critic_hidden_dim, attend_heads=attend_heads).to(self.device)
        
        # 硬更新目标评论家网络
        hard_update(self.target_critic, self.critic)
        
        # 设置评论家网络优化器
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=q_lr, weight_decay=1e-3)
        
        # 保存初始化参数
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.niter = 0  # 迭代次数

        # 初始化 self.init_dict
        self.init_dict = {
            'agent_init_params': agent_init_params,
            'agent_types': agent_types,
            'sa_size': sa_size,
            'gamma': gamma,
            'tau': tau,
            'pi_lr': pi_lr,
            'q_lr': q_lr,
            'reward_scale': reward_scale,
            'pol_hidden_dim': pol_hidden_dim,
            'critic_hidden_dim': critic_hidden_dim,
            'attend_heads': attend_heads,
            'device': device
        }

    @property
    def policies(self):
        """获取所有智能体的策略网络"""
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        """获取所有智能体的目标策略网络"""
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=False):
        """
        所有智能体在环境中前进一步
        参数:
            observations (list): 每个智能体的观察值列表
            explore (bool): 是否进行探索
        返回:
            actions (list): 每个智能体的动作列表
        """
        actions = []
        for i, (obs, agent_type) in enumerate(zip(observations, self.agent_types)):
            # 确保 obs 在正确的设备上
            obs = obs.to(self.device)
            action = self.agents[i].step(obs[:self.agent_init_params[i]['num_in_pol']], explore=explore)
            actions.append(action)
        return actions

    def update_critic(self, sample, soft=True, logger=None):
        """
        更新所有智能体的中央评论家网络
        参数:
            sample (tuple): 从回放缓冲区采样的数据 (obs, acs, rews, next_obs, dones)
            soft (bool): 是否使用软更新
            logger: 日志记录器
        """
        obs, acs, rews, next_obs, dones = sample
        MSELoss = nn.MSELoss()

        # 确保所有数据类型一致为 Float 类型
        rews = [torch.tensor(r, device=self.device, dtype=torch.float) if not isinstance(r, torch.Tensor) else r.to(self.device).float() for r in rews]
        dones = [torch.tensor(d, device=self.device, dtype=torch.float) if not isinstance(d, torch.Tensor) else d.to(self.device).float() for d in dones]
        
        obs = [ob.to(self.device).float() for i, ob in enumerate(obs)]
        next_obs = [ob.to(self.device).float() for i, ob in enumerate(next_obs)]
        
        next_acs = []
        next_log_pis = []
        for i, pi in enumerate(self.target_policies):
            curr_next_ac, curr_next_log_pi = pi(next_obs[i], return_log_pi=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)
        
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs.to(self.device)))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in, regularize=True, logger=logger, niter=self.niter)
        
        q_loss = 0
        for a_i, nq, log_pi, ret in zip(range(self.nagents), next_qs, next_log_pis, critic_rets):
            if len(ret) == 2:
                pq, regs = ret
            else:
                pq = ret
                regs = []

            rews[a_i] = rews[a_i].unsqueeze(0) if rews[a_i].dim() == 0 else rews[a_i]
            dones[a_i] = dones[a_i].unsqueeze(0) if dones[a_i].dim() == 0 else dones[a_i]

            target_q = (rews[a_i].view(-1, 1) + self.gamma * nq * (1 - dones[a_i].view(-1, 1))).float()
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(pq.float(), target_q.detach())
            for reg in regs:
                q_loss += reg  # 正则化注意力

        q_loss.backward()
        self.critic.scale_shared_grads()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
        self.niter += 1


    def update_policies(self, sample, soft=True, logger=None):
        """
        更新所有智能体的策略网络
        参数:
            sample (tuple): 从回放缓冲区采样的数据 (obs, acs, rews, next_obs, dones)
            soft (bool): 是否使用软更新
            logger: 日志记录器
        """
        obs, acs, rews, next_obs, dones = sample
        samp_acs = []
        all_probs = []
        all_log_pis = []
        all_pol_regs = []

        # 处理 obs 和 next_obs 以确保维度正确
        obs = [ob[:self.agent_init_params[i]['num_in_pol']].to(self.device) for i, ob in enumerate(obs)]
        next_obs = [ob[:self.agent_init_params[i]['num_in_pol']].to(self.device) for i, ob in enumerate(next_obs)]

        for i, (pi, ob, agent_type) in enumerate(zip(self.policies, obs, self.agent_types)):
            curr_ac, probs, log_pi, pol_regs, ent = pi(ob, return_all_probs=True, return_log_pi=True, regularize=True, return_entropy=True)
            if logger:
                logger.add_scalar(f'agent{i}/policy_entropy', ent, self.niter)
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)

        # 处理 critic_rets 以确保兼容单个或多个智能体的情况
        for i, (probs, log_pi, pol_regs) in enumerate(zip(all_probs, all_log_pis, all_pol_regs)):
            curr_agent = self.agents[i]

            if isinstance(critic_rets[i], tuple) and len(critic_rets[i]) == 2:
                q, all_q = critic_rets[i]
            elif isinstance(critic_rets[i], list) and len(critic_rets[i]) == 2:
                q, all_q = critic_rets[i][0], critic_rets[i][1]
            else:
                q = critic_rets[i]
                all_q = None

            v = (all_q * probs).sum(dim=1, keepdim=True) if all_q is not None else q
            pol_target = q - v

            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()

            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # 策略正则化

            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            if logger:
                logger.add_scalar(f'agent{i}/losses/pol_loss', pol_loss, self.niter)


    def update_all_targets(self):
        """更新所有目标网络（在每个代理的正常更新后调用）"""
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device='gpu'):
        """
        准备训练
        参数:
            device (str): 设备类型，'cpu'或'gpu'
        """
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        fn = lambda x: x.cuda() if device == 'gpu' else x.cpu()
        if self.device != device:
            for a in self.agents:
                a.policy = fn(a.policy)
                a.target_policy = fn(a.target_policy)
            self.critic = fn(self.critic)
            self.target_critic = fn(self.target_critic)
            self.device = device

    def prep_rollouts(self, device='cpu'):
        """
        准备执行回合
        参数:
            device (str): 设备类型，'cpu'或'gpu'
        """
        for a in self.agents:
            a.policy.eval()
        fn = lambda x: x.cuda() if device == 'gpu' else x.cpu()
        if self.device != device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.device = device

    def save(self, filename):
        """
        保存所有智能体的训练参数到一个文件中
        参数:
            filename (str): 保存文件的路径
        """
        original_device = self.device
        self.prep_training(device='cpu')  # 在保存前将参数移动到CPU
        save_dict = {
            'init_dict': self.init_dict,
            'agent_params': [a.get_params() for a in self.agents],
            'critic_params': {
                'critic': self.critic.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()
            }
        }
        torch.save(save_dict, filename)
        # 将模型移回原始设备
        self.prep_training(device=original_device)


    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.0, pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4, device='cpu'):
        """
        从多智能体环境实例化该类
        参数:
            env: 多智能体Gym环境
            gamma: 折扣因子
            tau: 目标网络的更新速率
            pi_lr: 策略网络的学习率
            q_lr: 评论家网络的学习率
            reward_scale: 奖励缩放
            pol_hidden_dim: 策略网络的隐藏层维度
            critic_hidden_dim: 评论家网络的隐藏层维度
            attend_heads: 注意力头的数量
        """
        agent_init_params = []
        sa_size = []
        for acsp, obsp in zip(env.action_space, env.observation_space):
            agent_init_params.append({'num_in_pol': obsp.shape[0], 'num_out_pol': acsp.n})
            sa_size.append((obsp.shape[0], acsp.n))

        init_dict = {
            'gamma': gamma,
            'tau': tau,
            'pi_lr': pi_lr,
            'q_lr': q_lr,
            'reward_scale': reward_scale,
            'pol_hidden_dim': pol_hidden_dim,
            'critic_hidden_dim': critic_hidden_dim,
            'attend_heads': attend_heads,
            'agent_init_params': agent_init_params,
            'sa_size': sa_size,
            'device': device
        }
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        从'保存'方法创建的文件中实例化该类
        参数:
            filename (str): 保存文件的路径
            load_critic (bool): 是否加载评论家网络参数
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.prep_training(device=save_dict['init_dict']['device'])  # 确保模型加载到正确的设备上
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
            a.policy.to(instance.device)  # 确保代理的策略网络在正确的设备上
            a.target_policy.to(instance.device)  # 确保目标策略网络在正确的设备上

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
            instance.critic.to(instance.device)  # 确保评论家网络在正确的设备上
            instance.target_critic.to(instance.device)  # 确保目标评论家网络在正确的设备上
        
        return instance
