from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from utils.misc import hard_update, gumbel_softmax, onehot_from_logits
from utils.policies import DiscretePolicy

class AttentionAgent(object):
    """
    General class for Attention agents (policy, target policy)
    """
    def __init__(self, num_in_pol, num_out_pol, hidden_dim=64,
                 lr=0.01, onehot_dim=0, device='cuda'):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            device (str or torch.device): the device on which to run the model
        """
        self.device = device

        # 初始化策略网络并移动到指定设备
        self.policy = DiscretePolicy(num_in_pol, num_out_pol,
                                     hidden_dim=hidden_dim,
                                     onehot_dim=onehot_dim).to(self.device)
        
        # 初始化目标策略网络并移动到指定设备
        self.target_policy = DiscretePolicy(num_in_pol,
                                            num_out_pol,
                                            hidden_dim=hidden_dim,
                                            onehot_dim=onehot_dim).to(self.device)

        # 将目标策略网络的参数与策略网络的参数同步
        hard_update(self.target_policy, self.policy)
        
        # 初始化优化器
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to sample
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        # 确保 obs 在正确的设备上
        obs = obs.to(self.device)
        
        # 使用策略网络生成动作
        return self.policy(obs, sample=explore)

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
