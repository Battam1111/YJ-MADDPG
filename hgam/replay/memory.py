from collections import namedtuple
import numpy as np
import torch
import os
import numpy.random as rd
from torch._C import device

Experience = namedtuple('Experience',
                        ('states', 'actions', 'next_states', 'rewards', 'done'))

class ReplayMemory:
    def __init__(
        self, capacity, n_agents, use_cuda,
        state_dim, action_dim, if_use_per=False,
        n_step=1, gamma=0.95
        ):
        self.max_len = capacity
        self.n_agents = n_agents
        self.use_cuda = use_cuda
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.action_dim = action_dim
        self.state_dim = state_dim

        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        self.data_type = FloatTensor
        self.device = torch.device("cpu")

        if if_use_per:
            self.per_tree = [BinarySearchTree(self.max_len) for _ in range(n_agents)]
        else:
            self.per_tree = None

        other_dim = self.action_dim + 1 + 1
        self.other_dim = other_dim
        self.buf_other = torch.empty((self.max_len, n_agents, other_dim), dtype=torch.float32, device=self.device)
        self.buf_state = torch.empty((self.max_len, n_agents, state_dim+2), device=self.device)
        # self.buf_state = []

        self.trajectory = torch.empty((self.max_len, ), device=self.device)
        self.n_step = n_step
        self.gamma = gamma

    def convert_experience_to_stateAndOther(self, state, adj, action, next_state, next_adj, reward, done):
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        tem_other = torch.empty((self.n_agents, self.other_dim), dtype=torch.float32, device=self.device)

        tem_state = torch.cat((state.cpu(), adj.cpu()), dim=1)
        for idx_agent in range(self.n_agents):
            tem_other[idx_agent] = torch.tensor(np.concatenate(
                (action[idx_agent].detach().numpy(), [reward[idx_agent]], [done[idx_agent]]),
                axis=0
            )).type(FloatTensor)
        return tem_state, tem_other

    def append_buffer(self, state, other, i_episode):  # CPU array to CPU array
        # if self.next_idx >= len(self.buf_state):
        #     self.buf_state.append(state)
        # else:
        #     self.buf_state[self.next_idx] = state
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other
        self.trajectory[self.next_idx] = i_episode

        if self.per_tree:
            # self.per_tree.update_id(self.next_idx)
            for tree in self.per_tree:
                tree.update_id(self.next_idx)

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    # 原memory接口
    def push(self, state, adj, action, next_state, next_adj, reward, done, i_episode):
        state, other = self.convert_experience_to_stateAndOther(state, adj, action, next_state, next_adj, reward, done)
        self.append_buffer(state, other, i_episode)
        self.update_now_len()

    def sample_batch(self, batch_size) -> tuple:
        if self.per_tree:
            state_all, action_all, next_state_all, reward_all, done_all, is_weights_all = [], [], [], [], [], []
            for tree in self.per_tree:
                beg = -self.max_len
                end = (self.now_len - self.max_len) if (self.now_len < self.max_len) else None

                indices, is_weights = tree.get_indices_is_weights(batch_size, beg, end)
                r_m_a = self.buf_other[indices]
                state_all.append(self.buf_state[indices])
                action_all.append(r_m_a[:, :, :3])
                next_state_all.append(self.buf_state[indices + 1])
                reward_all.append(r_m_a[:, :, 3])
                done_all.append(r_m_a[:, :, 4])
                is_weights_all.append(torch.tensor(is_weights, dtype=torch.float32, device=self.device))
            return state_all, action_all, next_state_all, reward_all, done_all, is_weights_all
        else:
            indices = rd.randint(self.now_len - 1, size=batch_size)
            r_m_a = self.buf_other[indices]
            return (
                self.buf_state[indices],# state
                r_m_a[:, :, :3],# action
                self.buf_state[indices + 1],# next_state
                r_m_a[:, :, 3],# reward
                r_m_a[:, :, 4],# done
            )

    # 原memory接口
    def sample(self, batch_size):
        state_all, action_all, next_state_all, reward_all, done_all, is_weights_all = self.sample_batch(batch_size)
        return state_all, action_all, next_state_all, reward_all, done_all, is_weights_all

    def sample_TO(self, batch_size, index_agent):
        states_batch, actions_batch, next_states_batch, rewards_batch, dones_batch, is_weights_batch, n_step_all = self.sample_batch_TO(batch_size, index_agent)
        return states_batch, actions_batch, next_states_batch, rewards_batch, dones_batch, is_weights_batch, n_step_all

    def sample_batch_TO(self, batch_size, index_agent):
        tree = self.per_tree[index_agent]

        beg = -self.max_len
        end = (self.now_len - self.max_len) if (self.now_len < self.max_len) else None

        indices, is_weights = tree.get_indices_is_weights(batch_size, beg, end)
        r_m_a = self.buf_other[indices]

        # n-step return
        done, next_state, reward, n_step = [], [], [], torch.zeros((batch_size, ), dtype=torch.float32, device=self.device)
        for j, idx in enumerate(indices):
            curr_trajectory_id = self.trajectory[idx]
            curr_sum_reward = torch.zeros((self.n_agents, ), dtype=torch.float32, device=self.device)
            for i in range(self.n_step):
                if self.trajectory[(idx + i) % self.max_len] != curr_trajectory_id:# or (idx + i) >= self.now_len:
                    vaild_next_state = False
                    break
                else:
                    curr_sum_reward += pow(self.gamma, i) * self.buf_other[(idx + i) % self.max_len, :, 2]
                    vaild_next_state = True
            if vaild_next_state is True:
                done.append(self.buf_other[(idx + i) % self.max_len, :, 3])
            else:
                done.append(torch.tensor([1.] * self.n_agents, dtype=torch.float32, device=self.device))
            next_state.append(self.buf_state[(idx + i + 1) % self.max_len])
            reward.append(curr_sum_reward)
            n_step[j] = i + 1
        next_state = torch.stack(next_state, dim=0)
        reward = torch.stack(reward, dim=0)
        done = torch.stack(done, dim=0)
        # state_re = [self.buf_state[id] for id in indices]
        return (
            self.buf_state[indices],# state
            r_m_a[:, :, :2],# action
            # self.buf_state[indices + 1],# next_state
            # r_m_a[:, :, 3],# reward
            # r_m_a[:, :, 4],# done
            next_state, 
            reward, 
            done,
            torch.tensor(is_weights, dtype=torch.float32, device=self.device), # is_weights
            n_step
        )

    def update_now_len(self):
        """update the a pointer `now_len`, which is the current data number of ReplayBuffer
        """
        self.now_len = self.max_len if self.if_full else self.next_idx

    def td_error_update(self, td_error):
        # self.per_tree.td_error_update(td_error)
        for tree, curr_td_error in zip(self.per_tree, td_error):
            tree.td_error_update(curr_td_error)

    def td_error_update_TO(self, td_error, curr_index_agent):
        """更新PER树"""
        for idx, tree, curr_td_error in zip(range(self.n_agents), self.per_tree, td_error):
            tree.td_error_update(curr_td_error)

    # 原memory接口
    def __len__(self):
        return self.now_len

"""[ElegantRL.2021.09.01](https://github.com/AI4Finance-LLC/ElegantRL)"""
class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, if_use_per, gpu_id=0):
        """Experience Replay Buffer
        save environment transition in a continuous RAM for high performance training
        we save trajectory in order and save state and other (action, reward, mask, ...) separately.
        `int max_len` the maximum capacity of ReplayBuffer. First In First Out
        `int state_dim` the dimension of state
        `int action_dim` the dimension of action (action_dim==1 for discrete action)
        `bool if_on_policy` on-policy or off-policy
        `bool if_gpu` create buffer space on CPU RAM or GPU
        `bool if_per` Prioritized Experience Replay for sparse reward
        """
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.data_type = torch.float32
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.per_tree = BinarySearchTree(max_len) if if_use_per else None

        other_dim = 1 + 1 + self.action_dim
        self.buf_other = torch.empty((max_len, other_dim), dtype=torch.float32, device=self.device)

        if isinstance(state_dim, int):  # state is pixel
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
        elif isinstance(state_dim, tuple):
            self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.uint8, device=self.device)
        else:
            raise ValueError('state_dim')

    def append_buffer(self, state, other):  # CPU array to CPU array
        self.buf_state[self.next_idx] = state
        self.buf_other[self.next_idx] = other

        if self.per_tree:
            self.per_tree.update_id(self.next_idx)

        self.next_idx += 1
        if self.next_idx >= self.max_len:
            self.if_full = True
            self.next_idx = 0

    def extend_buffer(self, state, other):  # CPU array to CPU array
        size = len(other)
        next_idx = self.next_idx + size

        if self.per_tree:
            self.per_tree.update_ids(data_ids=np.arange(self.next_idx, next_idx) % self.max_len)

        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        """randomly sample a batch of data for training
        :int batch_size: the number of data in a batch for Stochastic Gradient Descent
        :return torch.Tensor reward: reward.shape==(now_len, 1)
        :return torch.Tensor mask:   mask.shape  ==(now_len, 1), mask = 0.0 if done else gamma
        :return torch.Tensor action: action.shape==(now_len, action_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim)
        :return torch.Tensor state:  state.shape ==(now_len, state_dim), next state
        """
        if self.per_tree:
            beg = -self.max_len
            end = (self.now_len - self.max_len) if (self.now_len < self.max_len) else None

            indices, is_weights = self.per_tree.get_indices_is_weights(batch_size, beg, end)
            r_m_a = self.buf_other[indices]
            return (r_m_a[:, 0:1].type(torch.float32),  # reward
                    r_m_a[:, 1:2].type(torch.float32),  # mask
                    r_m_a[:, 2:].type(torch.float32),  # action
                    self.buf_state[indices].type(torch.float32),  # state
                    self.buf_state[indices + 1].type(torch.float32),  # next state
                    torch.as_tensor(is_weights, dtype=torch.float32, device=self.device))  # important sampling weights
        else:
            indices = rd.randint(self.now_len - 1, size=batch_size)
            r_m_a = self.buf_other[indices]
            return (r_m_a[:, 0:1],  # reward
                    r_m_a[:, 1:2],  # mask
                    r_m_a[:, 2:],  # action
                    self.buf_state[indices],
                    self.buf_state[indices + 1])

    def update_now_len(self):
        """update the a pointer `now_len`, which is the current data number of ReplayBuffer
        """
        self.now_len = self.max_len if self.if_full else self.next_idx

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        """print the state norm information: state_avg, state_std
        We don't suggest to use running stat state.
        We directly do normalization on state using the historical avg and std
        eg. `state = (state + self.neg_state_avg) * self.div_state_std` in `PreprocessEnv.step_norm()`
        neg_avg = -states.mean()
        div_std = 1/(states.std()+1e-5) or 6/(states.max()-states.min())
        :array neg_avg: neg_avg.shape=(state_dim)
        :array div_std: div_std.shape=(state_dim)
        """
        max_sample_size = 2 ** 14

        '''check if pass'''
        state_shape = self.buf_state.shape
        if len(state_shape) > 2 or state_shape[1] > 64:
            print(f"| print_state_norm(): state_dim: {state_shape} is too large to print its norm. ")
            return None

        '''sample state'''
        indices = np.arange(self.now_len)
        rd.shuffle(indices)
        indices = indices[:max_sample_size]  # len(indices) = min(self.now_len, max_sample_size)

        batch_state = self.buf_state[indices]

        '''compute state norm'''
        if isinstance(batch_state, torch.Tensor):
            batch_state = batch_state.cpu().data.numpy()
        assert isinstance(batch_state, np.ndarray)

        if batch_state.shape[1] > 64:
            print(f"| _print_norm(): state_dim: {batch_state.shape[1]:.0f} is too large to print its norm. ")
            return None

        if np.isnan(batch_state).any():  # 2020-12-12
            batch_state = np.nan_to_num(batch_state)  # nan to 0

        ary_avg = batch_state.mean(axis=0)
        ary_std = batch_state.std(axis=0)
        fix_std = ((np.max(batch_state, axis=0) - np.min(batch_state, axis=0)) / 6 + ary_std) / 2

        if neg_avg is not None:  # norm transfer
            ary_avg = ary_avg - neg_avg / div_std
            ary_std = fix_std / div_std

        print(f"print_state_norm: state_avg, state_std (fixed)")
        print(f"avg = np.{repr(ary_avg).replace('=float32', '=np.float32')}")
        print(f"std = np.{repr(ary_std).replace('=float32', '=np.float32')}")

    def td_error_update(self, td_error):
        self.per_tree.td_error_update(td_error)

    def save_or_load_history(self, cwd, if_save, buffer_id=0):
        save_path = f"{cwd}/replay_{buffer_id}.npz"
        if_load = None

        if if_save:
            self.update_now_len()
            state_dim = self.buf_state.shape[1]
            other_dim = self.buf_other.shape[1]

            buf_state_data_type = np.float16 \
                if self.buf_state.dtype in {np.float, np.float64, np.float32} \
                else np.uint8

            buf_state = np.empty((self.max_len, state_dim), dtype=buf_state_data_type)
            buf_other = np.empty((self.max_len, other_dim), dtype=np.float16)

            temp_len = self.max_len - self.now_len
            buf_state[0:temp_len] = self.buf_state[self.now_len:self.max_len].detach().cpu().numpy()
            buf_other[0:temp_len] = self.buf_other[self.now_len:self.max_len].detach().cpu().numpy()

            buf_state[temp_len:] = self.buf_state[:self.now_len].detach().cpu().numpy()
            buf_other[temp_len:] = self.buf_other[:self.now_len].detach().cpu().numpy()

            np.savez_compressed(save_path, buf_state=buf_state, buf_other=buf_other)
            print(f"| ReplayBuffer save in: {save_path}")
        elif os.path.isfile(save_path):
            buf_dict = np.load(save_path)
            buf_state = buf_dict['buf_state']
            buf_other = buf_dict['buf_other']

            buf_state = torch.as_tensor(buf_state, dtype=torch.float32, device=self.device)
            buf_other = torch.as_tensor(buf_other, dtype=torch.float32, device=self.device)
            self.extend_buffer(buf_state, buf_other)
            self.update_now_len()
            print(f"| ReplayBuffer load: {save_path}")
            if_load = True
        else:
            # print(f"| ReplayBuffer FileNotFound: {save_path}")
            if_load = False
        return if_load


class ReplayBufferMP:
    def __init__(self, state_dim, action_dim, max_len, if_use_per, buffer_num, gpu_id):
        """Experience Replay Buffer for Multiple Processing
        `int max_len` the max_len of ReplayBuffer, not the total len of ReplayBufferMP
        `int worker_num` the rollout workers number
        """
        self.now_len = 0
        self.max_len = max_len
        self.worker_num = buffer_num

        buf_max_len = max_len // buffer_num
        self.buffers = [ReplayBuffer(max_len=buf_max_len, state_dim=state_dim, action_dim=action_dim,
                                     if_use_per=if_use_per, gpu_id=gpu_id)
                        for _ in range(buffer_num)]

    def sample_batch(self, batch_size) -> list:
        bs = batch_size // self.worker_num
        list_items = [self.buffers[i].sample_batch(bs)
                      for i in range(self.worker_num)]
        # list_items of reward, mask, action, state, next_state
        # list_items of reward, mask, action, state, next_state, is_weights (PER)

        list_items = list(map(list, zip(*list_items)))  # 2D-list transpose
        return [torch.cat(item, dim=0) for item in list_items]

    def update_now_len(self):
        self.now_len = 0
        for buffer in self.buffers:
            buffer.update_now_len()
            self.now_len += buffer.now_len

    def print_state_norm(self, neg_avg=None, div_std=None):  # non-essential
        # for buffer in self.l_buffer:
        self.buffers[0].print_state_norm(neg_avg, div_std)

    def td_error_update(self, td_error):
        td_errors = td_error.view(self.worker_num, -1, 1)
        for i in range(self.worker_num):
            self.buffers[i].per_tree.td_error_update(td_errors[i])

    def save_or_load_history(self, cwd, if_save):
        for i in range(self.worker_num):
            self.buffers[i].save_or_load_history(cwd, if_save, buffer_id=i)


class BinarySearchTree:
    """Binary Search Tree for PER
    Contributor: Github GyChou, Github mississippiu
    Reference: https://github.com/kaixindelele/DRLib/tree/main/algos/pytorch/td3_sp
    Reference: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    """

    def __init__(self, memo_len):
        self.memo_len = memo_len  # replay buffer len
        self.prob_ary = np.zeros((memo_len - 1) + memo_len)  # parent_nodes_num + leaf_nodes_num
        self.max_len = len(self.prob_ary)
        self.now_len = self.memo_len - 1  # pointer
        self.indices = None
        self.depth = int(np.log2(self.max_len))

        # PER.  Prioritized Experience Replay. Section 4
        # alpha, beta = 0.7, 0.5 for rank-based variant
        # alpha, beta = 0.6, 0.4 for proportional variant
        self.per_alpha = 0.6  # alpha = (Uniform:0, Greedy:1)
        self.per_beta = 0.4  # beta = (PER:0, NotPER:1)

        self.abs_error_upper = 1.
        self.epsilon = 0.01
        self.beta_increment_per_sampling = 0.0001

    def update_id(self, data_id, prob=10):  # 10 is max_prob
        tree_id = data_id + self.memo_len - 1
        if self.now_len == tree_id:
            self.now_len += 1

        # delta = prob - self.prob_ary[tree_id]
        # self.prob_ary[tree_id] = prob
        delta = self.abs_error_upper - self.prob_ary[tree_id]
        self.prob_ary[tree_id] = self.abs_error_upper

        while tree_id != 0:  # propagate the change through tree
            tree_id = (tree_id - 1) // 2  # faster than the recursive loop
            self.prob_ary[tree_id] += delta

    def update_ids(self, data_ids, prob=10):  # 10 is max_prob
        ids = data_ids + self.memo_len - 1
        self.now_len += (ids >= self.now_len).sum()

        upper_step = self.depth - 1
        # self.prob_ary[ids] = prob  # here, ids means the indices of given children (maybe the right ones or left ones)
        self.prob_ary[ids] = self.abs_error_upper
        p_ids = (ids - 1) // 2

        while upper_step:  # propagate the change through tree
            ids = p_ids * 2 + 1  # in this while loop, ids means the indices of the left children
            self.prob_ary[p_ids] = self.prob_ary[ids] + self.prob_ary[ids + 1]
            p_ids = (p_ids - 1) // 2
            upper_step -= 1

        self.prob_ary[0] = self.prob_ary[1] + self.prob_ary[2]
        # because we take depth-1 upper steps, ps_tree[0] need to be updated alone

    def get_leaf_id(self, v):
        """Tree structure and array storage:
        Tree index:
              0       -> storing priority sum
            |  |
          1     2
         | |   | |
        3  4  5  6    -> storing priority for transitions
        Array type for storing: [0, 1, 2, 3, 4, 5, 6]
        """
        parent_idx = 0
        while True:
            l_idx = 2 * parent_idx + 1  # the leaf's left node
            r_idx = l_idx + 1  # the leaf's right node
            if l_idx >= (len(self.prob_ary)):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.prob_ary[l_idx]:
                    parent_idx = l_idx
                else:
                    v -= self.prob_ary[l_idx]
                    parent_idx = r_idx
        return min(leaf_idx, self.now_len - 2)  # leaf_idx

    def get_indices_is_weights(self, batch_size, beg, end):
        # self.per_beta = min(1., self.per_beta + 0.001)
        self.per_beta = min(1., self.per_beta + self.beta_increment_per_sampling)

        # get random values for searching indices with proportional prioritization
        values = (rd.rand(batch_size) + np.arange(batch_size)) * (self.prob_ary[0] / batch_size)

        # get proportional prioritization
        leaf_ids = np.array([self.get_leaf_id(v) for v in values])
        self.indices = leaf_ids - (self.memo_len - 1)

        prob_ary = self.prob_ary[leaf_ids] / self.prob_ary[beg:end].min()
        is_weights = np.power(prob_ary, -self.per_beta)  # important sampling weights
        return self.indices, is_weights

    def td_error_update(self, td_error):  # td_error = (q-q).detach_().abs()
        # prob = td_error.squeeze().clamp(1e-6, 10).pow(self.per_alpha)
        prob = td_error.squeeze().clamp(self.epsilon, self.abs_error_upper).pow(self.per_alpha)
        prob = prob.cpu().numpy()
        self.update_ids(self.indices, prob)

    # trade off
    def get_is_weight_TO(self, beg, end):
        leaf_ids = self.indices + (self.memo_len - 1)
        prob_ary = self.prob_ary[leaf_ids] / self.prob_ary[beg:end].min()
        is_weights = np.power(prob_ary, -self.per_beta)  # important sampling weights
        return is_weights