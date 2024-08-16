import os
from abc import ABC
from datetime import datetime
from os import path
from yaml import load, Loader
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from utils.utils import *
from env.utils import *
from env.sensingEnv import SensingEnv
from attention_sac import AttentionSAC
from utils.buffer import ReplayBuffer  # 导入经验重放缓冲区类

class PybulletRunner(ABC):
    def __init__(self, resume_run, if_render, device='gpu'):
        """
        初始化PybulletRunner类，加载配置文件，设置环境和控制器，初始化日志和经验重放缓冲区。

        参数:
            resume_run (bool): 是否从先前的检查点恢复运行
            if_render (bool): 是否渲染环境
            device (str): 使用的设备类型 ('cpu' 或 'gpu')
        """
        # 加载配置文件
        self.param_dict = {}
        config_dir = "hcanet-3.27_maddpg-MAAC/config"
        for file in os.listdir(config_dir):
            file_path = path.join(config_dir, file)
            param_dict_current = load(open(file_path, "r", encoding="utf-8"), Loader=Loader)
            self.param_dict.update(param_dict_current)

        self.device = device
        self.checkpoint_file = path.join(self.param_dict["LOG_DIR"], 'logs/')
        self.step_num = 0
        self.episode_num = 0
        previous_step_num = 0
        
        # 定义智能体类型
        self.node_types = [0, 0, 1]  # 2架UAV+1架移动充电站

        # 判断是否需要从之前的训练中恢复
        if resume_run:
            resume_path = get_load_path(self.checkpoint_file)
            print(f"Loading model from: {resume_path}")
            torch.load(resume_path, map_location=self.device)
            self.checkpoint_dir = latest_logdir(self.checkpoint_file)
            # 尝试加载重放缓冲区
            try:
                self.replay_buffer = torch.load(f'{self.checkpoint_dir}/replay_buffer_{self.episode_num}.pth')
            except FileNotFoundError:
                print("No replay buffer found, initializing a new one.")
                self.replay_buffer = ReplayBuffer(
                    max_steps=self.param_dict["MEMORY_SIZE"],
                    num_agents=len(self.node_types),
                    obs_dims=[self.param_dict["DIMENSION_OBS"][i] for i in self.node_types],
                    ac_dims=[self.param_dict["DIMENSION_ACTION"][i] for i in self.node_types],
                    agent_types = self.node_types
                )
        else:
            self.checkpoint_dir = path.join(self.checkpoint_file, datetime.now().strftime('%Y%m%d-%H:%M:%S'))
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.replay_buffer = ReplayBuffer(
                max_steps=self.param_dict["MEMORY_SIZE"],
                num_agents=len(self.node_types),
                obs_dims=[self.param_dict["DIMENSION_OBS"][i] for i in self.node_types],
                ac_dims=[self.param_dict["DIMENSION_ACTION"][i] for i in self.node_types],
                agent_types = self.node_types
            )

        self.step_num += previous_step_num

        # 设置随机数种子
        np.random.seed(self.param_dict["RANDOM_SEED"])
        torch.manual_seed(self.param_dict["RANDOM_SEED"])
        torch.cuda.manual_seed(self.param_dict["RANDOM_SEED"])

        # 创建数据文件
        save_path = "hcanet-3.27_maddpg-MAAC/env/data_signalPoint.npy"
        create_origData(save_path, self.param_dict["NUM_SIGNAL_POINT"], self.param_dict["RANDOM_SEED"])

        # 初始化环境和日志
        self.env = SensingEnv(self.device, render=if_render)
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir)

        # 配置状态和动作空间
        dim_obs_list = [self.param_dict["DIMENSION_OBS"][i] for i in range(len(self.param_dict["DIMENSION_OBS"]))]
        dim_act_list = [self.param_dict["DIMENSION_ACTION"][i] for i in range(len(self.param_dict["DIMENSION_ACTION"]))]

        # 为不同的agent类型创建独立的AttentionSAC控制器
        self.controllers = {}
        for node_type in set(self.node_types):
            agent_init_params = []
            sa_size = []

            for i, nt in enumerate(self.node_types):
                if nt == node_type:
                    agent_init_params.append({'num_in_pol': dim_obs_list[nt], 'num_out_pol': dim_act_list[nt]})
                    sa_size.append((dim_obs_list[nt], dim_act_list[nt]))

            self.controllers[node_type] = AttentionSAC(
                agent_init_params=agent_init_params,
                sa_size=sa_size,
                agent_types=[node_type] * len(agent_init_params),
                gamma=self.param_dict["GAMMA"],
                tau=self.param_dict["TAU"],
                pi_lr=self.param_dict["ACTOR_LR"],
                q_lr=self.param_dict["CRITIC_LR"],
                reward_scale=10.0,
                pol_hidden_dim=128,
                critic_hidden_dim=128,
                attend_heads=4,
                device=self.device
            )

    def run(self):
        """
        运行强化学习训练过程，包含环境交互、存储经验、模型更新和日志记录。
        """
        while self.episode_num < self.param_dict["N_EPISODES"]:
            current_state = self.env.reset()
            self.episode_num += 1
            episode_reward = np.zeros(len(self.node_types))
            self.step_num = 0

            # 初始化轨迹信息
            start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                        [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
            trajectory = [start_pos]  # 记录初始位置
            last_pos = start_pos  # 上一步的位置

            self.np_all_energy_consumption = np.zeros((self.param_dict["NUM_DRONE"], 3))

            for i_step in range(self.param_dict["MAX_STEPS"]):
                self.step_num += 1

                # 获取动作
                actions = []
                for node_type, controller in self.controllers.items():
                    actions.extend(controller.step(
                        [current_state[i] for i, nt in enumerate(self.node_types) if nt == node_type],
                        explore=True
                    ))

                # 执行动作并获取新的状态和奖励
                next_state, reward, done, energy_consumption = self.env.step(actions, self.step_num, np.array(trajectory))
                episode_reward += reward

                self.np_all_energy_consumption += energy_consumption

                # 存储经验到重放缓冲区
                self.replay_buffer.push(
                    np.array([state.cpu().numpy() for state in current_state]),  # 将张量移到CPU并转换为NumPy数组
                    np.array([action.cpu().numpy() for action in actions]),
                    np.array(reward),
                    np.array([next.cpu().numpy() for next in next_state]),
                    np.array(done)
                )


                # 更新模型
                if self.episode_num > self.param_dict["EPISODES_BEFORE_TRAIN"]:
                    sampled_data = self.replay_buffer.sample(self.param_dict["BATCH_SIZE"], to_gpu=(self.device == 'cuda'))
                    for node_type, controller in self.controllers.items():
                        if node_type in sampled_data:
                            controller.update_critic(sampled_data[node_type])
                            controller.update_policies(sampled_data[node_type])

                # 更新轨迹
                current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                            [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
                trajectory.append(current_pos)  # 记录当前的位置信息

                # 转移到下一个状态
                current_state = next_state

                # 更新目标网络
                if self.step_num % self.param_dict["SOFT_UPDATE_FREGUENCY"] == 0:
                    for controller in self.controllers.values():
                        controller.update_all_targets()

                if sum(done) > 0 or i_step == (self.param_dict["MAX_STEPS"] - 1):
                    # 计算数据收集率和能量效率
                    robot_data_sensed = sum([robot.dataSensed for robot in self.env.robot])
                    dataCollected_percentage = robot_data_sensed / self.env.scene.data_total
                    total_all_energy_consumption = self.np_all_energy_consumption.sum(axis=0)
                    total_energy_consumption_sensing = total_all_energy_consumption[1]
                    total_energy_consumption_moving = total_all_energy_consumption[2]
                    total_energy_consumption = total_all_energy_consumption[0]
                    energyEfficiency = total_energy_consumption_sensing / total_energy_consumption

                    # 记录日志
                    self.writer.add_scalar('Reward_sum', np.sum(episode_reward), self.episode_num)
                    self.writer.add_scalar('Reward_UAV', episode_reward[0], self.episode_num)
                    self.writer.add_scalar('Reward_charger', episode_reward[1], self.episode_num)
                    self.writer.add_scalar('Episode_length', self.step_num, self.episode_num)
                    self.writer.add_scalar('dataCollected_percentage', dataCollected_percentage, self.episode_num)
                    self.writer.add_scalar('Total_Energy_Consumption', total_energy_consumption, self.episode_num)
                    self.writer.add_scalar('EnergyEfficiency', energyEfficiency, self.episode_num)
                    self.writer.add_scalar('Energy_Consumption_Sensing', total_energy_consumption_sensing, self.episode_num)
                    self.writer.add_scalar('Energy_Consumption_Moving', total_energy_consumption_moving, self.episode_num)

                    print(f'Episode:{self.episode_num}, reward={np.sum(episode_reward)}, '
                          f'dataCollected_percentage={dataCollected_percentage}, '
                          f'energy_consumption={total_energy_consumption}, '
                          f'energyEfficiency={energyEfficiency}, ')

                    if self.episode_num % self.param_dict["CHECKPOINT_SAVE_INTERVAL"] == 0:
                        self.save_checkpoint()
                    break

            torch.cuda.empty_cache()

    def save_checkpoint(self):
        """
        保存当前模型和经验重放缓冲区的状态到文件。
        """
        print('Saving checkpoint...')
        for node_type, controller in self.controllers.items():
            controller.save(f'{self.checkpoint_dir}/checkpoint_{self.episode_num}_type_{node_type}.pth')
        # 保存重放缓冲区
        torch.save(self.replay_buffer, f'{self.checkpoint_dir}/replay_buffer_{self.episode_num}.pth')

    def evaluate(self):
        """
        评估模型的性能，不进行模型更新，只记录结果。
        """
        current_state = self.env.reset()
        with torch.no_grad():
            for i_step in range(self.param_dict["MAX_STEPS"]):
                actions = []
                for node_type, controller in self.controllers.items():
                    actions.extend(controller.step(
                        [current_state[i] for i, nt in enumerate(self.node_types) if nt == node_type],
                        explore=False
                    ))
                next_state, _, done, _ = self.env.step(actions, self.step_num, np.array([]))  # 注意此处轨迹已不再需要
                current_state = next_state

                if sum(done) > 0:
                    break


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 模式选择：train, test, random
    train_mode = "test"
    if train_mode == "train":
        runner = PybulletRunner(resume_run=False, if_render=False, device=device)
        runner.run()
    elif train_mode == "test":
        runner = PybulletRunner(resume_run=True, if_render=False, device=device)
        runner.evaluate()
