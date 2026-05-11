#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hcanet-3.27_maddpg-MAAC/maac-main.py 模块
==========================================

本模块整合了基于 PyBullet 的多智能体强化学习训练与评估流程，
主要功能包括：
  1. 从配置文件中加载所有训练与环境超参数；
  2. 初始化仿真环境（SensingEnv）和多智能体控制器（MADDPGController，内部可能基于 AttentionSAC）；
  3. 支持训练模式与测试模式（test 模式下加载保存的分类型检查点模型并进行评估）；
  4. 日志记录、检查点保存与（可选）经验重放缓冲区加载；
  5. 在训练或评估时输出所有训练相关的指标信息，包括：
     - 总奖励、各 agent 奖励（例如 UAV 与 Charger 的奖励）；
     - Episode 长度（步数）；
     - Critic 与 Policy 网络的损失；
     - 数据采集百分比；
     - 能量消耗（感知能量、移动能量、总能量）；
     - 能量效率；
     - 充电步数比例；
     - 充电公平性；
     - 其它调试或监控信息。

【注意】
  - 在创建检查点目录时，会自动检查并创建所有父目录，避免因目录不存在而抛出异常。
  - 当 resume_run 为 True 时，会从指定日志目录中加载最新检查点，并依次加载各智能体类型的模型状态。
  - 在评估阶段，为防止除零错误，对累计充电能量为 0 的情况进行了特殊处理，默认充电公平性设为 0。
  - 若测试结果始终相同，可能是环境状态（例如数据采集、充电）未发生变化，请检查环境设计和奖励设计。

作者：YourName  
日期：202X-XX-XX
"""

import os
from abc import ABC
from datetime import datetime
from os import path
from yaml import load, Loader
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import pybullet as p
import json
from math import sqrt

# 导入项目内部模块：工具函数、控制器、环境、经验重放缓冲区等
from utils.utils import get_load_path, latest_logdir
from controllers import MADDPGController
from env.sensingEnv import SensingEnv
from env.utils import *  # 此处假定 env/utils.py 中包含 fairness() 等辅助函数
from utils.buffer import ReplayBuffer

# =============================================================================
# PybulletRunner 类定义
# =============================================================================
class PybulletRunner(ABC):
    def __init__(self, resume_run, if_render, device='cpu'):
        """
        初始化 PybulletRunner 类

        参数:
            resume_run (bool): 是否从已有检查点恢复（加载训练好的模型参数）
            if_render (bool): 是否启用 PyBullet 的图形界面渲染
            device (str): 使用的设备类型（例如 'cpu' 或 'cuda'）
        """
        # ---------------------加载配置文件---------------------
        self.param_dict = {}
        config_dir = "hcanet-3.27_maddpg-MAAC/config"
        for file in os.listdir(config_dir):
            file_path = path.join(config_dir, file)
            param_dict_current = load(open(file_path, "r", encoding="utf-8"), Loader=Loader)
            self.param_dict.update(param_dict_current)
        
        self.device = device
        # 检查点根目录，检查点存放于 LOG_DIR/logs/ 下（LOG_DIR 在配置文件中定义）
        self.checkpoint_file = path.join(self.param_dict["LOG_DIR"], 'logs/')
        self.step_num = 0
        self.episode_num = 0
        previous_step_num = 0

        # ---------------------设置智能体类型---------------------
        # 例如 [0, 0, 0, 1, 1] 表示 3 架 UAV（类型0）与 2 架充电器（类型1）
        self.node_types = [0, 0, 0, 1, 1]

        # ---------------------加载检查点与经验重放缓冲区---------------------
        self.checkpoint = {}  # 用于存储各 agent 类型对应的状态字典
        if resume_run:
            # 若恢复，则获取最新运行目录，并尝试加载各类型检查点
            self.checkpoint_dir = latest_logdir(self.checkpoint_file)
            print(f"Using checkpoint directory: {self.checkpoint_dir}")
            unique_types = set(self.node_types)
            for agent_type in unique_types:
                cp_path = path.join(self.checkpoint_dir, f"checkpoint_{self.episode_num}_type_{agent_type}.pth")
                if os.path.exists(cp_path):
                    print(f"Loading checkpoint for agent type {agent_type} from: {cp_path}")
                    self.checkpoint[f"controller_state_type_{agent_type}"] = torch.load(cp_path, map_location=self.device)
                else:
                    print(f"Warning: checkpoint file for agent type {agent_type} not found at {cp_path}.")
            # 尝试加载经验重放缓冲区
            try:
                self.replay_buffer = torch.load(f'{self.checkpoint_dir}/replay_buffer_{self.episode_num}.pth')
            except FileNotFoundError:
                print("No replay buffer found, initializing a new one.")
                self.replay_buffer = ReplayBuffer(
                    max_steps=self.param_dict["MEMORY_SIZE"],
                    num_agents=len(self.param_dict["DIMENSION_OBS"]),
                    obs_dims=[self.param_dict["DIMENSION_OBS"][i] for i in range(len(self.param_dict["DIMENSION_OBS"]))],
                    ac_dims=[self.param_dict["DIMENSION_ACTION"][i] for i in range(len(self.param_dict["DIMENSION_ACTION"]))],
                    agent_types=self.node_types
                )
        else:
            # 若不恢复，则新建检查点目录（目录名称为当前日期时间字符串），并初始化新的经验重放缓冲区
            self.checkpoint_dir = path.join(self.checkpoint_file, datetime.now().strftime('%Y%m%d-%H-%M-%S'))
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.replay_buffer = ReplayBuffer(
                max_steps=self.param_dict["MEMORY_SIZE"],
                num_agents=len(self.param_dict["DIMENSION_OBS"]),
                obs_dims=[self.param_dict["DIMENSION_OBS"][i] for i in range(len(self.param_dict["DIMENSION_OBS"]))],
                ac_dims=[self.param_dict["DIMENSION_ACTION"][i] for i in range(len(self.param_dict["DIMENSION_ACTION"]))],
                agent_types=self.node_types
            )
        self.step_num += previous_step_num

        # ---------------------设置随机数种子---------------------
        np.random.seed(self.param_dict["RANDOM_SEED"])
        torch.manual_seed(self.param_dict["RANDOM_SEED"])
        torch.cuda.manual_seed(self.param_dict["RANDOM_SEED"])
        # 生成环境初始信号点数据文件，用于环境中信号点数据初始化
        save_path = "hcanet-3.27_maddpg/env/data_signalPoint.npy"
        create_origData(save_path, self.param_dict["NUM_SIGNAL_POINT"], self.param_dict["RANDOM_SEED"])

        # ---------------------初始化环境与日志---------------------
        self.env = SensingEnv(self.device, render=if_render)
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir)

        # ---------------------初始化控制器---------------------
        self.controller = MADDPGController(
            self.checkpoint_file,
            self.checkpoint_dir, 
            self.param_dict["OPTIMIZER"],
            self.param_dict["CRITIC_LR"],
            self.param_dict["ACTOR_LR"],
            self.param_dict["WEIGHT_DECAY"],
            self.param_dict["RMSPROP_ALPHA"],
            self.param_dict["RMSPROP_EPS"],
            self.param_dict["NUM_DRONE"],
            self.param_dict["NUM_CHARGER"],
            self.node_types,
            self.param_dict["DIMENSION_OBS"],
            self.param_dict["DIMENSION_ACTION"],
            self.param_dict["GAMMA"],
            self.param_dict["TAU"],
            self.device,
            resume_run,
            self.param_dict["MEMORY_SIZE"]
        )
        
        # 如果恢复模式并且检查点中保存了状态，则加载控制器状态
        if resume_run and self.checkpoint:
            print("Loading controller state from checkpoint...")
            self.controller.load_state_dict(self.checkpoint)
        else:
            print("No previous checkpoint loaded. Starting from scratch.")

    # =============================================================================
    # 训练主循环 run() 方法
    # =============================================================================
    def run(self):
        """
        训练主循环

        流程说明：
          1. 循环执行多个 episode，直至达到配置中设定的 N_EPISODES；
          2. 每个 episode 内部：
             - 重置环境，获取初始状态和初始轨迹；
             - 在每一步中：由控制器选择动作（包含探索噪声）、执行环境仿真、存储经验、模型更新、目标网络软更新以及轨迹记录；
             - 当任一智能体触发结束标志或达到最大步数时退出本 episode；
             - 记录 TensorBoard 指标，并按设定间隔保存检查点；
             - 输出训练过程中的所有关键信息和指标。
        """
        while self.episode_num < self.param_dict["N_EPISODES"]:
            # 重置环境，获取初始全局状态和初始轨迹
            current_state = self.env.reset()
            # 保存初始信号点数据，用于后续计算公平性等指标
            data_orig = list(self.env.scene.signalPointId2data.values())
            self.episode_num += 1
            episode_reward = np.zeros(len(self.node_types))
            self.step_num = 0

            # 获取所有智能体初始位置，构造轨迹列表
            start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                        [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
            trajectory = [start_pos]

            # 初始化能量消耗统计数组（针对 NUM_DRONE 个 UAV，每个 UAV 记录 [总能量, 感知能量, 移动能量]）
            self.np_all_energy_consumption = np.zeros((self.param_dict["NUM_DRONE"], 3))

            # ---------------------每步循环---------------------
            for i_step in range(self.param_dict["MAX_STEPS"]):
                self.step_num += 1

                # 控制器根据当前状态选择动作（训练时添加探索噪声）
                actions = self.controller.act(current_state, self.episode_num, self.param_dict["EPISODES_BEFORE_TRAIN"], True)
                # 重置每个智能体速度，防止累积物理误差
                for robot in self.env.robot:
                    p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
                for charger in self.env.charger:
                    p.resetBaseVelocity(charger.robot, linearVelocity=[0., 0., 0.])
                # 环境执行一步仿真，返回下一个状态、奖励、完成标志及能量消耗
                state_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(trajectory))
                self.np_all_energy_consumption += energy_consumption

                next_state = state_ if i_step != self.param_dict["MAX_STEPS"] - 1 else None
                episode_reward += reward
                if next_state is not None:
                    # 将经验存入经验重放缓冲区
                    self.controller.memory.push(current_state, actions.cpu(), next_state, reward, done, self.episode_num)
                    current_state = next_state

                # 当达到训练阶段后，执行模型更新与目标网络软更新
                if self.episode_num > self.param_dict["EPISODES_BEFORE_TRAIN"]:
                    critic_loss, policy_loss = self.controller.update(i_step, self.param_dict)
                    # 除了损失累加外，也可输出当前步的损失（如有需要，可在调试时打开相关打印）
                    if self.step_num % self.param_dict["SOFT_UPDATE_FREGUENCY"] == 0:
                        self.controller.update_target_net()

                # 更新轨迹记录
                current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                              [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
                trajectory.append(current_pos)

                # 如果任一智能体结束或达到最大步数，则退出本 episode
                if sum(done) > 0 or i_step == (self.param_dict["MAX_STEPS"] - 1):
                    # 记录 TensorBoard 指标（所有训练相关指标）
                    self.writer.add_scalar('Reward_sum', np.sum(episode_reward), self.episode_num)
                    # 假设 MADDPGController 内部记录了各个 agent 类型的损失信息（可根据需要扩展）
                    # 此处可增加其它指标记录，例如每步平均损失、平均奖励、步数等
                    self.writer.add_scalar('Episode_length', self.step_num, self.episode_num)
                    self.writer.add_scalar('Total_Steps', self.step_num + (self.episode_num - 1) * self.param_dict["MAX_STEPS"], self.episode_num)
                    
                    # 计算数据采集百分比（数据采集量 / 信号点数据总量）
                    robot_data_sensed = sum([robot.dataSensed for robot in self.env.robot])
                    dataCollected_percentage = robot_data_sensed / self.env.scene.data_total
                    # 能量消耗统计：分为感知能量、移动能量和总能量
                    total_all_energy_consumption = self.np_all_energy_consumption.sum(axis=0)
                    total_energy_consumption_sensing = total_all_energy_consumption[1]
                    total_energy_consumption_moving = total_all_energy_consumption[2]
                    total_energy_consumption = total_all_energy_consumption[0]
                    energyEfficiency = total_energy_consumption_sensing / total_energy_consumption if total_energy_consumption != 0 else 0.0

                    data_final = list(self.env.scene.signalPointId2data.values())

                    # 如果数据采集比例超过阈值，则保存检查点
                    if dataCollected_percentage >= 0.85:
                        self.controller.save_checkpoint(self.step_num, self.episode_num)

                    # 输出所有训练相关指标：
                    print("============================================")
                    print(f"Episode: {self.episode_num}")
                    print(f"Steps in Episode: {self.step_num}")
                    print(f"Total Reward: {np.sum(episode_reward)}")
                    # 如果需要输出分 agent 的奖励，可以逐一输出：
                    for i, r in enumerate(episode_reward):
                        print(f"Reward for Agent {i}: {r}")
                    print(f"Data Collected Percentage: {dataCollected_percentage}")
                    print(f"Total Energy Consumption: {total_energy_consumption}")
                    print(f"Energy Consumption (Sensing): {total_energy_consumption_sensing}")
                    print(f"Energy Consumption (Moving): {total_energy_consumption_moving}")
                    print(f"Energy Efficiency: {energyEfficiency}")
                    # 输出充电相关指标：充电步数比例（按每个充电器计算）
                    for charger in self.env.charger:
                        print(f"Charge Steps Ratio for Charger (steps/episode): {charger.charge_steps / self.step_num}")
                    # 计算充电公平性，防止除零
                    accumulated_charge_energy_list = [robot.accumulated_charge_energy for robot in self.env.robot]
                    total_accumulated = sum(accumulated_charge_energy_list)
                    if total_accumulated == 0:
                        nor_accumulated_charge_energy_list = np.zeros(len(accumulated_charge_energy_list))
                        fair_charge = 0.0
                    else:
                        nor_accumulated_charge_energy_list = np.array([en / total_accumulated for en in accumulated_charge_energy_list])
                        sum_sq = sum(nor_accumulated_charge_energy_list ** 2)
                        fair_charge = (sum(nor_accumulated_charge_energy_list) ** 2) / (len(nor_accumulated_charge_energy_list) * sum_sq) if sum_sq != 0 else 0.0
                    print(f"Fair Charge: {fair_charge}")
                    # 计算其他指标（例如 critic_loss、policy_loss 等，如控制器内部有记录则输出）
                    print("============================================")
                    break

            # 清空 CUDA 缓存，防止内存泄漏
            torch.cuda.empty_cache()

    # =============================================================================
    # 评估模式 evaluate() 方法
    # =============================================================================
    def evaluate(self):
        """
        评估模式

        流程说明：
          1. 将所有智能体的 actor 网络切换为评估模式（eval()）；
          2. 重置环境，获取初始状态与轨迹；
          3. 在无探索噪声条件下连续执行动作，直至任一智能体触发结束标志；
          4. 仅执行状态转换，不进行模型更新。
        """
        # 切换 UAV 和 Charger 的 actor 网络为评估模式
        for i in range(2):
            self.controller.UAVAgent.actor[i].eval()
        self.controller.chargerAgent.actor[0].eval()
        current_state = self.env.reset()
        # 保存初始信号点数据，用于后续计算公平性
        data_orig = list(self.env.scene.signalPointId2data.values())
        with torch.no_grad():
            for i_step in range(self.param_dict["MAX_STEPS"]):
                actions = self.controller.act(current_state, self.episode_num, self.param_dict["EPISODES_BEFORE_TRAIN"], False)
                for robot in self.env.robot:
                    p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
                for charger in self.env.charger:
                    p.resetBaseVelocity(charger.robot, linearVelocity=[0., 0., 0.])
                state_, _, done, _ = self.env.step(actions, i_step, np.array([]))
                next_state = state_ if i_step != self.param_dict["MAX_STEPS"] - 1 else None
                if next_state is not None:
                    current_state = next_state
                if sum(done) > 0:
                    break

    # =============================================================================
    # 随机策略模式 random() 方法
    # =============================================================================
    def random(self):
        """
        随机策略模式，用于测试环境与模型接口

        流程说明：
          1. 重置环境，获取初始状态与轨迹；
          2. 分别调用 UAV 和 Charger 的贪婪策略（greedy_action）获取动作；
          3. 执行环境 step 更新状态与轨迹，直至任一智能体结束任务；
          4. 输出评估指标：数据采集百分比、能量消耗、充电步数比例及充电公平性等。
        """
        current_state = self.env.reset()
        data_orig = list(self.env.scene.signalPointId2data.values())
        start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                    [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
        trajectory = [start_pos]
        num_agents = self.param_dict["NUM_DRONE"] + self.param_dict["NUM_CHARGER"]
        action_dim = self.param_dict["DIMENSION_ACTION"][0]
        actions = torch.zeros((num_agents, action_dim), device=self.device)
        with torch.no_grad():
            for i_step in range(self.param_dict["MAX_STEPS"]):
                for i, robot in enumerate(self.env.robot):
                    actions[i] = robot.greedy_action()
                for i, charger in enumerate(self.env.charger):
                    actions[i + self.param_dict["NUM_DRONE"]] = charger.greedy_action(None, None)
                state_, reward, done, _ = self.env.step(actions, i_step, np.array(trajectory))
                trajectory.append([list(p.getBasePositionAndOrientation(obj.robot)[0]) for obj in self.env.robot + self.env.charger])
                current_state = state_
                if sum(done) > 0:
                    break

    # =============================================================================
    # 保存检查点 save_checkpoint() 方法
    # =============================================================================
    def save_checkpoint(self):
        """
        保存当前模型状态和经验重放缓冲区

        对于每个智能体类型，分别保存对应状态字典到文件，文件名格式：
           checkpoint_<episode>_type_<agent_type>.pth
        同时保存整个经验重放缓冲区到 replay_buffer_<episode>.pth。
        """
        print('Saving checkpoint...')
        for agent_type in set(self.node_types):
            # 获取对应 agent_type 的状态字典，假设控制器提供 get_state_by_type 方法
            state_dict = self.controller.get_state_by_type(agent_type)
            cp_path = f'{self.checkpoint_dir}/checkpoint_{self.episode_num}_type_{agent_type}.pth'
            torch.save(state_dict, cp_path)
            print(f"Saved checkpoint for agent type {agent_type} at {cp_path}")
        rb_path = f'{self.checkpoint_dir}/replay_buffer_{self.episode_num}.pth'
        torch.save(self.replay_buffer, rb_path)
        print(f"Saved replay buffer at {rb_path}")

# =============================================================================
# 主入口
# =============================================================================
if __name__ == "__main__":
    """
    主入口

    根据设备情况选择 'cuda:0' 或 'cpu'，并依据 train_mode 参数选择训练、评估或随机策略模式。
    在 test 模式下，会调用 evaluate() 方法使用载入的训练完成模型进行测试。
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # train_mode 可设为 "train", "test", 或 "random"
    train_mode = "train"
    if train_mode == "train":
        runner = PybulletRunner(resume_run=False, if_render=False, device=device)
        runner.run()
    elif train_mode == "test":
        runner = PybulletRunner(resume_run=True, if_render=False, device=device)
        runner.evaluate()
    elif train_mode == "random":
        runner = PybulletRunner(resume_run=False, if_render=True, device=device)
        runner.random()
