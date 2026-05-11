#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PybulletRunner 模块
=====================

本模块整合了基于 PyBullet 的多智能体强化学习训练与评估的整个流程。主要功能包括：

1. 从配置文件中加载所有超参数（例如随机种子、训练参数、网络超参数等）。
2. 初始化仿真环境（SensingEnv），并设置仿真设备（CPU 或 GPU）。
3. 初始化多智能体控制器（MADDPGController），该控制器实现了多智能体深度确定性策略梯度算法。
4. 训练循环：采样当前状态、利用控制器选择动作、将动作施加到环境、获取下一状态、计算奖励、存入经验回放缓冲区、周期性更新网络、记录日志、保存检查点。
5. 评估流程：在训练结束或指定时刻进行评估，切换到评估模式后执行无噪声动作，并输出相关指标。
6. 检查点管理与日志记录：保存模型权重至检查点目录，并通过 TensorBoard 记录各项指标。

注意：
    - 各部分均添加了详细中文注释，解释了代码每一步的功能。
    - 为保证观测向量尺寸统一，环境中各智能体的观测构造函数已采用动态计算并补齐（或截断）的方式，使输出严格等于配置文件中定义的 DIMENSION_OBS[0]。
    - 请确保配置文件（位于 HGAT-MADDPG_ver2/config/ 目录下）参数正确设置后再运行。

作者：YourName
日期：202X-XX-XX
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime
from os import path
from yaml import load, Loader
from torch.utils.tensorboard import SummaryWriter
import torch
import pybullet as p
import numpy as np
import json

# 导入项目内部模块：工具函数、控制器、经验回放、环境
from utils import get_load_path, latest_logdir
from controllers import MADDPGController
from rl.replay import Transition
from env.sensingEnv import SensingEnv
from env.utils import *

class PybulletRunner(ABC):
    """
    PybulletRunner 类
    -------------------
    本类整合了基于 PyBullet 的多智能体仿真环境与 MADDPG 控制器，
    实现训练与评估全流程，包括配置加载、环境初始化、训练循环、日志记录及检查点保存。
    """
    def __init__(self, resume_run, if_render, device='cpu'):
        """
        初始化 PybulletRunner 对象

        参数:
            resume_run: 布尔值，是否从之前的检查点恢复训练
            if_render: 布尔值，是否启用 PyBullet 的 GUI 渲染
            device: 设备类型，'cpu' 或 'cuda'
        """
        # 1. 加载配置文件，将 config 目录下所有 YAML 文件参数合并到 param_dict 中
        self.param_dict = {}
        config_dir = "HGAT-MADDPG_ver2/config"
        for file in os.listdir(config_dir):
            file_path = os.path.join(config_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                param_dict_current = load(f, Loader=Loader)
            self.param_dict.update(param_dict_current)
        
        self.device = device
        # 检查点文件目录：LOG_DIR/logs/
        self.checkpoint_file = path.join(self.param_dict["LOG_DIR"], 'logs/')
        self.step_num = 0
        self.episode_num = 0
        previous_step_num = 0

        # 2. 检查是否需要从之前的检查点恢复
        if resume_run:
            resume_path = get_load_path(self.checkpoint_file)
            print(f"Loading model from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=self.device)
            self.checkpoint_dir = latest_logdir(self.checkpoint_file)
        else:
            # 若不恢复，则创建新的检查点目录，目录名为当前日期时间格式
            self.checkpoint_dir = path.join(self.checkpoint_file, datetime.now().strftime('%Y%m%d-%H:%M:%S'))
            if not path.exists(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)
        
        self.step_num += previous_step_num

        # 3. 设置随机数种子（确保实验可重复）
        np.random.seed(self.param_dict["RANDOM_SEED"])
        torch.manual_seed(self.param_dict["RANDOM_SEED"])
        torch.cuda.manual_seed(self.param_dict["RANDOM_SEED"])
        # 创建原始信号点数据文件（用于环境中信号点数据初始化）
        save_path = "HGAT-MADDPG_ver2/env/data_signalPoint.npy"
        create_origData(save_path, self.param_dict["NUM_SIGNAL_POINT"], self.param_dict["RANDOM_SEED"])

        # 4. 初始化环境：构造 SensingEnv（基于 PyBullet），传入设备与渲染参数
        self.env = SensingEnv(self.device, render=if_render)
        # 5. 初始化 TensorBoard 日志记录器，日志存放在检查点目录下
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir)
        # 6. 定义智能体类型数组：本例中3架 UAV（类型 0）与2架充电器（类型 1）
        self.node_types = [0, 0, 0, 1, 1]
        # 7. 定义每隔多少个 episode 保存一次检查点
        self.checkpoint_save_episodes = 100
        # 8. 初始化多智能体控制器：使用 MADDPGController 实现深度确定性策略梯度算法
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
            self.param_dict["encoding_output_size"],
            self.param_dict["graph_module_sizes"],
            self.param_dict["action_hidden_size"],
            self.param_dict["SHARE_ENCODING"],
            self.param_dict["ACR_ENCODEING"],
            self.param_dict["ACT_COMMS"],
            self.param_dict["ACT_ACTION"],
            self.param_dict["GAMMA"],
            self.param_dict["TAU"],
            self.device,
            resume_run,
            self.param_dict["MEMORY_SIZE"],
            self.param_dict["full_receptive_field"],
            self.param_dict["gat_n_heads"],
            self.param_dict["gat_average_last"],
            self.param_dict["dropout"],
            self.param_dict["add_loops"]
        )

    def sample_from_memory(self):
        """
        从经验回放缓冲区中采样一个批次数据

        返回:
            批次数据
        """
        if not self.memory.is_prioritized:
            return self.memory.sample(self.batch_size)
        else:
            return self.memory.sample(self.batch_size, self.replay_buffer_beta)

    def maybe_backup_buffer(self):
        """
        备份一部分回放缓冲区数据到文件，防止数据丢失
        """
        print('Saving a sample of the replay buffer to file...')
        torch.save(self.memory.copy(), self.replay_buffer_file)

    def run(self):
        """
        训练主循环

        流程说明：
            1. 重复进行多个 episode，直至达到配置中 N_EPISODES；
            2. 每个 episode 内部：
                - 重置环境，获取初始状态和邻接矩阵；
                - 记录初始轨迹、初始化累计距离与能量消耗；
                - 每步：
                    * 控制器根据当前状态选择动作（添加噪声用于探索）；
                    * 将动作施加到环境，并调用 env.step() 进行仿真一步，获取下一状态、奖励、完成标志、能量消耗等；
                    * 将经验存入回放缓冲区；
                    * 在足够 episode 后，调用控制器的 update() 方法更新网络；
                    * 每隔一定步数进行目标网络软更新；
                    * 更新轨迹记录、累计距离等；
                    * 当任一智能体触发完成标志或达到最大步数时，结束当前 episode；
                - 记录 TensorBoard 指标并保存检查点。
        """
        last_eval = 0
        training_start = datetime.now()
        while self.episode_num < self.param_dict["N_EPISODES"]:
            step_start = self.step_num
            time_start = datetime.now()
            # 初始化累计移动距离及能量消耗数组
            self.total_distance_np = np.zeros(self.param_dict["NUM_DRONE"] + self.param_dict["NUM_CHARGER"])
            self.np_all_energy_consumption = np.zeros((self.param_dict["NUM_DRONE"], 3))

            # 重置环境，获取初始全局状态和邻接矩阵
            current_state, cur_adj = self.env.reset()
            # 保存初始信号点数据，用于后续计算公平性等指标
            data_orig = list(self.env.scene.signalPointId2data.values())
            # 获取所有智能体的初始位置，用于构造轨迹
            start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                        [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
            self.trajectory = [start_pos]
            self.last_pos = start_pos
            self.episode_num += 1

            total_critic_loss = np.array([0, 0], dtype=np.float32)
            total_policy_loss = np.array([0, 0], dtype=np.float32)
            episode_reward = np.zeros((self.param_dict["NUM_DRONE"] + self.param_dict["NUM_CHARGER"]))
            self.step_num = 0
            
            # 每个 episode 内部循环
            for i_step in range(self.param_dict["MAX_STEPS"]):
                self.step_num += 1
                # 1. 控制器根据当前状态与邻接矩阵选择动作（包含探索噪声）
                actions = self.controller.act(current_state, cur_adj, self.episode_num, self.param_dict["EPISODES_BEFORE_TRAIN"], True)
                last_actions = actions  # 可用于后续调试
                # 2. 清零所有智能体的速度，防止累积误差
                for robot in self.env.robot:
                    p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
                for charger in self.env.charger:
                    p.resetBaseVelocity(charger.robot, linearVelocity=[0., 0., 0.])
                # 3. 环境一步仿真，获得下一状态、邻接矩阵、奖励、done标志、能量消耗
                state_, adj_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(self.trajectory))
                self.np_all_energy_consumption += energy_consumption

                if i_step != self.param_dict["MAX_STEPS"] - 1:
                    next_state = state_
                    next_adj = adj_
                else:
                    next_state = None
                    next_adj = None

                episode_reward += reward
                # 4. 若非终止状态，将当前经验存入回放缓冲区，并更新状态
                if next_state is not None:
                    self.controller.memory.push(current_state, cur_adj, actions.cpu(), next_state, next_adj, reward, done, self.episode_num)
                    current_state = next_state
                    cur_adj = next_adj

                # 5. 当训练期数达到 EPISODES_BEFORE_TRAIN 后，开始进行网络更新
                if self.episode_num > self.param_dict["EPISODES_BEFORE_TRAIN"]:
                    critic_loss, policy_loss = self.controller.update(i_step, self.param_dict)
                    total_critic_loss += np.array([c.cpu().detach().numpy() for c in critic_loss])
                    if policy_loss is not None:
                        total_policy_loss += np.array([p.cpu().detach().numpy() for p in policy_loss])
                    # 6. 每隔一定步数执行目标网络软更新
                    if self.step_num % self.param_dict["SOFT_UPDATE_FREGUENCY"] == 0:
                        self.controller.update_target_net()

                # 7. 更新轨迹：获取当前所有智能体的位置，并累计移动距离
                current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                              [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
                current_distance_np = np.array([np.sqrt(sum([(x - y) ** 2 for x, y in zip(sp, ep)])) for sp, ep in zip(self.last_pos, current_pos)])
                self.total_distance_np += current_distance_np
                self.last_pos = current_pos
                self.trajectory.append(current_pos)
                # 8. 当任一智能体触发完成条件（done==1）或达到最大步数时，结束本 episode
                if sum(done) > 0 or i_step == (self.param_dict["MAX_STEPS"] - 1):
                    if self.episode_num % self.checkpoint_save_episodes == 0 and self.episode_num > self.param_dict["EPISODES_BEFORE_TRAIN"]:
                        self.controller.save_checkpoint(self.step_num, self.episode_num)

                    robot_data_sensed = sum([robot.dataSensed for robot in self.env.robot])
                    dataCollected_percentage = robot_data_sensed / self.env.scene.data_total
                    total_all_energy_consumption = self.np_all_energy_consumption.sum(axis=0)
                    total_energy_consumption_sensing = total_all_energy_consumption[1]
                    total_energy_consumption_moving = total_all_energy_consumption[2]
                    total_energy_consumption = total_all_energy_consumption[0]
                    energyEfficiency = total_energy_consumption_sensing / total_energy_consumption

                    data_final = list(self.env.scene.signalPointId2data.values())

                    if dataCollected_percentage >= 0.9:
                        self.controller.save_checkpoint(self.step_num, self.episode_num)
                    break

            # 9. 计算平均损失，并记录 TensorBoard 日志
            po_lo = total_policy_loss / (self.step_num * 2)
            cr_lo = total_critic_loss / self.step_num
            self.writer.add_scalar('Reward_sum', np.sum(episode_reward), self.episode_num)
            self.writer.add_scalar('Reward_1', episode_reward[0], self.episode_num)
            self.writer.add_scalar('Reward_2', episode_reward[1], self.episode_num)
            self.writer.add_scalar('Reward_3', episode_reward[2], self.episode_num)
            self.writer.add_scalar('Episode_length', self.step_num, self.episode_num)
            self.writer.add_scalar('Critic_loss_UAV', cr_lo[0], self.episode_num)
            self.writer.add_scalar('Policy_loss_UAV', po_lo[0], self.episode_num)
            self.writer.add_scalar('Critic_loss_charger', cr_lo[1], self.episode_num)
            self.writer.add_scalar('Policy_loss_charger', po_lo[1], self.episode_num)
            self.writer.add_scalar('dataCollected_percentage', dataCollected_percentage, self.episode_num)
            print(f'Episode:{self.episode_num}, step_num={self.step_num}, reward={episode_reward}, critic_loss={cr_lo}, policy_loss={po_lo}, dataCollected_percentage={dataCollected_percentage}, time={datetime.now() - time_start}')
            print("---------------------------------------------------------------------------")
            # 清理 CUDA 内存，防止内存泄漏
            torch.cuda.empty_cache()

    def evaluate(self):
        """
        评估模式

        流程说明：
            1. 将所有智能体的 actor 网络设置为评估模式（eval）；
            2. 重置环境，获取初始状态和轨迹；
            3. 在无噪声条件下连续执行动作，直至任一智能体完成任务；
            4. 输出数据采集比例、能源消耗、公平性等指标，便于观察模型性能。
        """
        # 设置 UAV 与 Charger 的 actor 网络为评估模式
        for i in range(2):
            self.controller.UAVAgent.actor[i].eval()
        self.controller.chargerAgent.actor[0].eval()
        # 重置环境，获取初始状态与邻接矩阵
        current_state, cur_adj = self.env.reset()
        data_orig = list(self.env.scene.signalPointId2data.values())
        start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                    [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
        self.trajectory = [start_pos]
        self.last_pos = start_pos
        with torch.no_grad():
            for i_step in range(self.param_dict["MAX_STEPS"]):
                # 选择动作（无探索噪声）
                actions = self.controller.act(current_state, cur_adj, self.episode_num, self.param_dict["EPISODES_BEFORE_TRAIN"], False)
                for robot in self.env.robot:
                    p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
                for charger in self.env.charger:
                    p.resetBaseVelocity(charger.robot, linearVelocity=[0., 0., 0.])
                state_, adj_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(self.trajectory))
                if i_step != self.param_dict["MAX_STEPS"] - 1:
                    next_state = state_
                    next_adj = adj_
                else:
                    next_state = None
                if next_state is not None:
                    current_state = next_state
                    cur_adj = next_adj
                current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                              [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
                self.last_pos = current_pos
                self.trajectory.append(current_pos)
                if sum(done) > 0:
                    robot_data_sensed = sum([robot.dataSensed for robot in self.env.robot])
                    dataCollected_percentage = robot_data_sensed / self.env.scene.data_total
                    print("dataCollected_percentage", dataCollected_percentage)
                    data_final = list(self.env.scene.signalPointId2data.values())
                    fair = fairness(data_orig, data_final)
                    print("fair", fair)
                    energy_usage = sum([robot.consumption_energy / (1 + robot.accumulated_charge_energy) for robot in self.env.robot]) / len(self.env.robot)
                    print("energy_usage", energy_usage)
                    for charger in self.env.charger:
                        print("charge_steps_ratio", charger.charge_steps / i_step)
                    accumulated_charge_energy_list = [robot.accumulated_charge_energy for robot in self.env.robot]
                    nor_accumulated_charge_energy_list = np.array([en / sum(accumulated_charge_energy_list) for en in accumulated_charge_energy_list])
                    print("fair_charge", sum(nor_accumulated_charge_energy_list)**2 / (len(nor_accumulated_charge_energy_list) * sum(nor_accumulated_charge_energy_list**2)))
                    break

if __name__ == "__main__":
    """
    主入口

    根据是否有 GPU 可用选择设备，并根据 train_mode（"train" 或 "test"）执行训练或评估流程。
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 设置训练模式为 "train" 或 "test"
    train_mode = "test"  # 可修改为 "test" 以执行评估模式
    
    if train_mode == "train":
        # 训练模式：不恢复检查点，不渲染环境
        runner = PybulletRunner(resume_run=False, if_render=False, device=device)
        try:
            runner.run()
        except (Exception, KeyboardInterrupt) as e:
            # 异常处理：若为 MADDPGController 则先保存检查点
            if isinstance(runner.controller, MADDPGController):
                print('发生错误，保存检查点...')
                runner.controller.save_checkpoint(runner.step_num, runner.episode_num)
            # 若异常非键盘中断，则写入日志后重新抛出
            if not isinstance(e, KeyboardInterrupt):
                with open(path.join('HGAT-MADDPG_ver2/data/', 'log.txt'), 'a') as f:
                    import traceback
                    f.write(str(e))
                    f.write(traceback.format_exc())
                raise e
    elif train_mode == "test":
        # 测试模式：恢复检查点，不渲染环境
        runner = PybulletRunner(resume_run=True, if_render=False, device=device)
        try:
            runner.evaluate()
        except (Exception, KeyboardInterrupt) as e:
            raise e
