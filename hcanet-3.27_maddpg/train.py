#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hcanet-3.27_maddpg/train.py 模块
======================================

本模块整合了基于 PyBullet 的多智能体强化学习训练与评估流程，
主要功能包括：
  1. 从配置文件中加载所有训练与环境超参数；
  2. 初始化仿真环境（SensingEnv）和多智能体控制器（MADDPGController）；
  3. 执行训练循环：状态采集、动作选择、环境交互、经验存储、网络更新、目标网络软更新、轨迹记录、日志记录及检查点保存；
  4. 执行评估流程，记录各项评估指标；
  5. 提供随机策略模式，用于测试环境与模型接口。

【注意】
  - 在创建检查点目录时，会自动检查并创建所有父目录，避免因目录不存在而抛出异常。
  - 在评估阶段计算“充电公平性”时，对累计充电能量总和为 0 的情况进行了防除零处理。如果所有机器人均无累积充电能量，则默认公平性值为 0。
  - 关于“为什么加载了不同的模型结果一模一样”：这通常说明测试过程中环境状态（例如累计充电能量、数据采集比例等）没有发生改变（比如测试过程中没有发生充电），所以模型对这些指标没有产生区别。请检查环境设计、奖励设计和测试流程是否真正反映了模型差异。

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
from math import sqrt

# 导入项目内部模块：工具函数、控制器、经验回放、环境以及其它辅助函数
from utils import get_load_path, latest_logdir
from controllers import MADDPGController
from rl.replay import Transition
from env.sensingEnv import SensingEnv
from env.utils import *

# th.autograd.set_detect_anomaly(True)

class PybulletRunner(ABC):
    """
    PybulletRunner 类
    --------------------
    本类负责加载配置、初始化仿真环境与多智能体控制器，执行训练、评估及随机测试流程，
    同时记录日志与保存模型检查点。
    """
    def __init__(self, resume_run, if_render, device='cpu'):
        """
        初始化 PybulletRunner 对象

        参数:
            resume_run: 布尔值，是否从已有检查点恢复训练（加载模型参数）
            if_render: 布尔值，是否启用 PyBullet 的图形界面渲染
            device: 设备（例如 'cpu' 或 'cuda'）
        """
        self.param_dict = {}
        # 遍历配置文件目录，加载所有 YAML 配置并合并到 param_dict 中
        for file in os.listdir("hcanet-3.27_maddpg/config"):
            paths = os.path.join("hcanet-3.27_maddpg/config", file)
            param_dict_current = load(open(paths, "r", encoding="utf-8"), Loader=Loader)
            self.param_dict.update(param_dict_current)
        
        self.device = device
        # 检查点目录由配置文件中 LOG_DIR 指定，此处检查点存放在 LOG_DIR/logs/ 目录下
        self.checkpoint_file = path.join(self.param_dict["LOG_DIR"], 'logs/')
        self.step_num = 0
        self.episode_num = 0
        previous_step_num = 0

        # 若 resume_run 为 True，则加载最新的检查点
        if resume_run:
            resume_path = get_load_path(self.checkpoint_file)
            print(f"Using custom model path: {resume_path}")
            print(f"Loading model from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=self.device)
            self.checkpoint_dir = latest_logdir(self.checkpoint_file)
        else:
            # 否则新建检查点目录（名称为当前日期时间格式），并确保父目录存在
            self.checkpoint_dir = path.join(self.checkpoint_file, datetime.now().strftime('%Y%m%d-%H-%M-%S'))
            if not path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.step_num += previous_step_num

        # 设置随机数种子，保证实验结果可重复
        np.random.seed(self.param_dict["RANDOM_SEED"])
        torch.manual_seed(self.param_dict["RANDOM_SEED"])
        torch.cuda.manual_seed(self.param_dict["RANDOM_SEED"])
        # 生成初始信号点数据文件（用于环境中信号点数据初始化）
        save_path = "hcanet-3.27_maddpg/env/data_signalPoint.npy"
        create_origData(save_path, self.param_dict["NUM_SIGNAL_POINT"], self.param_dict["RANDOM_SEED"])

        # 初始化仿真环境（SensingEnv），传入设备和是否渲染的标志
        self.env = SensingEnv(self.device, render=if_render)
        # 初始化 TensorBoard 日志记录器，日志保存路径为检查点目录
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir)
        # 定义智能体类型数组，例如 [0, 0, 0, 1, 1] 表示 3 架 UAV（类型 0）与 2 架充电器（类型 1）
        self.node_types = [0, 0, 0, 1, 1]
        # 每隔多少个 episode 保存一次检查点
        self.checkpoint_save_episodes = 100
        # 初始化 MADDPG 控制器，传入所有必要参数
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
      
    def sample_from_memory(self):
        """
        从经验回放缓冲区中采样一个批次

        返回:
            采样得到的批次数据
        """
        return self.memory.sample(self.batch_size) if not self.memory.is_prioritized else self.memory.sample(self.batch_size, self.replay_buffer_beta)

    def maybe_backup_buffer(self):
        """
        备份一部分回放缓冲区数据到文件，以防数据丢失
        """
        print('Saving a sample of the replay buffer to file...')
        torch.save(self.memory.copy(), self.replay_buffer_file)

    def run(self):
        """
        训练主循环

        流程说明：
          1. 重复执行多个 episode，直至达到配置中 N_EPISODES；
          2. 每个 episode 内部：
              - 重置环境，获取初始状态和轨迹；
              - 循环每步：选择动作、环境一步仿真、经验存储、网络更新、目标网络软更新、轨迹更新；
              - 当任一智能体触发完成标志或达到最大步数时结束本 episode；
              - 记录 TensorBoard 指标，并按一定频率保存检查点。
        """
        last_eval = 0
        training_start = datetime.now()
        while self.episode_num < self.param_dict["N_EPISODES"]:
            step_start = self.step_num
            time_start = datetime.now()
            # 初始化累计移动距离与能量消耗数组
            self.total_distance_np = np.zeros(self.param_dict["NUM_DRONE"] + self.param_dict["NUM_CHARGER"])
            self.np_all_energy_consumption = np.zeros((self.param_dict["NUM_DRONE"], 3))

            # 重置环境，获取初始状态
            current_state = self.env.reset()
            # 保存初始信号点数据，用于后续计算公平性等指标
            data_orig = list(self.env.scene.signalPointId2data.values())
            # 获取所有智能体的初始位置，构造轨迹列表
            start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                        [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
            self.trajectory = [start_pos]
            self.last_pos = start_pos
            self.episode_num += 1

            total_critic_loss = np.array([0, 0], dtype=np.float32)
            total_policy_loss = np.array([0, 0], dtype=np.float32)
            episode_reward = np.zeros((self.param_dict["NUM_DRONE"] + self.param_dict["NUM_CHARGER"]))
            self.step_num = 0

            for i_step in range(self.param_dict["MAX_STEPS"]):
                self.step_num += 1

                # 控制器根据当前状态选择动作（包含探索噪声）
                actions = self.controller.act(current_state, self.episode_num, self.param_dict["EPISODES_BEFORE_TRAIN"], True)
                last_actions = actions  # 记录当前动作（调试用）
                # 重置所有智能体速度，防止累计误差
                for robot in self.env.robot:
                    p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
                for charger in self.env.charger:
                    p.resetBaseVelocity(charger.robot, linearVelocity=[0., 0., 0.])
                # 环境执行一步仿真，返回下一状态、奖励、完成标志及能量消耗
                state_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(self.trajectory))
                self.np_all_energy_consumption += energy_consumption

                next_state = state_ if i_step != self.param_dict["MAX_STEPS"] - 1 else None
                episode_reward += reward
                if next_state is not None:
                    self.controller.memory.push(current_state, actions.cpu(), next_state, reward, done, self.episode_num)
                    current_state = next_state

                if self.episode_num > self.param_dict["EPISODES_BEFORE_TRAIN"]:
                    critic_loss, policy_loss = self.controller.update(i_step, self.param_dict)
                    total_critic_loss += np.array([c.cpu().detach().numpy() for c in critic_loss])
                    if policy_loss is not None:
                        total_policy_loss += np.array([p.cpu().detach().numpy() for p in policy_loss])
                    if self.step_num % self.param_dict["SOFT_UPDATE_FREGUENCY"] == 0:
                        self.controller.update_target_net()

                # 更新轨迹记录
                current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                              [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
                current_distance_np = np.array([sqrt(sum([(x - y) ** 2 for x, y in zip(sp, ep)])) for sp, ep in zip(self.last_pos, current_pos)])
                self.total_distance_np += current_distance_np
                self.last_pos = current_pos
                self.trajectory.append(current_pos)
                if sum(done) > 0 or i_step == (self.param_dict["MAX_STEPS"] - 1):
                    if self.episode_num % self.checkpoint_save_episodes == 0 and self.episode_num > self.param_dict["EPISODES_BEFORE_TRAIN"]:
                        self.controller.save_checkpoint(self.step_num, self.episode_num)

                    # 计算数据采集百分比
                    robot_data_sensed = sum([robot.dataSensed for robot in self.env.robot])
                    dataCollected_percentage = robot_data_sensed / self.env.scene.data_total
                    total_all_energy_consumption = self.np_all_energy_consumption.sum(axis=0)
                    total_energy_consumption_sensing = total_all_energy_consumption[1]
                    total_energy_consumption_moving = total_all_energy_consumption[2]
                    total_energy_consumption = total_all_energy_consumption[0]
                    energyEfficiency = total_energy_consumption_sensing / total_energy_consumption

                    data_final = list(self.env.scene.signalPointId2data.values())

                    if dataCollected_percentage >= 0.85:
                        self.controller.save_checkpoint(self.step_num, self.episode_num)
                    break

            po_lo = total_policy_loss / self.step_num
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

            torch.cuda.empty_cache()

    def evaluate(self):
        """
        评估模式

        流程说明：
          1. 将所有智能体的 actor 网络切换为评估模式；
          2. 重置环境，获取初始状态和轨迹；
          3. 在无噪声条件下连续执行动作，直至任一智能体触发完成条件；
          4. 输出数据采集百分比、能源消耗和公平性指标等。
        """
        # 切换 UAV 与 Charger 的 actor 网络至评估模式
        for i in range(2):
            self.controller.UAVAgent.actor[i].eval()
        self.controller.chargerAgent.actor[0].eval()
        current_state = self.env.reset()
        data_orig = list(self.env.scene.signalPointId2data.values())
        start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                    [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
        self.trajectory = [start_pos]
        self.last_pos = start_pos
        with torch.no_grad():
            for i_step in range(self.param_dict["MAX_STEPS"]):
                actions = self.controller.act(current_state, self.episode_num, self.param_dict["EPISODES_BEFORE_TRAIN"], False)
                for robot in self.env.robot:
                    p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
                for charger in self.env.charger:
                    p.resetBaseVelocity(charger.robot, linearVelocity=[0., 0., 0.])
                state_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(self.trajectory))
                next_state = state_ if i_step != self.param_dict["MAX_STEPS"] - 1 else None
                if next_state is not None:
                    current_state = next_state
                current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                              [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
                self.last_pos = current_pos
                self.trajectory.append(current_pos)
                if sum(done) > 0:
                    # 计算数据采集百分比
                    robot_data_sensed = sum([robot.dataSensed for robot in self.env.robot])
                    dataCollected_percentage = robot_data_sensed / self.env.scene.data_total
                    print("dataCollected_percentage", dataCollected_percentage)
                    data_final = list(self.env.scene.signalPointId2data.values())
                    fair = fairness(data_orig, data_final)
                    print("fair", fair)
                    energy_usage = []
                    for robot in self.env.robot:
                        energy_usage.append(robot.consumption_energy / (1 + robot.accumulated_charge_energy))
                    print("energy_usage", sum(energy_usage) / len(energy_usage))
                    for charger in self.env.charger:
                        print("charge_steps_ratio", charger.charge_steps / i_step)
                    # 防止除零：如果所有机器人累积充电能量总和为0，则将归一化结果设为全零，公平性默认设为0
                    accumulated_charge_energy_list = [robot.accumulated_charge_energy for robot in self.env.robot]
                    total_accumulated = sum(accumulated_charge_energy_list)
                    if total_accumulated == 0:
                        nor_accumulated_charge_energy_list = np.zeros(len(accumulated_charge_energy_list))
                        fair_charge = 0.0
                    else:
                        nor_accumulated_charge_energy_list = np.array([en / total_accumulated for en in accumulated_charge_energy_list])
                        sum_sq = sum(nor_accumulated_charge_energy_list ** 2)
                        if sum_sq == 0:
                            fair_charge = 0.0
                        else:
                            fair_charge = (sum(nor_accumulated_charge_energy_list) ** 2) / (len(nor_accumulated_charge_energy_list) * sum_sq)
                    print("fair_charge", fair_charge)
                    break

    def random(self):
        """
        随机策略模式，用于测试环境与模型接口

        流程说明：
          1. 重置环境，获取初始状态与轨迹；
          2. 对于每一步，分别调用 UAV 和 Charger 的贪婪策略选择动作；
          3. 执行环境 step，更新状态与轨迹，直至任一智能体完成任务；
          4. 输出数据采集百分比、能源消耗与公平性指标。
        """
        current_state = self.env.reset()
        data_orig = list(self.env.scene.signalPointId2data.values())
        start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                    [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
        self.trajectory = [start_pos]
        self.last_pos = start_pos
        # 动作张量尺寸根据智能体数量动态确定
        num_agents = self.param_dict["NUM_DRONE"] + self.param_dict["NUM_CHARGER"]
        action_dim = self.param_dict["DIMENSION_ACTION"][0]
        actions = torch.zeros((num_agents, action_dim), device=self.device)
        total_reward = np.zeros(num_agents)
        with torch.no_grad():
            for i_step in range(self.param_dict["MAX_STEPS"]):
                # 获取 UAV 的位置与电量状态
                UAV_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot]
                UAV_energy = [[robot.electricity, robot.charged_energy] for robot in self.env.robot]
                # 对 UAV 执行贪婪策略，要求 UAV 类中已实现 greedy_action() 方法
                for i, robot in enumerate(self.env.robot):
                    actions[i] = robot.greedy_action()
                # 对 Charger 执行贪婪策略，注意 Charger 动作存储在索引从 NUM_DRONE 开始
                for i, robot in enumerate(self.env.charger):
                    actions[i + self.param_dict["NUM_DRONE"]] = robot.greedy_action(UAV_energy, UAV_pos)
                for robot in self.env.robot:
                    p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
                for charger in self.env.charger:
                    p.resetBaseVelocity(charger.robot, linearVelocity=[0., 0., 0.])
                state_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(self.trajectory))
                next_state = state_ if i_step != self.param_dict["MAX_STEPS"] - 1 else None
                if next_state is not None:
                    current_state = next_state
                current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + \
                              [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
                self.last_pos = current_pos
                self.trajectory.append(current_pos)
                if sum(done) > 0:
                    robot_data_sensed = 0.
                    for robot in self.env.robot:
                        robot_data_sensed += robot.dataSensed
                    dataCollected_percentage = robot_data_sensed / self.env.scene.data_total
                    print("dataCollected_percentage", dataCollected_percentage)
                    data_final = list(self.env.scene.signalPointId2data.values())
                    fair = fairness(data_orig, data_final)
                    print("fair", fair)
                    energy_usage = []
                    for robot in self.env.robot:
                        energy_usage.append(robot.consumption_energy / (1 + robot.accumulated_charge_energy))
                    print("energy_usage", sum(energy_usage) / len(energy_usage))
                    for charger in self.env.charger:
                        print("charge_steps_ratio", charger.charge_steps / i_step)
                    accumulated_charge_energy_list = [robot.accumulated_charge_energy for robot in self.env.robot]
                    total_accumulated = sum(accumulated_charge_energy_list)
                    if total_accumulated == 0:
                        nor_accumulated_charge_energy_list = np.zeros(len(accumulated_charge_energy_list))
                        fair_charge = 0.0
                    else:
                        nor_accumulated_charge_energy_list = np.array([en / total_accumulated for en in accumulated_charge_energy_list])
                        sum_sq = sum(nor_accumulated_charge_energy_list ** 2)
                        if sum_sq == 0:
                            fair_charge = 0.0
                        else:
                            fair_charge = (sum(nor_accumulated_charge_energy_list) ** 2) / (len(nor_accumulated_charge_energy_list) * sum_sq)
                    print("fair_charge", fair_charge)
                    break

if __name__ == "__main__":
    """
    主入口

    根据是否有 GPU 可用选择设备，并依据 train_mode（"train"、"test" 或 "random"）执行相应流程。
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 设置训练模式："train"：训练；"test"：评估；"random"：随机策略模式
    train_mode = "test"
    
    if train_mode == "train":
        runner = PybulletRunner(resume_run=False, if_render=False, device=device)
        try:
            runner.run()
        except (Exception, KeyboardInterrupt) as e:
            if isinstance(runner.controller, MADDPGController):
                print('Something happened, saving checkpoint...')
                runner.controller.save_checkpoint(runner.step_num, runner.episode_num)
            if not isinstance(e, KeyboardInterrupt):
                with open(path.join('hcanet-3.27_maddpg/data/', 'log.txt'), 'a') as f:
                    import traceback
                    f.write(str(e))
                    f.write(traceback.format_exc())
                raise e
    elif train_mode == "test":
        runner = PybulletRunner(resume_run=True, if_render=False, device=device)
        try:
            runner.evaluate()
        except (Exception, KeyboardInterrupt) as e:
            raise e
