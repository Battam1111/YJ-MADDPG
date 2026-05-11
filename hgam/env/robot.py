#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无人机与充电无人机（Drone 与 ChargeUAV）模块
================================================

该文件定义了两个类：
1. Drone：普通无人机，负责数据采集、感知环境、执行动作等；
2. ChargeUAV：移动充电站，负责为无人机充电，同时提供部分辅助信息。

本文件中对观测构造函数进行了统一处理，确保返回的观测向量尺寸与配置文件中定义的全局观测维度（DIMENSION_OBS[0]）一致，
即通过动态计算实际输出的尺寸，然后进行零填充（或截断）来达到要求，彻底消除因硬编码 pad 而导致的尺寸不匹配问题。

注意：所有配置参数在构造时均通过读取配置文件自动赋值，因此类内部可以直接使用 self.DIMENSION_OBS 等参数。
  
作者：YourName  
日期：202X-XX-XX
"""

import os
import pybullet as p
import time
from yaml import safe_load
import torch
from hgam.env.utils import *       # 包含各种工具函数，如 addSphere, caculate_2D_distance, direction_normalize, velocilty_normalize 等
from hgam.env.scene import *       # 场景构造模块，负责加载环境中各类物体


# ----------------------------------------------------------------------
# Phase-D performance fix: cache the YAML config dict at module load time
# rather than re-reading every YAML file on every Drone / ChargeUAV
# construction. Profiling showed yaml.load was responsible for ~11% of
# training step time because env.reset() instantiates a new Drone +
# ChargeUAV per drone, each previously re-parsing every yaml in
# configs/_legacy/.
# ----------------------------------------------------------------------

_CONFIG_DIR = "configs/_legacy"
_CONFIG_CACHE = None


def _load_robot_config():
    """Load the legacy-dir YAMLs exactly once per process.

    The returned dict is shared by reference across all Drone / ChargeUAV
    instances. Callers must not mutate it. If you need a per-instance
    override, deepcopy and mutate the copy; do not modify _CONFIG_CACHE.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        cfg = {}
        for fname in sorted(os.listdir(_CONFIG_DIR)):
            if not fname.endswith((".yaml", ".yml")):
                continue
            with open(os.path.join(_CONFIG_DIR, fname), "r", encoding="utf-8") as f:
                cfg.update(safe_load(f) or {})
        _CONFIG_CACHE = cfg
    return _CONFIG_CACHE


class Drone(object):
    """
    无人机（Drone）类，用于实现普通无人机的物理建模、动作执行、观测提取以及能量管理等功能
    """
    def __init__(self, basePos: list = [0., 0., 0.], sence_loadItems: dict = None, signalPointId2data: dict = None, 
                 physicsClientId: int = 0, device: str='cpu', index: int=-1):
        """
        初始化Drone类
        
        参数:
            basePos: 初始位置（列表形式，例如 [x, y, z]）
            sence_loadItems: 环境中载入的物体（字典），如障碍物、围墙等
            signalPointId2data: 信号点对应的数据字典
            physicsClientId: pybullet 物理客户端ID
            device: 设备（'cpu' 或 'cuda'）
            index: 无人机的索引编号（用于区分各个智能体）
        """
        self._physics_client_id = physicsClientId
        self.sence_loadItems = sence_loadItems
        self.signalPointId2data = signalPointId2data
        self.device = device
        # 从配置文件加载参数 (cached at module load — see _load_robot_config above).
        # We attach via __dict__.update for speed; setattr in a Python loop was
        # ~12ms each call and ran 176 times per 30 env-steps before this fix.
        self.__dict__.update(_load_robot_config())
        # 创建无人机实体（使用球体表示）
        self.robot = addSphere(
            pos=basePos,
            radius=self.DRONE_SCALE,
            mass=self.DRONE_WEIGHT,
            rgba=self.DRONE_COLOR
        )
        # 初始化能量、数据采集等状态
        self.electricity = 1.
        self.charged_energy = 0.
        self.accumulated_charge_energy = 0.
        self.consumption_energy = 0.
        self.status = 0         # 0: 正常飞行; 1: 正在充电; -1: 状态异常/停止
        self.dataSensed = 0.
        self.dataSensed_current = 0.
        self.reward = 0.
        self.dilemma_flag = False
        # 记录上一次动作执行前的位置，用于计算移动距离等
        self.Pos_last, _ = p.getBasePositionAndOrientation(self.robot)
        self.index = index

    def apply_action(self, action):
        """
        根据输入的动作向量为无人机施加外部力
        
        参数:
            action: 动作张量（通常为二维向量）
        """
        currentPos, _ = p.getBasePositionAndOrientation(self.robot)
        self.Pos_last = currentPos
        # 将二维动作扩展为三维（补0作为第三个分量）
        threed_action = torch.cat((action, torch.tensor([0]).to(self.device)), 0)
        # 应用外部力，注意这里使用 WORLD_FRAME 坐标系
        p.applyExternalForce(self.robot, -1, threed_action * self.DRONE_FORCE, currentPos, flags=p.WORLD_FRAME)

    def get_observation(self, robot_pos, charger_pos, curr_step, history_trajectory):
        """
        获取当前无人机的观测值
        
        输入：
            robot_pos: 所有无人机的位置列表
            charger_pos: 所有充电器的位置列表
            curr_step: 当前时间步（整数）
            history_trajectory: 无人机历史轨迹（可为None）
        
        返回：
            observation_all: 形状为 (1, DIMENSION_OBS[0]) 的观测张量，
                             必须与配置中 DIMENSION_OBS[0] 保持一致（例如630维）
            observed_robot: 最近邻（无人机和充电器）的索引（张量）
        """
        # 初始化最近邻的索引为 -1（未发现）
        observed_robot = np.ones((2)) * -1
        observed_robot = observed_robot.astype(int)
        
        # 获取当前无人机的位置信息和方向
        currentPos, _ = p.getBasePositionAndOrientation(self.robot)
        
        # 1. 利用激光雷达获取障碍物距离
        # 此处使用水平激光数量的一半，得到的张量维度为 int(LASER_NUM_HORIZONTAL/2)
        obstacle_pos = torch.zeros((int(self.LASER_NUM_HORIZONTAL/2)), dtype=torch.float32, device=self.device)
        unitRayVecs, froms, tos, results = rayTest(self.robot, self.LASER_LENGTH, 
                                                    ray_num_horizontal=int(self.LASER_NUM_HORIZONTAL/2), 
                                                    ray_num_vertical=0)
        # 根据检测结果填充距离信息
        for index, result in enumerate(results):
            if result[0] != -1:
                if index < int(self.LASER_NUM_HORIZONTAL/2):
                    # 如果检测到障碍物且障碍物属于环境中障碍物或围墙，则计算距离归一化值
                    if result[0] in self.sence_loadItems["obstacle"] or result[0] in self.sence_loadItems["fence"]:
                        obstacle_pos[index] = (caculate_2D_distance(currentPos, result[3]) - self.DRONE_SCALE) / self.LASER_LENGTH
            else:
                if index < int(self.LASER_NUM_HORIZONTAL/2):
                    obstacle_pos[index] = torch.tensor([1.]).to(self.device)  # 未检测到障碍物，归一化距离取1

        # 2. 计算与其他无人机和充电器的相对信息
        # 初始化邻居信息向量，维度为 ((NUM_DRONE+NUM_CHARGER-1)*3)
        near_robot_pos = torch.ones(((self.NUM_DRONE + self.NUM_CHARGER - 1) * 3), dtype=torch.float32, device=self.device)
        t = 0
        nearest_UAV = -1
        nearest_dis = 10000
        # 遍历其他无人机的位置，计算相对方向和归一化距离
        for id, pos in enumerate(robot_pos):
            if id != self.index:
                dis = caculate_2D_distance(pos, currentPos)
                if dis < nearest_dis:
                    nearest_dis = dis
                    nearest_UAV = id
                if dis <= self.LASER_LENGTH:
                    near_robot_pos[t*3:(t+1)*3-1] = torch.from_numpy(direction_normalize(np.subtract(pos[:2], currentPos[:2]))).to(self.device)
                    near_robot_pos[(t+1)*3-1] = torch.tensor([dis / self.LASER_LENGTH]).to(self.device)
                t += 1
        if nearest_UAV != -1:
            observed_robot[0] = int(nearest_UAV)
        
        # 处理充电器信息
        nearest_charger = -1
        nearest_dis = 10000
        for id, pos in enumerate(charger_pos):
            dis = caculate_2D_distance(pos, currentPos)
            if dis < nearest_dis:
                nearest_dis = dis
                nearest_charger = id
            if dis <= self.LASER_LENGTH:
                near_robot_pos[t*3:(t+1)*3-1] = torch.from_numpy(direction_normalize(np.subtract(pos[:2], currentPos[:2]))).to(self.device)
                near_robot_pos[(t+1)*3-1] = torch.tensor([dis / self.LASER_LENGTH]).to(self.device)
            t += 1
        if nearest_charger != -1:
            # 充电器在列表中的索引需要加上 NUM_DRONE 以区分
            observed_robot[1] = int(nearest_charger + self.NUM_DRONE)

        # 3. 获取当前角速度，并对二维部分进行归一化处理
        ang_vel = torch.tensor(velocilty_normalize(p.getBaseVelocity(self.robot)[0])[:2]).to(self.device)

        # 4. 计算无人机与信号点之间的信息
        # 该部分遍历所有信号点，计算方向、距离、数据量以及吸引力
        self.max_attratcion_SP = 0.  # 初始化最大吸引力
        self.max_attratcion_SP_pos = None  # 初始化最大吸引力位置
        sensed_signalPointdata = np.array([])  # 初始化空数组存放信号点特征
        
        for SP, data in self.signalPointId2data.items():
            # 获取信号点的位置
            pos_SP, _ = p.getBasePositionAndOrientation(SP)
            # 计算无人机与信号点之间的二维距离
            diss = caculate_2D_distance(currentPos, pos_SP)
            if diss <= self.LASER_LENGTH:
                # 计算并归一化方向向量（从信号点指向无人机）
                sensed_signalPointdata = np.append(sensed_signalPointdata, direction_normalize(np.subtract(currentPos[:2], pos_SP[:2])))
                # 归一化距离值
                sensed_signalPointdata = np.append(sensed_signalPointdata, diss / self.LASER_LENGTH)
                # 添加信号点数据量
                sensed_signalPointdata = np.append(sensed_signalPointdata, np.array(data))
                # 计算信号点吸引力（数据量乘以距离比例，激励靠近数据较多的信号点）
                curr_SP_attraction = data * min([caculate_2D_distance(pos, pos_SP) for pos in robot_pos]) / diss
                if curr_SP_attraction > self.max_attratcion_SP:
                    self.max_attratcion_SP = curr_SP_attraction
                    self.max_attratcion_SP_pos = pos_SP
                # 将当前信号点吸引力加入特征
                sensed_signalPointdata = np.append(sensed_signalPointdata, curr_SP_attraction)
        
        # 如果信号点特征数量不足，则填充1（默认值），保证固定长度
        if len(sensed_signalPointdata) < self.NUM_MAX_SENSED_SIGNAL_POINT * 6:
            pad = np.ones(self.NUM_MAX_SENSED_SIGNAL_POINT * 6 - len(sensed_signalPointdata))
            sensed_signalPointdata = np.concatenate((sensed_signalPointdata, pad))
        else:
            sensed_signalPointdata = sensed_signalPointdata[:self.NUM_MAX_SENSED_SIGNAL_POINT * 6]
        # 转换为 tensor
        sensed_signalPointdata = torch.from_numpy(sensed_signalPointdata).to(self.device)
        
        # 5. 当前时间步归一化（将当前步除以最大步数）
        sstep = torch.tensor([curr_step / self.MAX_STEPS], device=self.device)
        # 6. 工作状态（无人机状态）
        status = torch.tensor([self.status]).to(self.device)
        # 7. 检查“困境”状态（根据历史轨迹判断）
        if curr_step == 0 or history_trajectory is None:
            dilemma = torch.tensor([0]).to(self.device)
        else:
            dilemma = torch.tensor([self.detect_dilemma(history_trajectory, currentPos)]).to(self.device)
        # 8. 节点类型：0 表示无人机
        node_type = torch.tensor([0]).to(self.device)
        
        # 9. 拼接所有观测特征（顺序为：激光信息、邻居信息、角速度、信号点信息、时间步、状态、困境、节点类型）
        observation_all = torch.cat((obstacle_pos, near_robot_pos, ang_vel, sensed_signalPointdata, sstep, status, dilemma, node_type), dim=0).unsqueeze(0)
        
        # *** 动态统一输出观测向量的维度 ***
        # 目标：最终输出的观测向量维度必须严格等于配置文件中定义的 DIMENSION_OBS[0]
        expected_dim = self.DIMENSION_OBS[0]
        current_dim = observation_all.shape[1]
        if current_dim < expected_dim:
            # 如果当前维度不足，则补充零
            pad = torch.zeros((1, expected_dim - current_dim), device=self.device)
            observation_all = torch.cat((observation_all, pad), dim=1)
        elif current_dim > expected_dim:
            # 如果当前维度超出，则截断（保留前 expected_dim 个特征）
            observation_all = observation_all[:, :expected_dim]
        
        # 返回观测张量和最近邻的索引（转换为tensor）
        return observation_all, torch.from_numpy(observed_robot).to(self.device)

    def collision_check(self):
        """
        检查无人机是否发生碰撞
        
        返回:
            True：发生碰撞
            False：未发生碰撞
        """
        result_getContactPoints = p.getContactPoints(bodyA=self.robot, linkIndexA=-1, physicsClientId=self._physics_client_id)
        if result_getContactPoints:
            return True
        else:
            return False

    def movement_state(self, data_collected: float):
        """
        根据无人机数据采集和位置变化判断当前状态：
        sensing：正在采集数据
        moving：位置发生变化
        static：悬停状态（x、y方向位置不变）
        charging：与充电桩接触（开始充电）
        
        参数:
            data_collected: 当前采集的数据量
        
        返回:
            四个布尔值，分别表示上述状态
        """
        currentPos, _ = p.getBasePositionAndOrientation(self.robot)
        sensing = data_collected > 0
        moving = currentPos != self.Pos_last
        static = (currentPos[:2] == self.Pos_last[:2])
        result_getContactPoints = p.getContactPoints(bodyA=self.robot, linkIndexA=-1, physicsClientId=self._physics_client_id)
        if result_getContactPoints:
            collision_id = result_getContactPoints[0][2]
            # 如果碰撞对象为充电桩，则视为 charging
            if collision_id in self.sence_loadItems["charger"]:
                charging = True
            else:
                charging = False
        else:
            charging = False
        return sensing, moving, static, charging

    def energy_consumption(self, last_pos, current_pos,
                           dataCollectedConsumption_perUnit, movingConsumption_perUnit,
                           max_data_collected, max_moveDistance):
        """
        计算能量消耗量
        
        参数:
            last_pos: 上一时刻位置
            current_pos: 当前时刻位置
            dataCollectedConsumption_perUnit: 单位数据采集消耗能量
            movingConsumption_perUnit: 单位移动消耗能量
            max_data_collected: 单步最大数据采集量（归一化用）
            max_moveDistance: 单步最大移动距离（归一化用）
        
        返回:
            列表，包含：
              [总能量消耗, 数据采集消耗, 移动消耗, 数据采集归一化比例, 移动距离归一化比例]
        """
        dis = caculate_2D_distance(list(last_pos), list(current_pos))
        distance = min(dis, max_moveDistance)
        data_collected = min(self.dataSensed_current, max_data_collected)
        data_collected_units = data_collected / self.UNIT_DATA_COLLECTED_PER_STEP
        energyConsumption_dataCollecting = dataCollectedConsumption_perUnit * data_collected_units
        distance_ratio = distance / max_moveDistance
        energyConsumption_moving = movingConsumption_perUnit * distance_ratio
        current_energy_consumption = energyConsumption_dataCollecting + energyConsumption_moving
        return [
            current_energy_consumption, energyConsumption_dataCollecting, energyConsumption_moving,
            data_collected_units, distance_ratio
        ]

    def signalPoint_sensed(self):
        """
        判断当前无人机能感知到哪些信号点
        
        返回:
            感知到的信号点列表
        """
        self.dataSensed_current = 0.
        signalPoint_sensed_list = []
        for SP in self.sence_loadItems["signalPoint"]:
            Pos_SP, _ = p.getBasePositionAndOrientation(SP)
            Pos_robot, _ = p.getBasePositionAndOrientation(self.robot)
            if caculate_2D_distance(PosA=Pos_SP, PosB=Pos_robot) <= self.SENSING_EXTENT:
                signalPoint_sensed_list.append(SP)
        self.signalPoint_sensed_list = signalPoint_sensed_list
        return signalPoint_sensed_list

    def sensing_insection_percentage(self, point_a, point_b):
        """
        计算两个点之间的感知范围相交比例
        
        参数:
            point_a: 第一个点（列表）
            point_b: 第二个点（列表）
        返回:
            相交面积比例（浮点数）
        """
        sensing_extent = self.SENSING_EXTENT
        distance = caculate_2D_distance(point_a, point_b)
        if 2 * sensing_extent < distance:
            return 0.
        elif distance == 0:
            return 1.
        else:
            angle = 2 * acos(distance**2 / (2 * sensing_extent * distance))
            return (sensing_extent**2 * angle - sensing_extent**2 * sin(angle)) / (pi * sensing_extent**2)

    def detect_dilemma(self, history_trajectory, curr_pos):
        """
        检测无人机是否处于“困境”状态，即连续若干步未采集到数据且观测信息未更新
        
        参数:
            history_trajectory: 历史轨迹（列表或数组）
            curr_pos: 当前无人机位置
        返回:
            1 表示处于困境，0 表示正常
        """
        result_sensing_insection_percentage = [
            self.sensing_insection_percentage(curr_pos, history_trajectory[i])
            for i in range(len(history_trajectory))
        ]
        max_sensing_insection_percentage = max(result_sensing_insection_percentage)
        if max_sensing_insection_percentage != result_sensing_insection_percentage[-1]:
            self.dilemma_flag = True
            return 1
        else:
            self.dilemma_flag = False
            return 0

class ChargeUAV(object):
    """
    充电无人机（ChargeUAV）类，用于实现移动充电站的建模、动作执行、观测获取及充电逻辑
    """
    def __init__(self, basePos: list = [0., 0., 0.], sence_loadItems: dict = None, 
                 physicsClientId: int = 0, device: str='cpu', index: int=-1):
        """
        初始化充电无人机
        
        参数:
            basePos: 初始位置
            sence_loadItems: 环境中载入的物体（字典）
            physicsClientId: pybullet 客户端ID
            device: 设备（'cpu' 或 'cuda'）
            index: 充电无人机在整体智能体中的索引（通常在 NUM_DRONE 后开始编号）
        """
        self.sence_loadItems = sence_loadItems
        self._physics_client_id = physicsClientId
        self.device = device
        # 从配置文件加载参数 (cached at module load — see _load_robot_config above).
        self.__dict__.update(_load_robot_config())
        # 创建充电无人机实体（使用球体，颜色采用充电桩颜色）
        self.robot = addSphere(
            pos=basePos,
            radius=self.DRONE_SCALE,
            mass=self.DRONE_WEIGHT,
            rgba=self.CHARGER_COLOR
        )
        self.status = 0       # 充电状态：0 表示未充电，1 表示正在充电
        self.index = index
        self.charge_steps = 0

    def sensing_insection_percentage(self, point_a, point_b):
        """
        计算两个点之间的感知范围相交比例
        
        参数:
            point_a: 第一个点（列表）
            point_b: 第二个点（列表）
        返回:
            相交面积比例（浮点数）
        """
        sensing_extent = self.SENSING_EXTENT
        distance = caculate_2D_distance(point_a, point_b)
        if 2 * sensing_extent < distance:
            return 0.
        elif distance == 0:
            return 1.
        else:
            angle = 2 * acos(distance**2 / (2 * sensing_extent * distance))
            return (sensing_extent**2 * angle - sensing_extent**2 * sin(angle)) / (pi * sensing_extent**2)

    def detect_dilemma(self, history_trajectory, curr_pos):
        """
        检测无人机是否处于“困境”状态，即连续若干步未采集到数据且观测信息未更新
        
        参数:
            history_trajectory: 历史轨迹（列表或数组）
            curr_pos: 当前无人机位置
        返回:
            1 表示处于困境，0 表示正常
        """
        result_sensing_insection_percentage = [
            self.sensing_insection_percentage(curr_pos, history_trajectory[i])
            for i in range(len(history_trajectory))
        ]
        max_sensing_insection_percentage = max(result_sensing_insection_percentage)
        if max_sensing_insection_percentage != result_sensing_insection_percentage[-1]:
            self.dilemma_flag = True
            return 1
        else:
            self.dilemma_flag = False
            return 0

    def apply_action(self, action):
        """
        根据输入动作为充电无人机施加外部力
        
        参数:
            action: 动作张量（通常为二维向量）
        """
        currentPos, _ = p.getBasePositionAndOrientation(self.robot)
        self.Pos_last = currentPos
        threed_action = torch.cat((action, torch.tensor([0]).to(self.device)), 0)
        p.applyExternalForce(self.robot, -1, threed_action * self.DRONE_FORCE, currentPos, flags=p.WORLD_FRAME)

    def get_observation(self, robot_pos, charger_pos, UAV_energy, curr_step, history_trajectory):
        """
        获取当前充电无人机的观测值
        
        输入：
            robot_pos: 所有无人机的位置列表
            charger_pos: 所有充电器的位置列表
            UAV_energy: 所有无人机的电量与已充电量信息（列表，每个元素为 [electricity, charged_energy]）
            curr_step: 当前时间步
            history_trajectory: 历史轨迹（可为 None）
        
        返回：
            observation_all: 形状为 (1, DIMENSION_OBS[0]) 的观测张量，尺寸严格等于配置中的 DIMENSION_OBS[0]
            observed_robot: 最近邻的无人机/充电器索引（tensor）
        """
        # 初始化最近邻索引为 -1
        observed_robot = np.ones((2)) * -1
        observed_robot = observed_robot.astype(int)
        
        # 获取当前充电无人机的位置
        currentPos, _ = p.getBasePositionAndOrientation(self.robot)
        
        # 1. 利用激光雷达扫描障碍物，得到障碍物距离信息
        # 这里使用全部水平激光数量，得到维度为 LASER_NUM_HORIZONTAL
        obstacle_pos = torch.zeros((self.LASER_NUM_HORIZONTAL), dtype=torch.float32, device=self.device)
        unitRayVecs, froms, tos, results = rayTest(self.robot, self.LASER_LENGTH, 
                                                    ray_num_horizontal=self.LASER_NUM_HORIZONTAL, 
                                                    ray_num_vertical=0)
        for index, result in enumerate(results):
            if result[0] != -1:
                if index < self.LASER_NUM_HORIZONTAL:
                    if result[0] in self.sence_loadItems["obstacle"] or result[0] in self.sence_loadItems["fence"]:
                        obstacle_pos[index] = (caculate_2D_distance(currentPos, result[3]) - self.DRONE_SCALE) / self.LASER_LENGTH
            else:
                if index < self.LASER_NUM_HORIZONTAL:
                    obstacle_pos[index] = torch.tensor([1.]).to(self.device)
        
        # 2. 计算与其他无人机和充电器的相对信息
        # 初始化邻居信息向量，维度为 ((NUM_DRONE+NUM_CHARGER-1)*3)
        near_robot_pos = torch.ones(((self.NUM_DRONE + self.NUM_CHARGER - 1) * 3), dtype=torch.float32, device=self.device)
        t = 0
        nearest_UAV = -1
        nearest_dis = 10000
        # 遍历所有无人机，计算相对信息
        for id, pos in enumerate(robot_pos):
            dis = caculate_2D_distance(pos, currentPos)
            if dis < nearest_dis:
                nearest_dis = dis
                nearest_UAV = id
            if dis <= self.LASER_LENGTH:
                near_robot_pos[t*3:(t+1)*3-1] = torch.from_numpy(direction_normalize(np.subtract(pos[:2], currentPos[:2]))).to(self.device)
                near_robot_pos[(t+1)*3-1] = torch.tensor([dis / self.LASER_LENGTH]).to(self.device)
            t += 1
        if nearest_UAV != -1:
            observed_robot[0] = int(nearest_UAV)
        
        # 处理充电器信息
        nearest_charger = -1
        nearest_dis = 10000
        for id, pos in enumerate(charger_pos):
            # 注意此处充电器编号需排除本机（利用索引偏移判断）
            if id + self.NUM_DRONE != self.index:
                dis = caculate_2D_distance(pos, currentPos)
                if dis < nearest_dis:
                    nearest_dis = dis
                    nearest_charger = id + self.NUM_DRONE
                if dis <= self.LASER_LENGTH:
                    near_robot_pos[t*3:(t+1)*3-1] = torch.from_numpy(direction_normalize(np.subtract(pos[:2], currentPos[:2]))).to(self.device)
                    near_robot_pos[(t+1)*3-1] = torch.tensor([dis / self.LASER_LENGTH]).to(self.device)
                t += 1
        if nearest_charger != -1:
            observed_robot[1] = int(nearest_charger)
        
        # 3. 获取当前充电无人机的角速度（取二维部分并归一化）
        ang_vel = torch.tensor(velocilty_normalize(p.getBaseVelocity(self.robot)[0])[:2]).to(self.device)
        
        # 4. 获取所有无人机的电量信息（每架无人机提供2个特征：剩余电量和已充能量），并拼接成一个向量
        t = 0
        energy_info = torch.zeros((self.NUM_DRONE * 2), dtype=torch.float32, device=self.device)
        for energy in UAV_energy:
            energy_info[2*t] = energy[0]
            energy_info[2*t+1] = energy[1]
            t += 1
        
        # 5. 当前时间步归一化
        sstep = torch.tensor([curr_step / self.MAX_STEPS], device=self.device)
        # 6. 当前充电无人机状态
        status = torch.tensor([self.status]).to(self.device)
        # 7. 判断困境状态（如果历史轨迹存在则计算，否则默认0）
        if curr_step == 0 or history_trajectory is None:
            dilemma = torch.tensor([0]).to(self.device)
        else:
            dilemma = torch.tensor([self.detect_dilemma(history_trajectory, currentPos)]).to(self.device)
        # 8. 节点类型：1 表示充电无人机
        node_type = torch.tensor([1]).to(self.device)
        
        # 9. 拼接充电无人机各部分观测信息：激光信息、邻居信息、角速度、电量信息、时间步、状态、困境、节点类型
        observation_all = torch.cat((obstacle_pos, near_robot_pos, ang_vel, energy_info, sstep, status, dilemma, node_type), dim=0).unsqueeze(0)
        
        # *** 动态统一输出观测向量的维度 ***
        expected_dim = self.DIMENSION_OBS[0]
        current_dim = observation_all.shape[1]
        if current_dim < expected_dim:
            pad = torch.zeros((1, expected_dim - current_dim), device=self.device)
            observation_all = torch.cat((observation_all, pad), dim=1)
        elif current_dim > expected_dim:
            observation_all = observation_all[:, :expected_dim]
        
        # 返回充电无人机观测和最近邻信息（转换为tensor）
        return observation_all, torch.from_numpy(observed_robot).to(self.device)

    def collision_check(self):
        """
        检查充电无人机是否发生碰撞
        
        返回:
            True: 发生碰撞
            False: 未发生碰撞
        """
        result_getContactPoints = p.getContactPoints(bodyA=self.robot, linkIndexA=-1, physicsClientId=self._physics_client_id)
        if result_getContactPoints:
            return True
        else:
            return False

# 如果需要在该模块独立测试，可以取消以下代码注释
if __name__ == "__main__":
    # 示例：初始化 PyBullet 客户端，构造场景，创建一个 Drone 对象，并不断获取其观测
    import pybullet_data
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # 此处假设已构造场景和加载配置，此处仅为简单示例
    # 请确保配置文件 configs/_legacy 中有正确的参数设置
    # 构造虚拟的sence_loadItems与signalPointId2data，后续可替换为真实场景数据
    sence_loadItems = {"obstacle": [], "fence": []}
    signalPointId2data = {}
    # 创建一个 Drone 对象，初始位置为 [0, 0, 2]
    drone = Drone(basePos=[0, 0, 2], sence_loadItems=sence_loadItems, signalPointId2data=signalPointId2data, physicsClientId=0, device='cpu', index=0)
    # 模拟获取观测
    dummy_robot_pos = [[0, 0, 2]]
    dummy_charger_pos = [[1, 1, 1]]
    obs, neighbor = drone.get_observation(dummy_robot_pos, dummy_charger_pos, 0, None)
    print("无人机观测维度：", obs.shape)
    print("最近邻索引：", neighbor)
