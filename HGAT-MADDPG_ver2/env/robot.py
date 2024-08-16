from operator import pos
import pybullet as p
import time
from yaml import load, Loader
import torch
from env.utils import *
from env.scene import *

class Drone(object):
    def __init__(self, basePos : list = [0., 0., 0.], sence_loadItems: dict = None, signalPointId2data: dict = None, physicsClientId : int = 0, device:str='cpu', index: int=-1):
        self._physics_client_id = physicsClientId
        self.sence_loadItems = sence_loadItems
        self.signalPointId2data = signalPointId2data
        self.device = device
        for file in os.listdir("HGAT-MADDPG_ver2/config"):
            path = "HGAT-MADDPG_ver2/config/" + file
            param_dict = load(open(path, "r", encoding="utf-8"), Loader=Loader)
            for key, value in param_dict.items():
                setattr(self, key, value)

        self.robot = addSphere(
            pos=basePos,
            radius=self.DRONE_SCALE,
            mass=self.DRONE_WEIGHT,
            rgba=self.DRONE_COLOR
            )

        self.electricity = 1.
        self.charged_energy = 0.
        self.accumulated_charge_energy = 0.
        self.consumption_energy = 0.
        self.status = 0 #0: flying; 1:charging; -1: stop
        self.dataSensed = 0.
        self.dataSensed_current = 0.
        self.reward = 0.
        self.dilemma_flag = False
        # 记录执行新动作前的位置信息
        self.Pos_last, _= p.getBasePositionAndOrientation(self.robot)
        self.index = index

    def apply_action(self, action):
        currentPos, _ = p.getBasePositionAndOrientation(self.robot)
        self.Pos_last = currentPos
        threed_action = torch.cat((action, torch.tensor([0]).to(self.device)), 0)
        p.applyExternalForce(self.robot, -1, threed_action * self.DRONE_FORCE, currentPos, flags=p.WORLD_FRAME)

    def get_observation(self, robot_pos, charger_pos, curr_step, history_trajectory):
        """
        获取无人机当前的观测值
        输入：
            robot_pos: 所有无人机的位置
            charger_pos: 所有充电器的位置
            curr_step: 当前时间步
            history_trajectory: 无人机历史轨迹
        返回：
            observation_all: 包含所有观测值的张量
            observed_robot: 最近无人机和充电器的ID
        """
        # 初始化最近无人机和充电器的ID为-1
        observed_robot = np.ones((2)) * -1
        observed_robot = observed_robot.astype(int)
        
        # 获取当前无人机的位置和方向
        currentPos, _ = p.getBasePositionAndOrientation(self.robot)
        
        # 1. 激光探测障碍物距离，初始化障碍物位置张量
        obstacle_pos = torch.zeros((int(self.LASER_NUM_HORIZONTAL/2)), dtype=torch.float32, device=self.device)
        
        # 进行激光射线测试
        unitRayVecs, froms, tos, results = rayTest(self.robot, self.LASER_LENGTH, 
                                                ray_num_horizontal=int(self.LASER_NUM_HORIZONTAL/2), 
                                                ray_num_vertical=0)
        
        # 处理激光射线的返回结果
        t = 0
        for index, result in enumerate(results):
            if result[0] != -1:
                if index < int(self.LASER_NUM_HORIZONTAL/2):
                    if result[0] in self.sence_loadItems["obstacle"] or self.sence_loadItems["fence"]:
                        obstacle_pos[index] = (caculate_2D_distance(currentPos, result[3]) - self.DRONE_SCALE) / self.LASER_LENGTH
            else:
                if index < int(self.LASER_NUM_HORIZONTAL/2):
                    obstacle_pos[index] = torch.tensor([1.]).to(self.device)

        # 2. 计算与其他无人机和充电器的距离和方向
        near_robot_pos = torch.ones(((self.NUM_DRONE+self.NUM_CHARGER-1) *3), dtype=torch.float32, device=self.device)
        t = 0
        nearest_UAV = -1
        nearest_dis = 10000
        for id, pos in enumerate(robot_pos):
            if id != self.index:
                dis = caculate_2D_distance(pos, currentPos)
                if dis < nearest_dis:
                    nearest_dis = dis
                    nearest_UAV = id
                if dis <= self.LASER_LENGTH:
                    near_robot_pos[t*3:(t+1)*3-1] = torch.from_numpy(direction_normalize(np.subtract(pos[:2], currentPos[:2]))).to(self.device)
                    near_robot_pos[(t+1)*3-1] = torch.tensor([caculate_2D_distance(pos, currentPos) / self.LASER_LENGTH]).to(self.device)
                t += 1
        if nearest_UAV != -1:
            observed_robot[0] = int(nearest_UAV)
        
        # 处理与充电器的距离
        nearest_charger = -1
        nearest_dis = 10000
        for id, pos in enumerate(charger_pos):
            dis = caculate_2D_distance(pos, currentPos)
            if dis < nearest_dis:
                nearest_dis = dis
                nearest_charger = id
            if dis <= self.LASER_LENGTH:
                near_robot_pos[t*3:(t+1)*3-1] = torch.from_numpy(direction_normalize(np.subtract(pos[:2], currentPos[:2]))).to(self.device)
                near_robot_pos[(t+1)*3-1] = torch.tensor([caculate_2D_distance(pos, currentPos) / self.LASER_LENGTH]).to(self.device)
            t += 1
        if nearest_charger != -1:
            observed_robot[1] = int(nearest_charger+self.NUM_DRONE)

        # 3. 获取当前的角速度并归一化
        ang_vel = torch.tensor(velocilty_normalize(p.getBaseVelocity(self.robot)[0])[:2]).to(self.device)

        # 4. 计算到信号点的距离和方向，考虑吸引力
        self.max_attratcion_SP = 0.  # 初始化最大吸引力为0
        self.max_attratcion_SP_pos = None  # 初始化最大吸引力位置为空
        sensed_signalPointdata = np.array([])  # 初始化一个空的数组，用于存储信号点相关信息

        # 遍历所有的信号点及其对应的数据量
        for SP, data in self.signalPointId2data.items():
            # 获取信号点SP的位置
            pos_SP, _ = p.getBasePositionAndOrientation(SP)
            
            # 计算当前无人机位置与信号点之间的二维距离
            diss = caculate_2D_distance(currentPos, pos_SP)
            
            # 判断距离是否在激光雷达的探测范围内
            if diss <= self.LASER_LENGTH:
                # 计算并归一化方向向量，并添加到数据数组中
                sensed_signalPointdata = np.append(sensed_signalPointdata, direction_normalize(np.subtract(currentPos[:2], pos_SP[:2])))
                
                # 归一化距离，并添加到数据数组中
                sensed_signalPointdata = np.append(sensed_signalPointdata, caculate_2D_distance(currentPos, pos_SP) / self.LASER_LENGTH)
                
                # 将信号点的数据量添加到数据数组中
                sensed_signalPointdata = np.append(sensed_signalPointdata, np.array(data))
                
                # 计算信号点的吸引力
                curr_SP_attraction = data * min([caculate_2D_distance(pos, pos_SP) for pos in robot_pos]) / diss
                
                # 如果当前信号点的吸引力大于已记录的最大吸引力，更新最大吸引力及其位置
                if curr_SP_attraction > self.max_attratcion_SP:
                    self.max_attratcion_SP = curr_SP_attraction
                    self.max_attratcion_SP_pos = pos_SP
                
                # 将当前信号点的吸引力添加到数据数组中
                sensed_signalPointdata = np.append(sensed_signalPointdata, curr_SP_attraction)
            

        # 如果探测到的信号点少于最大数量，则进行填充
        if len(sensed_signalPointdata) <= self.NUM_MAX_SENSED_SIGNAL_POINT * 6:
            pad = np.ones(self.NUM_MAX_SENSED_SIGNAL_POINT * 6 - len(sensed_signalPointdata))
            sensed_signalPointdata = np.concatenate((sensed_signalPointdata, pad))
        else:
            sensed_signalPointdata = sensed_signalPointdata[:self.NUM_MAX_SENSED_SIGNAL_POINT * 6]
        sensed_signalPointdata = torch.from_numpy(sensed_signalPointdata).to(self.device)
        
        # 5. 当前时间步归一化
        sstep = torch.tensor([curr_step / self.MAX_STEPS], device=self.device)

        # 6. 工作状态
        status = torch.tensor([self.status]).to(self.device)

        # 7. 检查困境状态
        if curr_step == 0:
            dilemma = torch.tensor([0]).to(self.device)
        else:
            dilemma = torch.tensor([self.detect_dilemma(history_trajectory, currentPos)]).to(self.device)

        # 节点类型
        node_type = torch.tensor([0]).to(self.device)
        
        # 合并所有观测值并返回
        observation_all = torch.cat((obstacle_pos, near_robot_pos, ang_vel, sensed_signalPointdata, sstep, status, dilemma, node_type), dim=0).unsqueeze(0)
        return observation_all, torch.from_numpy(observed_robot).to(self.device)


    def collision_check(self):  
        result_getContactPoints = p.getContactPoints(bodyA=self.robot, linkIndexA=-1, physicsClientId=self._physics_client_id)
        if result_getContactPoints:   
            return True
        else:
            return False

    def movement_state(self, data_collected: float):  
        '''
        sensing: dataSensed增加
        moving: 位置改变
        static: 悬停（x、y方向位置不变）
        charging: 停靠在充电桩上（与充电桩产生接触）
        '''
        currentPos, _ = p.getBasePositionAndOrientation(self.robot)
        if data_collected > 0:
            sensing = True
        else:
            sensing = False
        if currentPos != self.Pos_last:
            moving = True
        else:
            moving = False
        if currentPos[:2] == self.Pos_last[:2]:
            static = True
        else:
            static = False

        result_getContactPoints = p.getContactPoints(bodyA=self.robot, linkIndexA=-1, physicsClientId=self._physics_client_id)
        if result_getContactPoints:
            collision_id = result_getContactPoints[0][2]
            if collision_id in self.sence_loadItems["charger"]:
                charging = True
            else:
                charging = False
        else:
            charging = False
        return sensing, moving, static, charging

    def energy_consumption(
        self, last_pos, current_pos,
        dataCollectedConsumption_perUnit, movingConsumption_perUnit,
        max_data_collected, max_moveDistance
        ):
        """能量消耗函数"""
        dis = caculate_2D_distance(list(last_pos), list(current_pos))
        distance = min(dis, max_moveDistance)
        data_collected = min(self.dataSensed_current, max_data_collected)
        data_collected_units = data_collected / self.UNIT_DATA_COLLECTED_PER_STEP
        energyConsumption_dataCollecting = dataCollectedConsumption_perUnit * data_collected_units
        distance_ratio = distance / max_moveDistance
        energyConsumption_moving = movingConsumption_perUnit * distance_ratio   #假设每单元路移动的能量消耗相等？
        current_energy_consumption = energyConsumption_dataCollecting + energyConsumption_moving
        return [
            current_energy_consumption, energyConsumption_dataCollecting, energyConsumption_moving,
            data_collected_units, distance_ratio
            ]

    def signalPoint_sensed(self):
        """
        感知范围内的PoI
        return 感知范围内的PoI列表
        """
        self.dataSensed_current = 0.
        signalPoint_sensed_list = []
        for SP in self.sence_loadItems["signalPoint"]:
            Pos_SP, _ = p.getBasePositionAndOrientation(SP)
            Pos_robot, _ = p.getBasePositionAndOrientation(self.robot)
            if caculate_2D_distance(PosA=Pos_SP, PosB=Pos_robot) <= self.SENSING_EXTENT: #and (Pos_robot[-1] - Pos_SP[-1]) < self.SENSING_HEIGHT_EXTENT:
                signalPoint_sensed_list.append(SP)
            else:
                pass
        self.signalPoint_sensed_list = signalPoint_sensed_list
        return signalPoint_sensed_list

    def sensing_insection_percentage(self, point_a, point_b):
        """计算感知范围的相交面积"""
        sensing_extent = self.SENSING_EXTENT
        distance = caculate_2D_distance(point_a, point_b)
        if 2 * sensing_extent < distance:
            return 0.
        elif distance == 0:
            return 1.
        else:
            angle = 2 * acos(distance**2 / 2 / sensing_extent / distance)
            return (sensing_extent**2 * angle - sensing_extent**2 * sin(angle)) / (pi * sensing_extent**2)

    def detect_dilemma(self, history_trajectory, curr_pos):
        """
        dilemma检测
        input 路径、位置
        return true/false
        """
        if self.dataSensed_current <= 0.:
            result_sensing_insection_percentage = [
                    self.sensing_insection_percentage(curr_pos, history_trajectory[i])
                    for i in range(len(history_trajectory))
                ]
            max_sensing_insection_percentage = max(result_sensing_insection_percentage)
            if max_sensing_insection_percentage != result_sensing_insection_percentage[-1]:
                self.dilemma_flag = True    #存在t'，使得Ot,t'>Ot,t+1
                return 1
            else:
                self.dilemma_flag = False
                return 0
        else:
            self.dilemma_flag = False
            return 0

class ChargeUAV(object):
    def __init__(self, basePos : list = [0., 0., 0.], sence_loadItems: dict = None,  physicsClientId : int = 0, device:str='cpu', index: int=-1):
        self.sence_loadItems = sence_loadItems
        self._physics_client_id = physicsClientId
        self.device = device

        for file in os.listdir("HGAT-MADDPG_ver2/config"):
            path = "HGAT-MADDPG_ver2/config/" + file
            param_dict = load(open(path, "r", encoding="utf-8"), Loader=Loader)
            for key, value in param_dict.items():
                setattr(self, key, value)

        self.robot = addSphere(
            pos=basePos,
            radius=self.DRONE_SCALE,
            mass=self.DRONE_WEIGHT,
            rgba=self.CHARGER_COLOR
            )
        
        self.status = 0 #0 is not charging; 1 is charging
        self.index = index
        self.charge_steps = 0

    def apply_action(self, action):
        currentPos, _ = p.getBasePositionAndOrientation(self.robot)
        self.Pos_last = currentPos
        threed_action = torch.cat((action, torch.tensor([0]).to(self.device)), 0)
        p.applyExternalForce(self.robot, -1, threed_action * self.DRONE_FORCE, currentPos, flags=p.WORLD_FRAME)

    def get_observation(self, robot_pos, charger_pos, UAV_energy, curr_step, history_trajectory):
        """
        获取当前无人机的观察值
        输入：
            robot_pos: 所有无人机的位置
            charger_pos: 所有充电器的位置
            UAV_energy: 所有无人机的剩余电量和已充电量
            curr_step: 当前时间步
            history_trajectory: 无人机历史轨迹
        返回：
            observation_all: 包含所有观察值的张量
            observed_robot: 最近无人机和充电器的ID
        """
        # 初始化最近无人机和充电器的ID为-1
        observed_robot = np.ones((2)) * -1
        observed_robot = observed_robot.astype(int)
        
        # 获取当前无人机的位置和方向
        currentPos, _ = p.getBasePositionAndOrientation(self.robot)
        
        # 1. 激光雷达扫描障碍物，初始化障碍物位置张量
        obstacle_pos = torch.zeros((self.LASER_NUM_HORIZONTAL), dtype=torch.float32, device=self.device)
        unitRayVecs, froms, tos, results = rayTest(self.robot, self.LASER_LENGTH, 
                                                ray_num_horizontal=self.LASER_NUM_HORIZONTAL, 
                                                ray_num_vertical=0)
        
        # 处理激光射线的返回结果
        for index, result in enumerate(results):
            if result[0] != -1:
                if index < self.LASER_NUM_HORIZONTAL:
                    if result[0] in self.sence_loadItems["obstacle"] or self.sence_loadItems["fence"]:
                        obstacle_pos[index] = (caculate_2D_distance(currentPos, result[3]) - self.DRONE_SCALE) / self.LASER_LENGTH
            else:
                # 无返回探测对象
                if index < self.LASER_NUM_HORIZONTAL:
                    obstacle_pos[index] = torch.tensor([1.]).to(self.device)
        
        # 2. 计算与其他无人机和充电器的距离和方向
        near_robot_pos = torch.ones(((self.NUM_DRONE+self.NUM_CHARGER-1) * 3), dtype=torch.float32, device=self.device)
        t = 0
        nearest_UAV = -1
        nearest_dis = 10000
        for id, pos in enumerate(robot_pos):
            dis = caculate_2D_distance(pos, currentPos)
            if dis < nearest_dis:
                nearest_dis = dis
                nearest_UAV = id
            if dis <= self.LASER_LENGTH:
                near_robot_pos[t*3:(t+1)*3-1] = torch.from_numpy(direction_normalize(np.subtract(pos[:2], currentPos[:2]))).to(self.device)
                near_robot_pos[(t+1)*3-1] = torch.tensor([caculate_2D_distance(pos, currentPos) / self.LASER_LENGTH]).to(self.device)
            t += 1
        if nearest_UAV != -1:
            observed_robot[0] = int(nearest_UAV)
        
        # 计算与充电器的距离和方向
        nearest_charger = -1
        nearest_dis = 10000
        for id, pos in enumerate(charger_pos):
            if id+self.NUM_DRONE != self.index:
                dis = caculate_2D_distance(pos, currentPos)
                if dis < nearest_dis:
                    nearest_dis = dis
                    nearest_charger = id+self.NUM_DRONE
                if dis <= self.LASER_LENGTH:
                    near_robot_pos[t*3:(t+1)*3-1] = torch.from_numpy(direction_normalize(np.subtract(pos[:2], currentPos[:2]))).to(self.device)
                    near_robot_pos[(t+1)*3-1] = torch.tensor([caculate_2D_distance(pos, currentPos) / self.LASER_LENGTH]).to(self.device)
                t += 1
        if nearest_charger != -1:
            observed_robot[1] = int(nearest_charger)
        
        # 3. 获取当前角速度并归一化
        ang_vel = torch.tensor(velocilty_normalize(p.getBaseVelocity(self.robot)[0])[:2]).to(self.device)

        # 4. 获取所有无人机的剩余电量和已充电量
        t = 0
        energy_info = torch.zeros((self.NUM_DRONE * 2), dtype=torch.float32, device=self.device)
        for energy in UAV_energy:
            energy_info[2*t] = energy[0]
            energy_info[2*t+1] = energy[1]
            t += 1
        
        # 5. 当前时间步归一化
        sstep = torch.tensor([curr_step / self.MAX_STEPS], device=self.device)
        
        # 6. 工作状态
        status = torch.tensor([self.status]).to(self.device)

        # 7. 检查困境状态
        if curr_step == 0:
            dilemma = torch.tensor([0]).to(self.device)
        else:
            dilemma = torch.tensor([self.detect_dilemma(history_trajectory, currentPos)]).to(self.device)

        # 8. 设置节点类型
        node_type = torch.tensor([1]).to(self.device)

        # 9. 合并所有观察值并返回
        observation_all = torch.cat((obstacle_pos, near_robot_pos, ang_vel, energy_info, sstep, status, dilemma, node_type), dim=0).unsqueeze(0)
        
        # 如果观察值的维度小于预期，填充0以达到预定维度
        pad = torch.zeros((self.DIMENSION_OBS[0] - self.DIMENSION_OBS[1]), device=self.device)
        observation_all = torch.cat((observation_all, pad.unsqueeze(0)), dim=1)
        
        return observation_all, torch.from_numpy(observed_robot).to(self.device)


    def collision_check(self):  
        """碰撞检测"""

        result_getContactPoints = p.getContactPoints(bodyA=self.robot, linkIndexA=-1, physicsClientId=self._physics_client_id)
        if result_getContactPoints:   
            return True
        else:
            return False
        
    def sensing_insection_percentage(self, point_a, point_b):
        """计算感知范围的相交面积"""
        sensing_extent = self.SENSING_EXTENT
        distance = caculate_2D_distance(point_a, point_b)
        if 2 * sensing_extent < distance:
            return 0.
        elif distance == 0:
            return 1.
        else:
            angle = 2 * acos(distance**2 / 2 / sensing_extent / distance)
            return (sensing_extent**2 * angle - sensing_extent**2 * sin(angle)) / (pi * sensing_extent**2)
        
    def detect_dilemma(self, history_trajectory, curr_pos):
        """
        dilemma检测
        input 路径、位置
        return true/false
        """
        result_sensing_insection_percentage = [
                self.sensing_insection_percentage(curr_pos, history_trajectory[i])
                for i in range(len(history_trajectory))
            ]
        max_sensing_insection_percentage = max(result_sensing_insection_percentage)
        if max_sensing_insection_percentage != result_sensing_insection_percentage[-1]:
            self.dilemma_flag = True    #存在t'，使得Ot,t'>Ot,t+1
            return 1
        else:
            self.dilemma_flag = False
            return 0

# if __name__ == "__main__":
#     cid = p.connect(p.GUI)
#     scene = Scence(physicsClientId=cid)
#     scene.construct()

#     robot = Drone(basePos=[0,0,1], baseOri=p.getQuaternionFromEuler([0., 0., 1.5707963]), sence_loadItems=scene.load_items,signalPointId2data=scene.signalPointId2data)
#     print('1', p.getCollisionShapeData(robot.robot, -1))
#     print('2', p.getBasePositionAndOrientation(robot.robot)[0])
#     while (1):
#         p.stepSimulation()
#         time.sleep(1. / 240.)
#         _,_ = robot.get_observation()
#         print('obs', obs)
#         print(edge)