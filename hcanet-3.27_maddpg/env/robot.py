from operator import pos
import pybullet as p
import time
from yaml import load, Loader
import torch
from env.utils import *
from env.scene import *
# from utils import *
# from scene import *
class Drone(object):
    def __init__(self, basePos : list = [0., 0., 0.], sence_loadItems: dict = None, signalPointId2data: dict = None, physicsClientId : int = 0, device:str='cpu', index: int=-1):
        self._physics_client_id = physicsClientId
        self.sence_loadItems = sence_loadItems
        self.signalPointId2data = signalPointId2data
        self.device = device
        for file in os.listdir("hcanet-3.27_maddpg/config"):
            path = "hcanet-3.27_maddpg/config/" + file
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
        激光射线：
            水平（与障碍物距离）
        其他UAV的距离和方向
        当前角速度
        到PoI的距离 + 方向 + 剩余数据 + 吸引力
        当前sstep
        dilemma
        """
        observed_robot = np.ones((2)) * -1
        observed_robot = observed_robot.astype(int)
        currentPos, _ = p.getBasePositionAndOrientation(self.robot)
        # 1. 激光探测障碍物距离
        # 水平方向 + 竖直方向
        obstacle_pos = torch.zeros((int(self.LASER_NUM_HORIZONTAL/2)), dtype=torch.float32, device=self.device)
        unitRayVecs, froms, tos, results = rayTest(self.robot, self.LASER_LENGTH, ray_num_horizontal=int(self.LASER_NUM_HORIZONTAL/2), ray_num_vertical=0)
        # results[0][0] == -1: 激光射线返回异常（坠机/越出限制高度）
        t=0
        for index, result in enumerate(results):
            if result[0] != -1:
                # Pos_hitten, _ = p.getBasePositionAndOrientation(result[0])
                # if index < self.LASER_NUM_HORIZONTAL or result[0] != 0:
                if index < int(self.LASER_NUM_HORIZONTAL/2):
                    # 水平
                    # print("HORIZONTAL lasers")
                    if result[0] in self.sence_loadItems["obstacle"] or self.sence_loadItems["fence"]:
                        # print(p.getBasePositionAndOrientation(result[0])[0][:2])
                        # print(result[3][:2])
                        obstacle_pos[index] = (caculate_2D_distance(currentPos, result[3]) - self.DRONE_SCALE) / self.LASER_LENGTH
                    # elif result[0] in robot_id:
                    #     obstacle_pos[index*2:index*2+2] = torch.tensor(p.getBasePositionAndOrientation(result[0])[0][:2]) +self.INTERNAL_LENGTH / 2
                    #     observed_robot[t] = result[0]
                    #     t += 1
                    #     observation_all = np.concatenate(
                    #         (observation_all, 
                    #         np.array([(caculate_distance(currentPos, result[3]) - self.DRONE_SCALE) / self.LASER_LENGTH, 
                    #         0]
                    #         )), 
                    #         axis=0)# distance
                        # print(np.array([caculate_distance(currentPos, result[3]) - self.DRONE_SCALE, 0]))
                # else:
                #     # 垂直
                #     # print("VERTICAL lasers")
                #     distance = caculate_distance(currentPos, result[3]) - self.DRONE_SCALE
                #     distance *= ((currentPos[2] - self.MIN_HEIGHT_RESTRICTION) / currentPos[2])
                #     # print(np.array([distance]))
                #     observation_all = np.concatenate(
                #         (
                #             observation_all, 
                #             np.array([distance / self.LASER_LENGTH]
                #             )), 
                #             axis=0)# distance
            else:
                # 无返回探测对象
                if index < int(self.LASER_NUM_HORIZONTAL/2):
                    # print("HORIZONTAL lasers")
                    obstacle_pos[index] = torch.tensor([1.]).to(self.device)
            #     elif index >= (self.LASER_NUM_HORIZONTAL + self.LASER_NUM_VERTICAL * 2):
            #         # alpha = (np.pi / 2 / self.LASER_NUM_VERTICAL) * index
            #         alpha = np.linspace(0, np.pi / 2, 10)[index % self.LASER_NUM_VERTICAL]
            #         # print(sin(alpha))
            #         if sin(alpha) == 0:
            #             observation_all = np.concatenate((
            #                 observation_all, 
            #                 # np.array([self.LASER_LENGTH])), 
            #                 np.array([1.])), 
            #                 axis=0)# distance
            #         else:
            #             observation_all = np.concatenate((
            #                 observation_all, 
            #                 np.array([
            #                     # min((self.MAX_HEIGHT_RESTRICTION - currentPos[2]) / sin(alpha), self.LASER_LENGTH)
            #                     min((self.MAX_HEIGHT_RESTRICTION - currentPos[2]) / sin(alpha) / self.LASER_LENGTH, 1.)
            #                     ])), 
            #                 axis=0)# distance
            #     else:
            #         # print("VERTICAL lasers")
            #         observation_all = np.concatenate((
            #             observation_all, 
            #             # np.array([self.LASER_LENGTH])), 
            #             np.array([1.])), 
            #             axis=0)# distance
        # 显示rays
        # p.removeAllUserDebugItems()
        # rayDebugLineIds = []
        # for index, result in enumerate(results):
        #     color = self.MISS_COLOR if result[0] == -1 else self.HIT_COLOR
        #     rayDebugLineIds.append(p.addUserDebugLine(froms[index], tos[index], color))

        # 2. 距离限制高度
        # print("current position:", currentPos)
        # height_restriction = self.MAX_HEIGHT_RESTRICTION - self.MIN_HEIGHT_RESTRICTION
        # observation_all = np.append(observation_all, np.array([(self.MAX_HEIGHT_RESTRICTION - currentPos[2]) / height_restriction]))
        # observation_all = np.append(observation_all, np.array([(currentPos[2] - self.MIN_HEIGHT_RESTRICTION) / height_restriction]))
       
        # 其他robot
        near_robot_pos = torch.ones(((self.NUM_DRONE+self.NUM_CHARGER-1) *3), dtype = torch.float32, device=self.device)
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

        # 3. 角速度
        # observation_all = np.append(observation_all, np.array(p.getBaseVelocity(self.robot)[0]))
        # observation_all = np.append(observation_all, velocilty_normalize(p.getBaseVelocity(self.robot)[0]))
        ang_vel = torch.tensor(velocilty_normalize(p.getBaseVelocity(self.robot)[0])[:2]).to(self.device)
        # 4. 当前高度
        # observation_all = np.append(observation_all, np.array(currentPos[-1] / self.MAX_HEIGHT_RESTRICTION))

        # 5. 到信号点的方向 + 距离 + 数据剩余量 + 吸引力
        self.max_attratcion_SP = 0.
        self.max_attratcion_SP_pos = None
        sensed_signalPointdata = np.array([])
        for SP, data in self.signalPointId2data.items():
            pos_SP, _ = p.getBasePositionAndOrientation(SP)
            diss = caculate_2D_distance(currentPos, pos_SP)
            if diss <= self.LASER_LENGTH:
                sensed_signalPointdata = np.append(sensed_signalPointdata, direction_normalize(np.subtract(currentPos[:2], pos_SP[:2])))
                sensed_signalPointdata = np.append(sensed_signalPointdata, caculate_2D_distance(currentPos, pos_SP) / self.LASER_LENGTH)
                sensed_signalPointdata = np.append(sensed_signalPointdata, np.array(data))
                curr_SP_attraction = data * min([caculate_2D_distance(pos, pos_SP) for pos in robot_pos]) / diss
                if curr_SP_attraction > self.max_attratcion_SP:
                    self.max_attratcion_SP = curr_SP_attraction
                    self.max_attratcion_SP_pos = pos_SP
                sensed_signalPointdata = np.append(sensed_signalPointdata, curr_SP_attraction)
                



        # print('len', sensed_signalPointdata)

        if len(sensed_signalPointdata) <= self.NUM_MAX_SENSED_SIGNAL_POINT * 6:
            pad = np.ones(self.NUM_MAX_SENSED_SIGNAL_POINT * 6 - len(sensed_signalPointdata))
            sensed_signalPointdata = np.concatenate((sensed_signalPointdata, pad))
        else:
            sensed_signalPointdata = sensed_signalPointdata[:self.NUM_MAX_SENSED_SIGNAL_POINT * 6]
        sensed_signalPointdata = torch.from_numpy(sensed_signalPointdata).to(self.device)
            # direction_to_SP = direction_normalize(np.array(pos_SP) - np.array(currentPos))
            # observation_all = np.append(observation_all, direction_to_SP)
            # distance_to_SP = caculate_distance(currentPos, pos_SP) / self.INTERNAL_LENGTH
            # distance_to_SP = caculate_2D_distance(currentPos, pos_SP) / self.INTERNAL_LENGTH
            # observation_all = np.append(observation_all, distance_to_SP)
            # observation_all = np.append(observation_all, data)
            # curr_SP_attraction = data * min([caculate_2D_distance(pos, pos_SP) for pos in others_pos]) / caculate_2D_distance(currentPos, pos_SP)
            # observation_all = np.append(observation_all, curr_SP_attraction)
            # if curr_SP_attraction > self.max_attratcion_SP:
            #     self.max_attratcion_SP = curr_SP_attraction
            #     self.max_attratcion_SP_pos = pos_SP
            # else:
            #     pass
            
        
        # 6. 当前step
        sstep = torch.tensor([curr_step / self.MAX_STEPS], device = self.device)
        # observation_all = np.append(observation_all, curr_step / self.MAX_STEPS)
        # if not self.max_attratcion_SP_pos:
        #     print("ee")

        # 工作状态
        status = torch.tensor([self.status]).to(self.device)

        # 7. dilemma
        if curr_step == 0:
            dilemma = torch.tensor([0]).to(self.device)
        else:
            dilemma = torch.tensor([self.detect_dilemma(history_trajectory, currentPos)]).to(self.device)

        # node_type
        node_type = torch.tensor([0]).to(self.device)

        # print("p.getBaseVelocity(self.robot):", p.getBaseVelocity(self.robot))
        # print("len(observation_all):", len(observation_all), "observation_all:", observation_all)
    
        # node_type = torch.tensor([0]).to(self.device)
        # print("obstacle_pos", obstacle_pos)
        # print("near_robot_pos", near_robot_pos)
        # print("curr_pos", curr_pos)
        # # print("ang_vel", ang_vel)
        # print("sensed_signalPointdata", sensed_signalPointdata)
        observation_all = torch.cat((obstacle_pos, near_robot_pos, ang_vel, sensed_signalPointdata, sstep, status, dilemma, node_type), dim=0).unsqueeze(0)
        # print(observation_all.shape)
        return observation_all, torch.from_numpy(observed_robot).to(self.device)

    def collision_check(self):  
        # currentPos, _ = p.getBasePositionAndOrientation(self.robot)
        result_getContactPoints = p.getContactPoints(bodyA=self.robot, linkIndexA=-1, physicsClientId=self._physics_client_id)
        if result_getContactPoints:   
            # if result_getContactPoints[0][2] in self.sence_loadItems["obstacle"] or self.sence_loadItems["fence"]:# 碰撞对象不是充电桩
            #     return True
            # else:
            #     return False
            return True
        # elif currentPos[2] >= self.MAX_HEIGHT_RESTRICTION or currentPos[2] <= self.MIN_HEIGHT_RESTRICTION:
        #     return True
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
        """感知范围内的信号点"""
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
            # return pi * sensing_extent**2
            return 1.
        else:
            angle = 2 * acos(distance**2 / 2 / sensing_extent / distance)
            # insection = sensing_extent**2 * angle - sensing_extent**2 * sin(angle)
            # area = pi * sensing_extent**2
            # print(insection, area)
            return (sensing_extent**2 * angle - sensing_extent**2 * sin(angle)) / (pi * sensing_extent**2)

    def detect_dilemma(self, history_trajectory, curr_pos):
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

        for file in os.listdir("hcanet-3.27_maddpg/config"):
            path = "hcanet-3.27_maddpg/config/" + file
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
        激光射线：
            水平（与障碍物距离）
        其他UAV的距离和方向
        当前角速度
        MUAV的剩余电量和已充电量
        当前sstep
        dilemma
        """
        observed_robot = np.ones((2)) * -1
        observed_robot = observed_robot.astype(int)
        currentPos, _ = p.getBasePositionAndOrientation(self.robot)
        # laser
        obstacle_pos = torch.zeros((self.LASER_NUM_HORIZONTAL), dtype=torch.float32, device=self.device)
        unitRayVecs, froms, tos, results = rayTest(self.robot, self.LASER_LENGTH, ray_num_horizontal=self.LASER_NUM_HORIZONTAL, ray_num_vertical=0)
        for index, result in enumerate(results):
            if result[0] != -1:
                if index < self.LASER_NUM_HORIZONTAL:
                    if result[0] in self.sence_loadItems["obstacle"] or self.sence_loadItems["fence"]:
                        obstacle_pos[index] = (caculate_2D_distance(currentPos, result[3]) - self.DRONE_SCALE) / self.LASER_LENGTH
            else:
                # 无返回探测对象
                if index < self.LASER_NUM_HORIZONTAL:
                    obstacle_pos[index] = torch.tensor([1.]).to(self.device)
        
        # 其他robot
        near_robot_pos = torch.ones(((self.NUM_DRONE+self.NUM_CHARGER-1) *3), dtype = torch.float32, device=self.device)
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
        
        # 当前角速度
        ang_vel = torch.tensor(velocilty_normalize(p.getBaseVelocity(self.robot)[0])[:2]).to(self.device)

        t=0
        energy_info = torch.zeros((self.NUM_DRONE*2), dtype=torch.float32, device=self.device)
        # MUAV的剩余电量和已充电量
        for energy in UAV_energy:
            energy_info[2*t]=energy[0]
            energy_info[2*t+1]=energy[1]
            t+=1
            
        # 当前step
        sstep = torch.tensor([curr_step / self.MAX_STEPS], device = self.device)
        
        # 工作状态
        status = torch.tensor([self.status]).to(self.device)

        # dilemma
        if curr_step == 0:
            dilemma = torch.tensor([0]).to(self.device)
        else:
            dilemma = torch.tensor([self.detect_dilemma(history_trajectory, currentPos)]).to(self.device)

        # node_type
        node_type = torch.tensor([1]).to(self.device)

        observation_all = torch.cat((obstacle_pos, near_robot_pos, ang_vel, energy_info, sstep, status, dilemma, node_type), dim=0).unsqueeze(0)
        pad = torch.zeros((self.DIMENSION_OBS[0]-self.DIMENSION_OBS[1]), device=self.device)
        observation_all = torch.cat((observation_all, pad.unsqueeze(0)), dim=1)
        return observation_all, torch.from_numpy(observed_robot).to(self.device)

    def collision_check(self):  
        # currentPos, _ = p.getBasePositionAndOrientation(self.robot)
        result_getContactPoints = p.getContactPoints(bodyA=self.robot, linkIndexA=-1, physicsClientId=self._physics_client_id)
        if result_getContactPoints:   
            # if result_getContactPoints[0][2] in self.sence_loadItems["obstacle"] or self.sence_loadItems["fence"]:# 碰撞对象不是充电桩
            #     return True
            # else:
            #     return False
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
            # return pi * sensing_extent**2
            return 1.
        else:
            angle = 2 * acos(distance**2 / 2 / sensing_extent / distance)
            # insection = sensing_extent**2 * angle - sensing_extent**2 * sin(angle)
            # area = pi * sensing_extent**2
            # print(insection, area)
            return (sensing_extent**2 * angle - sensing_extent**2 * sin(angle)) / (pi * sensing_extent**2)
        
    def detect_dilemma(self, history_trajectory, curr_pos):
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