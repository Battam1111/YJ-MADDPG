from collections import Counter
from random import seed
import gym
from yaml import load, Loader
import numpy as np

from env.scene import *
from env.robot import *
# from scene import *
# from robot import *

class SensingEnv(gym.Env):
    def __init__(self, device, render : bool = False):
        param_path = "hcanet-3.27_maddpg_old/config/task.yaml"
        param_dict = load(open(param_path, "r", encoding="utf-8"), Loader=Loader)
        for key, value in param_dict.items():
            setattr(self, key, value)
        self._render = render
        # 根据参数选择引擎的连接方式
        self._physics_client_id = p.connect(p.GUI if self._render else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.device = device
        # self.scene = Scence(physicsClientId=self._physics_client_id)
        self.seed(self.RANDOM_SEED)
        # self.reset()
        self.dilemma = [[] for _ in range(self.NUM_DRONE)]

    def seed(self, seed):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def UAV_reward_func(
        self, whole_last_pos, whole_current_pos,
        last_signalPointId2data, current_signalPointId2data,
        curr_dilemma, action, curr_step
        ):
        reward_list = []
        for index, robot in enumerate(self.robot):
            penalty_current = 0.
            reward_current = 0.
            
            last_pos, current_pos = whole_last_pos[index], whole_current_pos[index]

            unitRayVecs, froms, tos, results = rayTest(robot.robot, self.LASER_LENGTH, ray_num_horizontal=self.LASER_NUM_HORIZONTAL, ray_num_vertical=0)
            distance_laser = []
            for idx, result in enumerate(results):
                if result[0] != -1:
                    Pos_hitten, _ = p.getBasePositionAndOrientation(result[0])
                    if idx < self.LASER_NUM_HORIZONTAL or result[0] != 0:
                    # 水平
                        distance_laser.append(caculate_2D_distance(current_pos, result[3]) - self.DRONE_SCALE)
                    # else:
                    #     # 垂直
                    #     distance_vertical = caculate_distance(current_pos, result[3]) - self.DRONE_SCALE
                    #     distance_laser.append(distance_vertical * ((current_pos[2] - self.MIN_HEIGHT_RESTRICTION) / current_pos[2]))
                else:
                    distance_laser.append(self.LASER_LENGTH)
            iftooshort = (np.array(distance_laser) < self.DISTANCE_LASER_TOO_CLOSE).astype(int)
            if sum(iftooshort) == 0:
            # if min(distance_laser) > self.DISTANCE_LASER_TOO_CLOSE:
                reward_current += self.REWARD_NOT_TOO_CLOSE
            else:
                penalty_current += sum(iftooshort) * self.PENALTY_TOO_CLOSE

            # if current_pos[2] > (self.MIN_HEIGHT_RESTRICTION + self.HEIGHT_RESTRICTION_DISTANCE_TOO_CLOSE) and current_pos[2] < (self.MAX_HEIGHT_RESTRICTION - self.HEIGHT_RESTRICTION_DISTANCE_TOO_CLOSE):
            #     reward_current += self.REWARD_HEIGHT_RESTRICTION
            # else:
            #     penalty_current += self.PENALTY_DANGER_HEIGHT

            if robot.collision_check():
                penalty_current += self.PENALTY_COLLISON
            else:
                pass
                
            len_signalPoint_sensed_list = len(robot.signalPoint_sensed_list)
            if len_signalPoint_sensed_list > 0:
                if robot.dataSensed_current > 0:
                    reward_current += (
                        self.REWARD_SENSED * 
                        len_signalPoint_sensed_list * 
                        (robot.dataSensed_current / (len_signalPoint_sensed_list * self.DATA_PER_ROUND)))
                else:
                    penalty_current += self.PENALTY_NO_SENSED
            else:
                penalty_current += self.PENALTY_NO_SENSED

            if sum(self.scene.signalPointId2data.values()) < (1 - self.DATA_SENED_THRESHOLD) * self.scene.data_total:
                reward_current += self.REWARD_SENSING_TASK_COMPLETED
            else:
                pass
            
            last_SP_inPotentialExtent_dict = dict()
            current_SP_inPotentialExtent_dict = dict()
            last_SP_id_inPotentialExtent_list = []
            current_SP_id_inPotentialExtent_list = []
            last_allRestSP_list = []
            zero_data_list = []
            class idAndDistanceAndData(object):
                def __init__(self, id, distance, data):
                    self.id = id
                    self.distance = distance
                    self.data = data
            for SP, data in last_signalPointId2data.items():
                pos_SP, _ = p.getBasePositionAndOrientation(SP)
                dist = caculate_2D_distance(last_pos, pos_SP)
                if data > 0.:
                    last_allRestSP_list.append(idAndDistanceAndData(SP, dist, data))
                # else:
                #     zero_data_list.append(idAndDistanceAndData(SP, dist, data))
                if dist < self.POTENTIAL_SENSING_EXTENT:
                    last_SP_inPotentialExtent_dict[SP] = idAndDistanceAndData(SP, dist, data)
                    last_SP_id_inPotentialExtent_list.append(SP)
                else:
                    pass
            for SP, data in current_signalPointId2data.items():
                pos_SP, _ = p.getBasePositionAndOrientation(SP)
                dist = caculate_2D_distance(current_pos, pos_SP)
                # if data == 0.:
                #     curr_zero_data_list.append()
                if dist < self.POTENTIAL_SENSING_EXTENT:
                    current_SP_inPotentialExtent_dict[SP] = idAndDistanceAndData(SP, dist, data)
                    current_SP_id_inPotentialExtent_list.append(SP)
                else:
                    pass
            len_last_SP_inPotentialExtent_dict= len(last_SP_inPotentialExtent_dict)
            len_current_SP_inPotentialExtent_dict = len(current_SP_inPotentialExtent_dict)
            if len_current_SP_inPotentialExtent_dict > len_last_SP_inPotentialExtent_dict:
                reward_current += self.REWARD_POTENTIAL_SENSING
            else:
                pass
            if len_last_SP_inPotentialExtent_dict > 0 and len_current_SP_inPotentialExtent_dict > 0:
                intersection_lastAndCurrent = list(set(last_SP_id_inPotentialExtent_list).intersection(set(current_SP_id_inPotentialExtent_list)))
                if len(intersection_lastAndCurrent) > 0:
                    for SP in intersection_lastAndCurrent:
                        if current_SP_inPotentialExtent_dict[SP].distance < last_SP_inPotentialExtent_dict[SP].distance:
                            reward_current += self.REWARD_POTENTIAL_SENSING
                            break
                    # penalty_current += self.PENALTY_INTER_POI * len(intersection_lastAndCurrent) / len_current_SP_inPotentialExtent_dict

            # inspire
            if robot.dataSensed_current <= 0.:
                last_allRestSP_list = sorted(last_allRestSP_list, key=lambda x: x.distance)
                closest_SP = last_allRestSP_list[0].id
                pos_closestSP, _ = p.getBasePositionAndOrientation(closest_SP)
                angle_to_closest_SP = np.array(pos_closestSP) - np.array(last_pos)
                flag_exist_obstacle, obstacle_id = exist_obstacle(last_pos, pos_closestSP, self.scene.load_items["obstacle"])
                if flag_exist_obstacle is False: # angle_to_closest_SP 方向上没有障碍物
                    if sum(action[index].detach().cpu().numpy()[:2] * angle_to_closest_SP[:2]) <= 0: # 角度大于90
                        penalty_current += self.PENALTY_INSPIRE
                    else:
                        reward_current += self.REWARD_INSPIRE
                else:
                    # if action[index][2] > 0 and p.getCollisionShapeData(obstacle_id, -1)[0][3][2] > current_pos[2] and sum(action[index].detach().cpu().numpy()[:2] * angle_to_closest_SP[:2]) > 0:
                    #     reward_current += (self.REWARD_INSPIRE + self.REWARD_INSPIRE_FLY_OVER)
                    if sum(action[index].detach().cpu().numpy()[:2] * angle_to_closest_SP[:2]) > 0:
                        reward_current += self.REWARD_INSPIRE
                    else:
                        penalty_current += self.PENALTY_INSPIRE
            else:
                pass

            # step
            if curr_step >= self.STEP_THRESHOLD:
                penalty_current += self.PENALTY_STEP_THRESHOLD
            else:
                pass

            # dilemma
            if curr_dilemma[index] == 1:
                penalty_current += self.PENALTY_DILEMMA
            elif curr_dilemma[index] == 2:
                penalty_current += 50
            else:
                pass

            # most attractive SP
            if robot.max_attratcion_SP_pos:
                if sum(action[index].detach().cpu().numpy()[:2] * robot.max_attratcion_SP_pos[:2]) > 0:
                    reward_current += self.REWARD_CLOSE_TO_MOST_ATTRACTIVE_SP

            penalty_current += self.PENALTY_CONSTANT
            
            reward_current -= penalty_current
            reward_list.append(reward_current)
        
        return reward_list

    def charger_reward_func(self, whole_current_pos, UAV_energy, UAV_pos, charger_target):
        reward_list = []
        for index, charger in enumerate(self.charger):
            penalty_current = 0.
            reward_current = 0.

            current_pos = whole_current_pos[index]

            # dis from the obstacle
            unitRayVecs, froms, tos, results = rayTest(charger.robot, self.LASER_LENGTH, ray_num_horizontal=self.LASER_NUM_HORIZONTAL, ray_num_vertical=0)
            distance_laser = []
            for idx, result in enumerate(results):
                if result[0] != -1:
                    # Pos_hitten, _ = p.getBasePositionAndOrientation(result[0])
                    if idx < self.LASER_NUM_HORIZONTAL or result[0] != 0:
                        distance_laser.append(caculate_2D_distance(current_pos, result[3]) - self.DRONE_SCALE)
                else:
                    distance_laser.append(self.LASER_LENGTH)
            iftooshort = (np.array(distance_laser) < self.DISTANCE_LASER_TOO_CLOSE).astype(int)
            if sum(iftooshort) == 0:
                reward_current += self.REWARD_NOT_TOO_CLOSE
            else:
                penalty_current += sum(iftooshort) * self.PENALTY_TOO_CLOSE

            # collision
            if charger.collision_check():
                penalty_current += self.PENALTY_COLLISON

            # charging reward
            iftoomuch = False
            remain_energy_list = [u for u, _ in UAV_energy]
            charged_energy_list = [min(1, c) for _, c in UAV_energy]
            if charger.status == 1:
                for robot in self.robot:
                        if robot.robot in charger_target.keys() and charger_target[robot.robot] == charger.robot:
                            if robot.electricity > 1.1 or robot.charged_energy > 1:
                                iftoomuch = True
                            else:
                                fairness_factor = sum(charged_energy_list)**2 / (self.NUM_DRONE * sum([c**2 for c in charged_energy_list]))
                                urgency_factor = sum(min([1]*self.NUM_DRONE, remain_energy_list))**2 / (self.NUM_DRONE * sum([r**2 for r in min([1]*self.NUM_DRONE, remain_energy_list)]))
                                sum_factor = self.WEI * fairness_factor + (1-self.WEI) * urgency_factor
                                reward_current += self.REWARD_CHARGE * sum_factor

            # moving to the most needed uav
            sorted_id = sorted(range(len(remain_energy_list)), key=lambda k: remain_energy_list[k])
            distance = caculate_2D_distance(UAV_pos[sorted_id[0]], current_pos)
            # if curr_step < 75:
            #     reward_current += (self.DIS_TO_CLOSEST_UAV * distance + self.ENERGY_TO_CLOSEST_UAV * remain_energy_list[sorted_id[0]])/2
            # else:
            if charger.status == 0 or iftoomuch == True:
                reward_current += self.DIS_TO_CLOSEST_UAV * distance + self.ENERGY_TO_CLOSEST_UAV * remain_energy_list[sorted_id[0]]

            # uav reach the lowest energy and have to stop
            ifenergybelow = (np.array(remain_energy_list) < self.ENERGY_SHREHOLD).astype(int)
            if sum(ifenergybelow) > 0:
                if charger.status == 0:
                    penalty_current += self.PENALTY_NOT_CHARGE
                else:
                    for robot in self.robot:
                        if robot.robot in charger_target.keys() and charger_target[robot.robot] == charger.robot:
                            if robot.electricity > 1 or robot.charged_energy > 1:
                                penalty_current += self.PENALTY_NOT_CHARGE * 6 / 5
                            elif robot.electricity > (sum(remain_energy_list)/len(remain_energy_list)):
                                penalty_current += self.PENALTY_NOT_CHARGE / 3
                            else:
                                penalty_current += self.PENALTY_NOT_CHARGE / 4

            reward_current -= penalty_current
            reward_list.append(reward_current)
            
        for robot in self.robot:
            robot.electricity = min(1, robot.electricity)
            robot.charged_energy = min(1, robot.charged_energy)
        
        return reward_list
                
    def update_signalPoint_And_dataCollected(self):
        """所收集数据量、信号点数据量更新"""
        all_signalPoint_sensed_list = []
        for robot in self.robot:
            all_signalPoint_sensed_list.extend(robot.signalPoint_sensed())
        all_signalPoint_sensed2num = dict(Counter(all_signalPoint_sensed_list))
        for robot in self.robot:
            for SP in robot.signalPoint_sensed_list:
                if all_signalPoint_sensed2num[SP] > 1:
                    if self.scene.signalPointId2data[SP] > all_signalPoint_sensed2num[SP] * self.DATA_PER_ROUND:
                        robot.dataSensed_current += self.DATA_PER_ROUND
                    else:
                        robot.dataSensed_current += self.scene.signalPointId2data[SP] / all_signalPoint_sensed2num[SP]
                else:
                    robot.dataSensed_current += min(self.DATA_PER_ROUND, self.scene.signalPointId2data[SP])
            robot.dataSensed += robot.dataSensed_current
        for SP, num in all_signalPoint_sensed2num.items():
            if num == 1:
                self.scene.signalPointId2data[SP] = max(0, self.scene.signalPointId2data[SP] - self.DATA_PER_ROUND)
            else:
                self.scene.signalPointId2data[SP] = max(0, self.scene.signalPointId2data[SP] - self.DATA_PER_ROUND * num)
        for robot in self.robot:
            robot.signalPointId2data = self.scene.signalPointId2data
            robot.chargerId2state = self.scene.chargerId2state

    def update_energy(self, dict_pos):
        for robot in self.robot:
            robot.status = 0
        for charger in self.charger:
            charger.status = 0

        charger_target = dict() #charger id match UAV id
        charge_dis = dict() #the dis of charger to the cloest UAV
        for charger in self.charger:
            charger_cur_pos = list(p.getBasePositionAndOrientation(charger.robot)[0])
            nearest_UAV_id = 0
            nearest_UAV_dis = 1000
            for UAV_id, UAV_cur_pos in dict_pos.items():
                dis = caculate_2D_distance(UAV_cur_pos, charger_cur_pos)
                if dis <= self.CHARGE_RANGE and dis < nearest_UAV_dis:
                    nearest_UAV_id = UAV_id
                    nearest_UAV_dis = dis
            if nearest_UAV_id != 0:
                charge_dis[charger.robot] = nearest_UAV_dis
                if nearest_UAV_id not in charger_target.keys():
                    charger_target[nearest_UAV_id] = charger.robot
                else:
                    if charge_dis[charger_target[nearest_UAV_id]] > nearest_UAV_dis:
                        charger_target[nearest_UAV_id] = charger.robot
        for robot in self.robot:
            if robot.robot in charger_target.keys():
                robot.electricity = robot.electricity + self.CHARGE
                robot.charged_energy = (robot.charged_energy * self.MAX_CHARGE + self.CHARGE) / self.MAX_CHARGE
                robot.accumulated_charge_energy += self.CHARGE
                robot.status = 1
        for charger in self.charger:
            if charger.robot in charger_target.values():
                charger.status = 1
                charger.charge_steps += 1

        for robot in self.robot:
            if robot.electricity < self.ENERGY_SHREHOLD:
                robot.status = -1
            elif (robot.electricity-self.ENERGY_PERSTEP-robot.dataSensed_current*self.ENERGY_PERSENSE) < self.ENERGY_SHREHOLD:
                robot.status = -1
                robot.electricity = max(0, robot.electricity - self.ENERGY_PERSTEP-robot.dataSensed_current*self.ENERGY_PERSENSE)
                robot.consumption_energy += min(robot.electricity, self.ENERGY_PERSTEP-robot.dataSensed_current*self.ENERGY_PERSENSE)
            else:
                robot.electricity = robot.electricity-self.ENERGY_PERSTEP-robot.dataSensed_current*self.ENERGY_PERSENSE
                robot.consumption_energy += self.ENERGY_PERSTEP-robot.dataSensed_current*self.ENERGY_PERSENSE
        return charger_target

    def step(self, action, curr_step, trajectory):
        last_UAV_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.robot]
        for index, robot in enumerate(self.robot):
            robot.apply_action(action=action[index])
        for index, charger in enumerate(self.charger):
            charger.apply_action(action=action[index+self.NUM_DRONE])
        p.stepSimulation()
        last_signalPointId2data = self.scene.signalPointId2data
        self.update_signalPoint_And_dataCollected()
        UAV_dict_pos = {robot.robot: list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.robot}
        charger_target = self.update_energy(UAV_dict_pos)

        current_signalPointId2data = self.scene.signalPointId2data
        UAV_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.robot]
        charger_pos = [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.charger]
        before_UAV_energy = [[robot.electricity, robot.charged_energy] for robot in self.robot]

        reward_charger = self.charger_reward_func(charger_pos, before_UAV_energy, UAV_pos, charger_target)

        after_UAV_energy = [[robot.electricity, robot.charged_energy] for robot in self.robot]

        global_x = torch.zeros((self.NUM_DRONE + self.NUM_CHARGER, self.DIMENSION_OBS[0]), dtype = torch.float32, device=self.device)
        adj = torch.zeros((self.NUM_DRONE + self.NUM_CHARGER, 2), dtype=torch.int64, device=self.device)
        for index, robot in enumerate(self.robot):
            global_x[index], adj[index] = robot.get_observation(UAV_pos, charger_pos, curr_step, trajectory[:, index])
        for index, charger in enumerate(self.charger):
            global_x[index+self.NUM_DRONE], adj[index+self.NUM_DRONE] = charger.get_observation(UAV_pos, charger_pos, after_UAV_energy, curr_step, trajectory[:, index+self.NUM_DRONE])

        dones = []
        energy_consumption_list = []
        data_collected_unitsAndDistance_ratio_list = []
        for index in range(self.NUM_DRONE):
            rets = self.robot[index].energy_consumption(
                last_UAV_pos[index], UAV_pos[index],
                self.SENSE_COMSUMPTION, self.MOVE_COMSUMPTION,
                self.MAX_DATA_COLLECTED_PER_STEP, self.MAX_DISTANCE_MOVEMENT_PER_STEP
            )
            energy_consumption_list.append(rets[:3])
            data_collected_unitsAndDistance_ratio_list.append(rets[3:])
            if self.robot[index].collision_check():# 与除充电以外的任何物体接触
                dones.append(1)
            elif self.robot[index].status == -1:
                dones.append(1)
            elif sum(self.scene.signalPointId2data.values()) < self.scene.data_total * (1 - self.DATA_SENED_THRESHOLD):
                dones.append(1)
            else:
                dones.append(0)
        for index in range(self.NUM_CHARGER):
            if self.charger[index].collision_check():
                dones.append(1)
            else:
                dones.append(0)
        # dones = np.array(dones)

        # dilemma
        idx = 0
        curr_dilemma = [0 for _ in range(self.NUM_DRONE)]
        for i in range(self.NUM_DRONE):
            self.dilemma[idx].append(global_x[i][-2].item())
            curr_dilemma[idx] = global_x[i][-2]
            idx += 1
        # for i in range(self.NUM_CHARGER):
        #     self.dilemma[idx].append(global_x[i+self.NUM_DRONE][-self.DIMENSION_OBS[0]+self.DIMENSION_OBS[1]-2].item())
        #     curr_dilemma[idx] =global_x[i+self.NUM_DRONE][-self.DIMENSION_OBS[0]+self.DIMENSION_OBS[1]-2]
        #     idx += 1

        if curr_step >= 15:
            for i in range(self.NUM_DRONE):
                if sum(self.dilemma[i][-15:]) == 15:
                    # dones[i] = 1
                    curr_dilemma[i] += 1
        #     print(curr_dilemma)
        # if dones.sum() == 0:
        #     for index_agent in range(self.NUM_DRONE):
        #         curr_dilemma[index_agent] = self.robot[index_agent].detect_dilemma(
        #                                     trajectory[:, index_agent, :2], current_pos[index_agent])
        #     if sum(curr_dilemma) > 0:
        #         print(curr_dilemma)
        # else:
        #     pass

        reward_UAV = self.UAV_reward_func(
            last_UAV_pos, UAV_pos,
            last_signalPointId2data, current_signalPointId2data,
            curr_dilemma, action, curr_step
            )

        reward = reward_UAV + reward_charger
        return global_x, reward, dones, np.array(energy_consumption_list)

    def reset(self):
        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0., 0., -9.8, physicsClientId=self._physics_client_id)
        p.setRealTimeSimulation(0)
        self.step_num = 0
        self.scene = Scence(physicsClientId=self._physics_client_id)
        p.resetDebugVisualizerCamera(cameraDistance=9, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[0,0,0])
        self.scene.construct()
        self.robot = []
        self.charger = []
        for i in range(self.NUM_DRONE):
            # while True:
            #     departure_pos = [self.CENTER_POS[0] + (self.INTERNAL_WIDTH / 2 * random.uniform(-1,1)), self.CENTER_POS[1] + (self.INTERNAL_LENGTH / 2 * random.uniform(-1,1)), 2]
            #     collide = False
            #     for id in (self.scene.load_items["fence"] + self.scene.load_items["obstacle"]):
            #         extents_current = p.getCollisionShapeData(id, -1)[0][3]
            #         pos_current = p.getBasePositionAndOrientation(id)[0]
            #         if ((abs(pos_current[0] - departure_pos[0]) >= (extents_current[0] + self.DRONE_SCALE)) or ((abs(pos_current[1] - departure_pos[1]) >= (extents_current[1] + self.DRONE_SCALE)))):
            #             pass
            #         else:
            #             collide = True
            #             break
            #     if collide == True:
            #         pass
            #     else:
            #         break
            self.robot.append(
                Drone(
                basePos=[self.DEPARTURE_POS[0] + 1.5 * i, self.DEPARTURE_POS[1] + 1.5 * i, self.DEPARTURE_POS[2]],
                # basePos=[
                #     self.DEPARTURE_POS[i][0], #- (1.5 * self.NUM_DRONE / 2) + 1.5 * i, 
                #     self.DEPARTURE_POS[i][1], #- (1.5 * self.NUM_DRONE / 2) + 2 * i, 
                #     self.DEPARTURE_POS[i][2]],
                sence_loadItems=self.scene.load_items,
                signalPointId2data=self.scene.signalPointId2data,
                physicsClientId=self._physics_client_id,
                device=self.device,
                index=i
                ))
            
        for i in range(self.NUM_CHARGER):
            self.charger.append(
                ChargeUAV(
                    basePos=[self.DEPARTURE_POS_CHARGER[i][0], self.DEPARTURE_POS_CHARGER[i][1], self.DEPARTURE_POS_CHARGER[i][2]],
                    sence_loadItems=self.scene.load_items,
                    physicsClientId=self._physics_client_id,
                    device=self.device,
                    index=i+self.NUM_DRONE
                )
            )
        # p.resetDebugVisualizerCamera(cameraDistance=1000, cameraYaw=0, cameraPitch=-90, cameraTargetPosition=[0,0,0])
        # current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.robot]
        UAV_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.robot]
        charger_pos = [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.charger]
        UAV_energy = [[robot.electricity, robot.charged_energy] for robot in self.robot]
        global_x = torch.zeros((self.NUM_DRONE + self.NUM_CHARGER, self.DIMENSION_OBS[0]), dtype = torch.float32, device=self.device)
        adj = torch.zeros((self.NUM_DRONE + self.NUM_CHARGER, 2), dtype=torch.int64, device=self.device)
        for index, robot in enumerate(self.robot):
            global_x[index], adj[index] = robot.get_observation(UAV_pos, charger_pos, 0, None)
        for index, charger in enumerate(self.charger):
            global_x[index+self.NUM_DRONE], adj[index+self.NUM_DRONE] = charger.get_observation(UAV_pos, charger_pos, UAV_energy, 0, None)
    
        return global_x

if __name__ == "__main__":
    param_path = "./config/task.yaml"
    param_dict = load(open(param_path, "r", encoding="utf-8"), Loader=Loader)
    # 设置随机数种子
    np.random.seed(param_dict["RANDOM_SEED"])
    sensingEnv = SensingEnv(render=True)
    sensingEnv.reset()
    print("robot", [robot for robot in sensingEnv.robot])
    print("load_items:", sensingEnv.scene.load_items)
    count = 0
    while (1):
        force_list = [torch.from_numpy(np.random.uniform(-1,1,(2))) for _ in range(param_dict["NUM_DRONE"])]
        print("force_list",force_list )
        sensingEnv.step(action=force_list)
        time.sleep(1. / 240.)
        if count % 10 == 0:
            print("step_num:", sensingEnv.step_num)
            print("robot.dataSensed:", [robot.dataSensed for robot in sensingEnv.robot])
        count += 1