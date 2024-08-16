from collections import Counter
from random import seed
import gym
from yaml import load, Loader
import numpy as np

from env.scene import *
from env.robot import *

class SensingEnv(gym.Env):
    """
    传感环境类，继承自 gym.Env
    """
    def __init__(self, device, render: bool = False):
        """
        初始化 SensingEnv 类

        参数:
            device (str): 设备类型，如 'cpu' 或 'cuda'
            render (bool): 是否渲染环境
        """
        # 加载配置文件
        param_path = "HGAT-MADDPG_ver2/config/task.yaml"
        param_dict = load(open(param_path, "r", encoding="utf-8"), Loader=Loader)
        for key, value in param_dict.items():
            setattr(self, key, value)
        
        self._render = render
        # 根据参数选择引擎的连接方式
        self._physics_client_id = p.connect(p.GUI if self._render else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.device = device
        self.seed(self.RANDOM_SEED)
        self.dilemma = [[] for _ in range(self.NUM_DRONE)]

    def seed(self, seed):
        """
        设置随机数种子
        参数:
            seed (int): 随机数种子
        返回:
            list: 包含设置的随机数种子
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def UAV_reward_func(self, whole_last_pos, whole_current_pos, last_signalPointId2data, current_signalPointId2data, curr_dilemma, action, curr_step):
        """
        MUAV 奖励函数

        参数:
            whole_last_pos (list): 所有无人机的上一次位置
            whole_current_pos (list): 所有无人机的当前位位置
            last_signalPointId2data (dict): 上一次的信号点数据量
            current_signalPointId2data (dict): 当前的信号点数据量
            curr_dilemma (list): 当前困境
            action (list): 动作
            curr_step (int): 当前步数

        返回:
            list: 奖励值列表
        """
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
                        distance_laser.append(caculate_2D_distance(current_pos, result[3]) - self.DRONE_SCALE)
                else:
                    distance_laser.append(self.LASER_LENGTH)
            iftooshort = (np.array(distance_laser) < self.DISTANCE_LASER_TOO_CLOSE).astype(int)
            if sum(iftooshort) == 0:
                reward_current += self.REWARD_NOT_TOO_CLOSE
            else:
                penalty_current += sum(iftooshort) * self.PENALTY_TOO_CLOSE

            if robot.collision_check():
                penalty_current += self.PENALTY_COLLISON

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
                if dist < self.POTENTIAL_SENSING_EXTENT:
                    last_SP_inPotentialExtent_dict[SP] = idAndDistanceAndData(SP, dist, data)
                    last_SP_id_inPotentialExtent_list.append(SP)
            for SP, data in current_signalPointId2data.items():
                pos_SP, _ = p.getBasePositionAndOrientation(SP)
                dist = caculate_2D_distance(current_pos, pos_SP)
                if dist < self.POTENTIAL_SENSING_EXTENT:
                    current_SP_inPotentialExtent_dict[SP] = idAndDistanceAndData(SP, dist, data)
                    current_SP_id_inPotentialExtent_list.append(SP)

            len_last_SP_inPotentialExtent_dict = len(last_SP_inPotentialExtent_dict)
            len_current_SP_inPotentialExtent_dict = len(current_SP_inPotentialExtent_dict)
            if len_current_SP_inPotentialExtent_dict > len_last_SP_inPotentialExtent_dict:
                reward_current += self.REWARD_POTENTIAL_SENSING

            if len_last_SP_inPotentialExtent_dict > 0 and len_current_SP_inPotentialExtent_dict > 0:
                intersection_lastAndCurrent = list(set(last_SP_id_inPotentialExtent_list).intersection(set(current_SP_id_inPotentialExtent_list)))
                if len(intersection_lastAndCurrent) > 0:
                    for SP in intersection_lastAndCurrent:
                        if current_SP_inPotentialExtent_dict[SP].distance < last_SP_inPotentialExtent_dict[SP].distance:
                            reward_current += self.REWARD_POTENTIAL_SENSING
                            break

            if robot.dataSensed_current <= 0.:
                last_allRestSP_list = sorted(last_allRestSP_list, key=lambda x: x.distance)
                closest_SP = last_allRestSP_list[0].id
                pos_closestSP, _ = p.getBasePositionAndOrientation(closest_SP)
                angle_to_closest_SP = np.array(pos_closestSP) - np.array(last_pos)
                flag_exist_obstacle, obstacle_id = exist_obstacle(last_pos, pos_closestSP, self.scene.load_items["obstacle"])
                if not flag_exist_obstacle:
                    if sum(action[index].detach().cpu().numpy()[:2] * angle_to_closest_SP[:2]) <= 0:
                        penalty_current += self.PENALTY_INSPIRE
                    else:
                        reward_current += self.REWARD_INSPIRE
                else:
                    if sum(action[index].detach().cpu().numpy()[:2] * angle_to_closest_SP[:2]) > 0:
                        reward_current += self.REWARD_INSPIRE
                    else:
                        penalty_current += self.PENALTY_INSPIRE

            if curr_step >= self.STEP_THRESHOLD:
                penalty_current += self.PENALTY_STEP_THRESHOLD

            if curr_dilemma[index] == 1:
                penalty_current += self.PENALTY_DILEMMA
            elif curr_dilemma[index] == 2:
                penalty_current += 50

            if robot.max_attratcion_SP_pos:
                if sum(action[index].detach().cpu().numpy()[:2] * robot.max_attratcion_SP_pos[:2]) > 0:
                    reward_current += self.REWARD_CLOSE_TO_MOST_ATTRACTIVE_SP

            penalty_current += self.PENALTY_CONSTANT
            reward_current -= penalty_current
            reward_list.append(reward_current)

        return reward_list

    def charger_reward_func(self, whole_current_pos, UAV_energy, UAV_pos, charger_target):
        """
        CUAV奖励函数

        参数:
            whole_current_pos (list): 所有充电无人机的当前位置
            UAV_energy (list): 所有无人机的电量
            UAV_pos (list): 所有无人机的位置
            charger_target (dict): 充电目标

        返回:
            list: 奖励值列表
        """
        reward_list = []
        for index, charger in enumerate(self.charger):
            penalty_current = 0.
            reward_current = 0.
            current_pos = whole_current_pos[index]

            unitRayVecs, froms, tos, results = rayTest(charger.robot, self.LASER_LENGTH, ray_num_horizontal=self.LASER_NUM_HORIZONTAL, ray_num_vertical=0)
            distance_laser = []
            for idx, result in enumerate(results):
                if result[0] != -1:
                    if idx < self.LASER_NUM_HORIZONTAL or result[0] != 0:
                        distance_laser.append(caculate_2D_distance(current_pos, result[3]) - self.DRONE_SCALE)
                else:
                    distance_laser.append(self.LASER_LENGTH)
            iftooshort = (np.array(distance_laser) < self.DISTANCE_LASER_TOO_CLOSE).astype(int)
            if sum(iftooshort) == 0:
                reward_current += self.REWARD_NOT_TOO_CLOSE
            else:
                penalty_current += sum(iftooshort) * self.PENALTY_TOO_CLOSE

            if charger.collision_check():
                penalty_current += self.PENALTY_COLLISON

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

            sorted_id = sorted(range(len(remain_energy_list)), key=lambda k: remain_energy_list[k])
            distance = caculate_2D_distance(UAV_pos[sorted_id[0]], current_pos)
            if charger.status == 0 or iftoomuch:
                reward_current += self.DIS_TO_CLOSEST_UAV * distance + self.ENERGY_TO_CLOSEST_UAV * remain_energy_list[sorted_id[0]]

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
        """
        更新收集到的数据量和信号点数据量
        """
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
        """
        更新 MUAV 和 CUAV 的状态及 MUAV 的电量

        参数:
            dict_pos (dict): MUAV 的位置字典

        返回:
            dict: 充电配对情况
        """
        for robot in self.robot:
            robot.status = 0
        for charger in self.charger:
            charger.status = 0

        charger_target = dict()  # 充电无人机与 MUAV 的配对情况
        charge_dis = dict()  # 充电无人机与最近 MUAV 的距离
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
                robot.electricity += self.CHARGE
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
            elif (robot.electricity - self.ENERGY_PERSTEP - robot.dataSensed_current * self.ENERGY_PERSENSE) < self.ENERGY_SHREHOLD:
                robot.status = -1
                robot.electricity = max(0, robot.electricity - self.ENERGY_PERSTEP - robot.dataSensed_current * self.ENERGY_PERSENSE)
                robot.consumption_energy += min(robot.electricity, self.ENERGY_PERSTEP - robot.dataSensed_current * self.ENERGY_PERSENSE)
            else:
                robot.electricity -= (self.ENERGY_PERSTEP + robot.dataSensed_current * self.ENERGY_PERSENSE)
                robot.consumption_energy += self.ENERGY_PERSTEP + robot.dataSensed_current * self.ENERGY_PERSENSE
        return charger_target

    def step(self, action, curr_step, trajectory):
        """
        环境演化

        参数:
            action (list): 动作列表，包括所有无人机和充电器的动作
            curr_step (int): 当前步数
            trajectory (list): 无人机或充电器的路径信息

        返回:
            tuple: (global_x, adj, reward, dones, energy_consumption_list)
                global_x (torch.Tensor): 所有无人机和充电器的观察值
                adj (torch.Tensor): 邻接矩阵，表示无人机和充电器之间的连接关系
                reward (float): 当前步的总奖励值
                dones (list): 标志列表，表示是否结束
                energy_consumption_list (np.array): 能量消耗信息
        """
        # 1. 获取所有无人机的初始位置
        last_UAV_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.robot]

        # 2. 执行动作
        for index, robot in enumerate(self.robot):
            robot.apply_action(action=action[index])  # 每个无人机执行其对应的动作
        for index, charger in enumerate(self.charger):
            charger.apply_action(action=action[index + self.NUM_DRONE])  # 每个充电器执行其对应的动作

        # 3. 进行物理引擎步进
        p.stepSimulation()

        # 4. 更新信号点和数据收集情况
        last_signalPointId2data = self.scene.signalPointId2data
        self.update_signalPoint_And_dataCollected()

        # 5. 更新无人机和充电器的位置信息
        UAV_dict_pos = {robot.robot: list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.robot}
        charger_target = self.update_energy(UAV_dict_pos)

        # 6. 更新信号点数据
        current_signalPointId2data = self.scene.signalPointId2data

        # 7. 获取当前无人机和充电器的位置
        UAV_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.robot]
        charger_pos = [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.charger]

        # 8. 获取执行动作前的无人机能量状态
        before_UAV_energy = [[robot.electricity, robot.charged_energy] for robot in self.robot]

        # 9. 计算充电器的奖励
        reward_charger = self.charger_reward_func(charger_pos, before_UAV_energy, UAV_pos, charger_target)

        # 10. 获取执行动作后的无人机能量状态
        after_UAV_energy = [[robot.electricity, robot.charged_energy] for robot in self.robot]

        # 11. 初始化全局观察值和邻接矩阵
        global_x = torch.zeros((self.NUM_DRONE + self.NUM_CHARGER, self.DIMENSION_OBS[0]), dtype=torch.float32, device=self.device)
        adj = torch.zeros((self.NUM_DRONE + self.NUM_CHARGER, 2), dtype=torch.int64, device=self.device)

        # 12. 获取每个无人机的观察值和邻接信息
        for index, robot in enumerate(self.robot):
            global_x[index], adj[index] = robot.get_observation(UAV_pos, charger_pos, curr_step, trajectory[:, index])

        # 13. 获取每个充电器的观察值和邻接信息
        for index, charger in enumerate(self.charger):
            global_x[index + self.NUM_DRONE], adj[index + self.NUM_DRONE] = charger.get_observation(UAV_pos, charger_pos, after_UAV_energy, curr_step, trajectory[:, index + self.NUM_DRONE])

        # 14. 初始化完成标志、能量消耗列表和数据收集比率列表
        dones = []
        energy_consumption_list = []
        data_collected_unitsAndDistance_ratio_list = []

        # 15. 计算每个无人机的能量消耗并判断是否完成
        for index in range(self.NUM_DRONE):
            rets = self.robot[index].energy_consumption(
                last_UAV_pos[index], UAV_pos[index],
                self.SENSE_COMSUMPTION, self.MOVE_COMSUMPTION,
                self.MAX_DATA_COLLECTED_PER_STEP, self.MAX_DISTANCE_MOVEMENT_PER_STEP
            )
            energy_consumption_list.append(rets[:3])  # 存储能量消耗信息
            data_collected_unitsAndDistance_ratio_list.append(rets[3:])  # 存储数据收集和距离比率信息

            # 判断是否完成
            if self.robot[index].collision_check():  # 碰撞检查
                dones.append(1)
            elif self.robot[index].status == -1:  # 状态异常
                dones.append(1)
            elif sum(self.scene.signalPointId2data.values()) < self.scene.data_total * (1 - self.DATA_SENED_THRESHOLD):  # 数据收集完成检查
                dones.append(1)
            else:
                dones.append(0)

        # 16. 判断充电器是否完成
        for index in range(self.NUM_CHARGER):
            if self.charger[index].collision_check():  # 碰撞检查
                dones.append(1)
            else:
                dones.append(0)

        # 17. 更新困境状态
        idx = 0
        curr_dilemma = [0 for _ in range(self.NUM_DRONE)]
        for i in range(self.NUM_DRONE):
            self.dilemma[idx].append(global_x[i][-2].item())  # 记录当前困境状态
            curr_dilemma[idx] = global_x[i][-2]
            idx += 1

        # 如果连续15步都处于困境，则标记为完成
        if curr_step >= 15:
            for i in range(self.NUM_DRONE):
                if sum(self.dilemma[i][-15:]) == 15:
                    dones[i] = 1
                    curr_dilemma[i] += 1

        # 18. 计算无人机的奖励
        reward_UAV = self.UAV_reward_func(
            last_UAV_pos, UAV_pos,
            last_signalPointId2data, current_signalPointId2data,
            curr_dilemma, action, curr_step
        )

        # 19. 总奖励为无人机奖励和充电器奖励之和
        reward = reward_UAV + reward_charger

        # 20. 返回当前步的观察值、邻接矩阵、奖励、完成标志和能量消耗列表
        return global_x, adj, reward, dones, np.array(energy_consumption_list)


    def reset(self):
        """
        环境重置

        返回:
            tuple: 观察值、邻居节点
        """
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
            self.robot.append(
                Drone(
                    basePos=[self.DEPARTURE_POS[0] + 1.5 * i, self.DEPARTURE_POS[1] + 1.5 * i, self.DEPARTURE_POS[2]],
                    sence_loadItems=self.scene.load_items,
                    signalPointId2data=self.scene.signalPointId2data,
                    physicsClientId=self._physics_client_id,
                    device=self.device,
                    index=i
                )
            )
        for i in range(self.NUM_CHARGER):
            self.charger.append(
                ChargeUAV(
                    basePos=[self.DEPARTURE_POS_CHARGER[i][0], self.DEPARTURE_POS_CHARGER[i][1], self.DEPARTURE_POS_CHARGER[i][2]],
                    sence_loadItems=self.scene.load_items,
                    physicsClientId=self._physics_client_id,
                    device=self.device,
                    index=i + self.NUM_DRONE
                )
            )
        UAV_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.robot]
        charger_pos = [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.charger]
        UAV_energy = [[robot.electricity, robot.charged_energy] for robot in self.robot]
        global_x = torch.zeros((self.NUM_DRONE + self.NUM_CHARGER, self.DIMENSION_OBS[0]), dtype=torch.float32, device=self.device)
        adj = torch.zeros((self.NUM_DRONE + self.NUM_CHARGER, 2), dtype=torch.int64, device=self.device)
        for index, robot in enumerate(self.robot):
            global_x[index], adj[index] = robot.get_observation(UAV_pos, charger_pos, 0, None)
        for index, charger in enumerate(self.charger):
            global_x[index + self.NUM_DRONE], adj[index + self.NUM_DRONE] = charger.get_observation(UAV_pos, charger_pos, UAV_energy, 0, None)
    
        return global_x, adj

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
    while True:
        force_list = [torch.from_numpy(np.random.uniform(-1, 1, (2))) for _ in range(param_dict["NUM_DRONE"])]
        print("force_list", force_list)
        sensingEnv.step(action=force_list, curr_step=count, trajectory=np.zeros((param_dict["NUM_DRONE"] + param_dict["NUM_CHARGER"], 3)))
        time.sleep(1. / 240.)
        if count % 10 == 0:
            print("step_num:", sensingEnv.step_num)
            print("robot.dataSensed:", [robot.dataSensed for robot in sensingEnv.robot])
        count += 1
