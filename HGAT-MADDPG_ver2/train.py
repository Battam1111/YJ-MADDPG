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
from utils import *
from controllers import MADDPGController
from rl.replay import Transition
from env.sensingEnv import SensingEnv
from env.utils import *


class PybulletRunner(ABC):

    def __init__(self, resume_run, if_render, device='cpu'):
        self.param_dict = {}
        
        # 加载配置文件
        for file in os.listdir("HGAT-MADDPG_ver2/config"):
            file_path = "HGAT-MADDPG_ver2/config/" + file
            with open(file_path, "r", encoding="utf-8") as f:
                param_dict_current = load(f, Loader=Loader)
            self.param_dict.update(param_dict_current)
        
        self.device = device
        self.checkpoint_file = path.join(self.param_dict["LOG_DIR"], 'logs/')
        self.step_num = 0
        self.episode_num = 0
        previous_step_num = 0

        # 加载模型
        if resume_run:
            resume_path = get_load_path(self.checkpoint_file)
            print(f"Loading model from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=self.device)
            self.checkpoint_dir = latest_logdir(self.checkpoint_file)
        else:
            self.checkpoint_dir = path.join(self.checkpoint_file, datetime.now().strftime('%Y%m%d-%H:%M:%S'))
            if not path.exists(self.checkpoint_dir):
                os.mkdir(self.checkpoint_dir)
        
        self.step_num += previous_step_num

        # 设置随机数种子
        np.random.seed(self.param_dict["RANDOM_SEED"])
        torch.manual_seed(self.param_dict["RANDOM_SEED"])
        torch.cuda.manual_seed(self.param_dict["RANDOM_SEED"])
        save_path = "HGAT-MADDPG_ver2/env/data_signalPoint.npy"
        create_origData(save_path, self.param_dict["NUM_SIGNAL_POINT"], self.param_dict["RANDOM_SEED"])

        # 初始化环境和控制器
        self.env = SensingEnv(self.device, render=if_render)
        self.writer = SummaryWriter(log_dir=self.checkpoint_dir)
        self.node_types = [0, 0, 1]  # 2架UAV + 1架移动充电站
        self.checkpoint_save_episodes = 100
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
        # 从记忆中采样
        return self.memory.sample(
            self.batch_size) if not self.memory.is_prioritized else self.memory.sample(
            self.batch_size, self.replay_buffer_beta)

    def maybe_backup_buffer(self):
        # 备份回放缓冲区
        print('Saving a sample of the replay buffer to file...')
        torch.save(self.memory.copy(), self.replay_buffer_file)

    def run(self):
        # 训练模型
        last_eval = 0
        training_start = datetime.now()
        while self.episode_num < self.param_dict["N_EPISODES"]:
            step_start = self.step_num
            time_start = datetime.now()
            self.total_distance_np = np.zeros(self.param_dict["NUM_DRONE"] + self.param_dict["NUM_CHARGER"])
            self.np_all_energy_consumption = np.zeros((self.param_dict["NUM_DRONE"], 3))

            current_state, cur_adj = self.env.reset()
            data_orig = list(self.env.scene.signalPointId2data.values())
            start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
            self.trajectory = [start_pos]
            self.last_pos = start_pos
            self.episode_num += 1
            total_critic_loss = np.array([0, 0]).astype(np.float32)
            total_policy_loss = np.array([0, 0]).astype(np.float32)
            episode_reward = np.zeros((self.param_dict["NUM_DRONE"] + self.param_dict["NUM_CHARGER"]))
            self.step_num = 0
            for i_step in range(self.param_dict["MAX_STEPS"]):
                self.step_num += 1

                actions = self.controller.act(current_state, cur_adj, self.episode_num, self.param_dict["EPISODES_BEFORE_TRAIN"], True)
                last_actions = actions
                for robot in self.env.robot:
                    p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
                for charger in self.env.charger:
                    p.resetBaseVelocity(charger.robot, linearVelocity=[0., 0., 0.])
                state_, adj_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(self.trajectory))
                self.np_all_energy_consumption += energy_consumption

                if i_step != self.param_dict["MAX_STEPS"] - 1:
                    next_state = state_
                    next_adj = adj_
                else:
                    next_state = None
                    next_adj = None

                episode_reward += reward
                if next_state is not None:
                    self.controller.memory.push(current_state, cur_adj, actions.cpu(), next_state, next_adj, reward, done, self.episode_num)
                    current_state = next_state
                    cur_adj = next_adj

                if self.episode_num > self.param_dict["EPISODES_BEFORE_TRAIN"]:
                    critic_loss, policy_loss = self.controller.update(i_step, self.param_dict)
                    total_critic_loss += np.array([c.cpu().detach().numpy() for c in critic_loss])
                    if policy_loss is not None:
                        total_policy_loss += np.array([p.cpu().detach().numpy() for p in policy_loss])
                    if self.step_num % self.param_dict["SOFT_UPDATE_FREGUENCY"] == 0:
                        self.controller.update_target_net()

                current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
                current_distance_np = np.array([np.sqrt(sum([(x - y) ** 2 for x, y in zip(sp, ep)])) for sp, ep in zip(self.last_pos, current_pos)])
                self.total_distance_np += current_distance_np
                self.last_pos = current_pos
                self.trajectory.append(current_pos)
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
            torch.cuda.empty_cache()

    def evaluate(self):
        # 评估模型
        for i in range(2):
            self.controller.UAVAgent.actor[i].eval()
        self.controller.chargerAgent.actor[0].eval()
        current_state, cur_adj = self.env.reset()
        data_orig = list(self.env.scene.signalPointId2data.values())
        start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
        self.trajectory = [start_pos]
        self.last_pos = start_pos
        with torch.no_grad():
            for i_step in range(self.param_dict["MAX_STEPS"]):
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
                current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
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
                    print("fair_charge", sum(nor_accumulated_charge_energy_list) ** 2 / (len(nor_accumulated_charge_energy_list) * sum(nor_accumulated_charge_energy_list ** 2)))
                    break


if __name__ == "__main__":
    # 检查是否有可用的GPU，如果有则使用第一个GPU，否则使用CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 设置训练模式为"test" train random
    train_mode = "test"
    
    if train_mode == "train":
        # 如果训练模式为训练，初始化PybulletRunner对象，不恢复运行，不渲染，使用指定设备
        runner = PybulletRunner(resume_run=False, if_render=False, device=device)
        try:
            # 运行训练过程
            runner.run()
        except (Exception, KeyboardInterrupt) as e:
            # 如果发生异常且控制器为MADDPGController类型，保存检查点
            if isinstance(runner.controller, MADDPGController):
                print('发生错误，保存检查点...')
                runner.controller.save_checkpoint(runner.step_num, runner.episode_num)

            # 如果异常不是键盘中断，将错误写入日志文件
            if not isinstance(e, KeyboardInterrupt):
                with open(path.join('HGAT-MADDPG_ver2/data/', 'log.txt'), 'a') as f:
                    import traceback
                    f.write(str(e))
                    f.write(traceback.format_exc())
                raise e
    
    elif train_mode == "test":
        # 如果训练模式为测试，初始化PybulletRunner对象，恢复运行，不渲染，使用指定设备
        runner = PybulletRunner(resume_run=True, if_render=False, device=device)
        try:
            # 运行测试过程
            runner.evaluate()
        except (Exception, KeyboardInterrupt) as e:
            # 捕获异常并重新抛出
            raise e