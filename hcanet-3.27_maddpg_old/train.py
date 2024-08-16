import os
# import wandb
from abc import ABC, abstractmethod
from datetime import datetime
from os import path
from yaml import load, Loader
from torch.utils.tensorboard import SummaryWriter
import torch
# from tqdm import tqdm
import pybullet as p
import numpy as np
import json
from utils import *
from controllers import MADDPGController
# from ..encoding import LabelEncoder
from rl.replay import  Transition
# from ..training import TrainingConfig
from env.sensingEnv import SensingEnv
from env.utils import *

# th.autograd.set_detect_anomaly(True)

class PybulletRunner(ABC):

   def __init__(self, resume_run, if_render, device = 'cpu'):
      self.param_dict = {}
      for file in os.listdir("hcanet-3.27_maddpg_old/config"):
         paths = "hcanet-3.27_maddpg_old/config/" + file
         param_dict_current = load(open(paths, "r", encoding="utf-8"), Loader=Loader)
         self.param_dict.update(param_dict_current)
      
      self.device = device
      self.checkpoint_file = path.join(self.param_dict["LOG_DIR"], 'logs/')
      # resume_run = path.isfile(self.checkpoint_file)
      self.step_num = 0
      self.episode_num = 0
      previous_step_num = 0

      if resume_run:
         # resume_path = get_load_path(self.checkpoint_file)
         # print(f"Loading model from: {resume_path}")
         # checkpoint = torch.load(resume_path, map_location=self.device)
         # previous_step_num = checkpoint['total_steps']
         # self.episode_num = checkpoint['n_episodes']
         self.checkpoint_dir = latest_logdir(self.checkpoint_file)
      else:
        self.checkpoint_dir = path.join(self.checkpoint_file, datetime.now().strftime('%Y%m%d-%H-%M-%S'))
        if not path.exists(self.checkpoint_dir):
          os.mkdir(self.checkpoint_dir)
   
      self.step_num += previous_step_num
      # 设置随机数种子
      np.random.seed(self.param_dict["RANDOM_SEED"])
      torch.manual_seed(self.param_dict["RANDOM_SEED"])
      torch.cuda.manual_seed(self.param_dict["RANDOM_SEED"])
      save_path = "hcanet-3.27_maddpg_old/env/data_signalPoint.npy"
      create_origData(save_path, self.param_dict["NUM_SIGNAL_POINT"], self.param_dict["RANDOM_SEED"])
      # if not trainer.resume_run:
      #    trainer.max_num_steps += previous_step_num
      # if self.step_num >= self.param_dict["MAX_STEPS"]:
      #    print("Number of training steps achieved or surpassed. EXITING.")
      #    exit(0)

      # if not trainer.dry_run:
      #    if not path.exists(self.param_dict["LOG_DIR"]):
      #       os.makedirs(self.param_dict["LOG_DIR"])

      self.env = SensingEnv(self.device, render=if_render)
      self.writer = SummaryWriter(log_dir=self.checkpoint_dir)
      self.node_types = [0, 0, 1] #2架UAV+1架移动充电站
      self.checkpoint_save_episodes = 100
      self.controller = MADDPGController(self.checkpoint_file,
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
                                       # self.training_config.training_mode,
                                       )
      
      # if trainer.dry_run:
      #    exit(0)

   def sample_from_memory(self):
      return self.memory.sample(
          self.batch_size) if not self.memory.is_prioritized else self.memory.sample(
              self.batch_size, self.replay_buffer_beta)

   def maybe_backup_buffer(self):
      print('Saving a sample of the replay buffer to file...')
      torch.save(self.memory.copy(), self.replay_buffer_file)

   # def log_episode(self, things_to_log, prefix='episode'):
   #    # add the prefix to arg names
   #    loggers_poggers = {}
   #    for key in things_to_log:
   #       loggers_poggers[prefix + '/' + key] = things_to_log[key]

   #    wandb.log(loggers_poggers, step=self.step_num)

   def run(self):
      last_eval = 0
      training_start = datetime.now()
      while self.episode_num < self.param_dict["N_EPISODES"]:
         step_start = self.step_num
         time_start = datetime.now()
         self.total_distance_np = np.zeros(self.param_dict["NUM_DRONE"]+self.param_dict["NUM_CHARGER"])
         self.np_all_energy_consumption = np.zeros((self.param_dict["NUM_DRONE"],3))

         # episode, episode_reward = self.play_episode()
         current_state = self.env.reset()
         data_orig = list(self.env.scene.signalPointId2data.values())
         start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
         self.trajectory = [start_pos]
         self.last_pos = start_pos
         self.episode_num += 1
         total_critic_loss = np.array([0,0]).astype(np.float32)
         total_policy_loss = np.array([0,0]).astype(np.float32)
         episode_reward = np.zeros((self.param_dict["NUM_DRONE"]+self.param_dict["NUM_CHARGER"]))
         # episode_steps = 0
         self.step_num = 0
         # with torch.no_grad():
         # self.controller.policy_net.eval()
         # self.controller.policy_net.action_layer.init_hidden(1)
         # last_actions = torch.zeros((self.param_dict["NUM_DRONE"]+self.param_dict["NUM_CHARGER"])).type(torch.float32).to(self.device)
         for i_step in range(self.param_dict["MAX_STEPS"]):
            # episode_steps += 1
            self.step_num += 1

            actions = self.controller.act(current_state, self.episode_num, self.param_dict["EPISODES_BEFORE_TRAIN"], True)
            # for index, robot in enumerate(self.env.robot):
            #    if robot.status == -1:
            #       actions[index] = torch.tensor([0., 0.], device = self.device)
            last_actions = actions
            for robot in self.env.robot:
               p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
            for charger in self.env.charger:
               p.resetBaseVelocity(charger.robot, linearVelocity=[0., 0., 0.])
            state_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(self.trajectory))
            # print("state", state_[0].x)
            # print(reward)
            self.np_all_energy_consumption += energy_consumption

            if i_step != self.param_dict["MAX_STEPS"] - 1:
                next_state = state_
            else:
                next_state = None

            episode_reward += reward
            if next_state is not None:
                self.controller.memory.push(current_state, actions.cpu(), next_state, reward, done, self.episode_num)
                # Move to the next state
                current_state = next_state

            if self.episode_num > self.param_dict["EPISODES_BEFORE_TRAIN"]:
               critic_loss, policy_loss = self.controller.update(i_step, self.param_dict)
               total_critic_loss += np.array([c.cpu().detach().numpy() for c in critic_loss])
               if policy_loss != None:
                  total_policy_loss += np.array([p.cpu().detach().numpy() for p in policy_loss])
               if self.step_num % self.param_dict["SOFT_UPDATE_FREGUENCY"] == 0:
                  self.controller.update_target_net()

            current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
            # print("current_pos", current_pos)
            current_distance_np = np.array([sqrt(sum([(x - y) * (x - y) for x, y in zip(sp, ep)])) for sp, ep in zip(self.last_pos, current_pos)])
            self.total_distance_np += current_distance_np
            self.last_pos = current_pos
            self.trajectory.append(current_pos)
            # if self.episode_num > self.param_dict["EPISODES_BEFORE_TRAIN"] and critic_loss < 0.0005:
            #    break
            if sum(done) > 0 or i_step == (self.param_dict["MAX_STEPS"] - 1):
               if self.episode_num % self.checkpoint_save_episodes == 0 and self.episode_num > self.param_dict["EPISODES_BEFORE_TRAIN"]:
                  self.controller.save_checkpoint(self.step_num, self.episode_num)

               robot_data_sensed = 0.
               robot_data_sensed_list = []
               for robot in self.env.robot:
                  robot_data_sensed += robot.dataSensed
                  robot_data_sensed_list.append(robot.dataSensed)
               dataCollected_percentage = robot_data_sensed / self.env.scene.data_total
               total_all_energy_consumption = self.np_all_energy_consumption.sum(axis=0)
               total_energy_consumption_sensing = total_all_energy_consumption[1]
               total_energy_consumption_moving = total_all_energy_consumption[2]
               total_energy_consumption = total_all_energy_consumption[0]
               energyEfficiency = total_energy_consumption_sensing / total_energy_consumption

               data_final = list(self.env.scene.signalPointId2data.values())

               if dataCollected_percentage >= 0.85:
                  self.controller.save_checkpoint(self.step_num, self.episode_num)
               # result_train_dict = {
               #       "episode": self.episode_num,
               #       "time_secs":(datetime.now() - time_start).total_seconds(),
               #       "reward_all": list(self.np_reward_agents),
               #       "step_num": self.step_num,
               #       "total_signalPointData": self.env.scene.data_total,
               #       "drone_dataCollected": robot_data_sensed_list,
               #       "dataCollected_percentage": dataCollected_percentage,
               #       "energyConsumption_sensing": total_energy_consumption_sensing,
               #       "energyConsumption_moving": total_energy_consumption_moving,
               #       "energyConsumption_total": total_energy_consumption,
               #       "energyEfficiency": energyEfficiency,
               #       # "step_num_task80": [],
               #       "critic_loss": [critic_loss],
               #       "actor_loss": [policy_loss],
               #       "moving_distance": list(self.total_distance_np),
               #       "fairness": fairness(data_orig, data_final)}
               break

         # torch.cuda.empty_cache()
         po_lo = total_policy_loss / self.step_num
         cr_lo = total_critic_loss / self.step_num
         self.writer.add_scalar('Reward_sum', np.sum(episode_reward), self.episode_num)
         self.writer.add_scalar('Reward_1', episode_reward[0], self.episode_num)
         self.writer.add_scalar('Reward_2', episode_reward[1], self.episode_num)
         self.writer.add_scalar('Reward_3', episode_reward[2], self.episode_num)
         # self.writer.add_scalar('Reward_4', episode_reward[3], self.episode_num)
         self.writer.add_scalar('Episode_length', self.step_num, self.episode_num)
         self.writer.add_scalar('Critic_loss_UAV', cr_lo[0], self.episode_num)
         self.writer.add_scalar('Policy_loss_UAV', po_lo[0], self.episode_num)
         self.writer.add_scalar('Critic_loss_charger', cr_lo[1], self.episode_num)
         self.writer.add_scalar('Policy_loss_charger', po_lo[1], self.episode_num)
         self.writer.add_scalar('dataCollected_percentage', dataCollected_percentage, self.episode_num)
         print(f'Episode:{self.episode_num}, step_num={self.step_num}, reward={episode_reward}, critic_loss={cr_lo}, policy_loss={po_lo}, dataCollected_percentage={dataCollected_percentage}, time={datetime.now() - time_start}')
         print("---------------------------------------------------------------------------")
         # 追踪并保存模型参数和梯度
         # if self.episode_num > 200:
         #    with open('./data/weights.txt', 'a') as f_weights, open('./data/gradients.txt', 'a') as f_gradients:
         #       for name, param in self.controller.UAVAgent.actor[0].named_parameters():
         #          if param.requires_grad:
         #             f_weights.write(f"actor1_{name} {param.data}\n")
         #             if param.grad is not None:
         #                f_gradients.write(f"actor1_{name} {param.grad}\n")

               # for name, param in self.controller.UAVAgent.actor[1].named_parameters():
               #    if param.requires_grad:
               #       f_weights.write(f"actor2_{name} {param.data}\n")
               #       if param.grad is not None:
               #          f_gradients.write(f"actor2_{name} {param.grad}\n")

               # for name, param in self.controller.UAVAgent.actor[2].named_parameters():
               #    if param.requires_grad:
               #       f_weights.write(f"actor3_{name} {param.data}\n")
               #       if param.grad is not None:
               #          f_gradients.write(f"actor3_{name} {param.grad}\n")

               # for name, param in self.controller.UAVAgent.critic.named_parameters():
               #    if param.requires_grad:
               #       f_weights.write(f"critic_{name} {param.data}\n")
               #       if param.grad is not None:
               #          f_gradients.write(f"critic_{name} {param.grad}\n")

         torch.cuda.empty_cache()

   def evaluate(self):
      # time_start = datetime.now()
      # battles_won = dead_allies = dead_enemies = eval_reward = 0
      for i in range(2):
         self.controller.UAVAgent.actor[i].eval()
      # self.controller.UAVAgent.critic.eval()
      self.controller.chargerAgent.actor[0].eval()
      # self.controller.chargerAgent.critic.eval()
      current_state = self.env.reset()
      data_orig = list(self.env.scene.signalPointId2data.values())
      start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
      self.trajectory = [start_pos]
      self.last_pos = start_pos
      with torch.no_grad():
         for i_step in range(self.param_dict["MAX_STEPS"]):
            actions = self.controller.act(current_state, self.episode_num, self.param_dict["EPISODES_BEFORE_TRAIN"], False)
            # print(actions[0])
            for robot in self.env.robot:
               p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
            for charger in self.env.charger:
               p.resetBaseVelocity(charger.robot, linearVelocity=[0., 0., 0.])
            state_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(self.trajectory))
            # print('reward', reward)
            # print(state_[0].x)
            if i_step != self.param_dict["MAX_STEPS"] - 1:
               next_state = state_
            else:
               next_state = None
            if next_state is not None:
               # Move to the next state
               current_state = next_state
            current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
            self.last_pos = current_pos
            self.trajectory.append(current_pos)
            if sum(done)>0:
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
               print("energy_usage", sum(energy_usage)/len(energy_usage))
               for robot in self.env.charger:
                  print("charge_steps_ratio", charger.charge_steps/i_step)
               accumulated_charge_energy_list = []
               for robot in self.env.robot:
                  accumulated_charge_energy_list.append(robot.accumulated_charge_energy)
               nor_accumulated_charge_energy_list = np.array([en / sum(accumulated_charge_energy_list) for en in accumulated_charge_energy_list])
               print("fair_charge", sum(nor_accumulated_charge_energy_list)**2 / (len(nor_accumulated_charge_energy_list) * sum(nor_accumulated_charge_energy_list**2)))

               # json.dump(self.trajectory, open("./data/tra/trajectory_3300.json", "w"), ensure_ascii=False)
               break
   

   def random(self):
      # 重置环境，获取初始状态
      current_state = self.env.reset()
      # 获取初始场景中的信号点数据
      data_orig = list(self.env.scene.signalPointId2data.values())
      # 初始化动作tensor
      actions = torch.zeros((3, 2), device=self.device)
      # 初始化总奖励
      total_reward = np.zeros(3)
      # 获取机器人和充电器的初始位置
      start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
      # 初始化轨迹
      self.trajectory = [start_pos]
      # 记录上一次的位置
      self.last_pos = start_pos
      # 禁用梯度计算以提高性能
      with torch.no_grad():
         for i_step in range(self.param_dict["MAX_STEPS"]):
            # 获取当前每个机器人的位置
            UAV_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot]
            # 获取当前每个机器人的电量和充电能量
            UAV_energy = [[robot.electricity, robot.charged_energy] for robot in self.env.robot]
            # 为每个机器人选择贪婪动作
            for i, robot in enumerate(self.env.robot):
               actions[i] = robot.greedy_action()
            # 为每个充电器选择贪婪动作
            for i, robot in enumerate(self.env.charger):
               actions[i+2] = robot.greedy_action(UAV_energy, UAV_pos)
            # 重置每个机器人的速度
            for robot in self.env.robot:
               p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
            # 重置每个充电器的速度
            for charger in self.env.charger:
               p.resetBaseVelocity(charger.robot, linearVelocity=[0., 0., 0.])
            # 执行环境的step函数，获取下一个状态、奖励、完成标志和能量消耗
            state_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(self.trajectory))
            # 如果不是最后一步，则更新下一状态
            if i_step != self.param_dict["MAX_STEPS"] - 1:
               next_state = state_
            else:
               next_state = None
            # 如果存在下一状态，则移动到下一状态
            if next_state is not None:
               current_state = next_state
            # 获取当前每个机器人和充电器的位置
            current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
            # 更新上一次的位置
            self.last_pos = current_pos
            # 记录轨迹
            self.trajectory.append(current_pos)
            # 如果有任何一个机器人完成任务，则退出循环
            if sum(done) > 0:
               robot_data_sensed = 0.
               # 统计每个机器人感知到的数据量
               for robot in self.env.robot:
                  robot_data_sensed += robot.dataSensed
               # 计算数据收集百分比
               dataCollected_percentage = robot_data_sensed / self.env.scene.data_total
               print("dataCollected_percentage", dataCollected_percentage)
               # 获取最终场景中的信号点数据
               data_final = list(self.env.scene.signalPointId2data.values())
               # 计算公平性
               fair = fairness(data_orig, data_final)
               print("fair", fair)
               # 统计能量消耗
               energy_usage = []
               for robot in self.env.robot:
                  energy_usage.append(robot.consumption_energy / (1 + robot.accumulated_charge_energy))
               print("energy_usage", sum(energy_usage) / len(energy_usage))
               # 打印充电步数与总步数的比例
               for robot in self.env.charger:
                  print("charge_steps_ratio", charger.charge_steps / i_step)
               # 统计累积充电能量
               accumulated_charge_energy_list = []
               for robot in self.env.robot:
                  accumulated_charge_energy_list.append(robot.accumulated_charge_energy)
               # 归一化累积充电能量
               nor_accumulated_charge_energy_list = np.array([en / sum(accumulated_charge_energy_list) for en in accumulated_charge_energy_list])
               # 计算充电公平性
               print("fair_charge", sum(nor_accumulated_charge_energy_list) ** 2 / (len(nor_accumulated_charge_energy_list) * sum(nor_accumulated_charge_energy_list ** 2)))
               # 退出循环
               break



if __name__ == "__main__":

   # 检查是否有可用的GPU，如果有则使用第一个GPU，否则使用CPU
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

   # 设置训练模式为"test" random
   train_mode = "random"
   
   if train_mode == "train":
      # 如果训练模式为训练，则初始化PybulletRunner对象，不恢复运行，不渲染，使用指定设备
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
            with open(path.join('data/', 'log.txt'), 'a') as f:
               import traceback
               f.write(str(e))
               f.write(traceback.format_exc())
            raise e
   
   elif train_mode == "test":
      # 如果训练模式为测试，则初始化PybulletRunner对象，恢复运行，渲染，使用指定设备
      runner = PybulletRunner(resume_run=True, if_render=True, device=device)
      runner.evaluate()
      # 下面是一些注释掉的代码，可能用于模型选择
      # cheli = './data/logs/20240414-11-14-22'
      # models = [file for file in os.listdir(cheli) if 'model' in file]
      # modelname = []
      # for model in models:
      #    resume_path = os.path.join(cheli, model)
      #    if runner.evaluate() > 0.6:
      #       modelname.append(model)
      # json.dump(modelname, open("./data/op_model.json", "w"), ensure_ascii=False)
   
   elif train_mode == "random":
      # 如果训练模式为随机，则初始化PybulletRunner对象，不恢复运行，渲染，使用指定设备
      runner = PybulletRunner(resume_run=False, if_render=True, device=device)
      try:
         # 运行随机过程
         runner.random()
      except (Exception, KeyboardInterrupt) as e:
         raise e
