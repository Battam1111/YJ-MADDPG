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
from utils.utils import *
from controllers import MADDPGController
from rl.replay import  Transition
from env.sensingEnv import SensingEnv
from env.utils import *


class PybulletRunner(ABC):

   def __init__(self, resume_run, if_render, device = 'cpu'):
      self.param_dict = {}
      for file in os.listdir("hcanet-3.27_maddpg/config"):
         paths = "hcanet-3.27_maddpg/config/" + file
         param_dict_current = load(open(paths, "r", encoding="utf-8"), Loader=Loader)
         self.param_dict.update(param_dict_current)
      
      self.device = device
      self.checkpoint_file = path.join(self.param_dict["LOG_DIR"], 'logs/')
      self.step_num = 0
      self.episode_num = 0
      previous_step_num = 0

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
      save_path = "hcanet-3.27_maddpg/env/data_signalPoint.npy"
      create_origData(save_path, self.param_dict["NUM_SIGNAL_POINT"], self.param_dict["RANDOM_SEED"])


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
                                       )
      
   def sample_from_memory(self):
      return self.memory.sample(
          self.batch_size) if not self.memory.is_prioritized else self.memory.sample(
              self.batch_size, self.replay_buffer_beta)

   def maybe_backup_buffer(self):
      print('Saving a sample of the replay buffer to file...')
      torch.save(self.memory.copy(), self.replay_buffer_file)


   def run(self):
      last_eval = 0
      training_start = datetime.now()
      while self.episode_num < self.param_dict["N_EPISODES"]:
         step_start = self.step_num
         time_start = datetime.now()
         self.total_distance_np = np.zeros(self.param_dict["NUM_DRONE"]+self.param_dict["NUM_CHARGER"])
         self.np_all_energy_consumption = np.zeros((self.param_dict["NUM_DRONE"],3))

         current_state = self.env.reset()
         data_orig = list(self.env.scene.signalPointId2data.values())
         start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
         self.trajectory = [start_pos]
         self.last_pos = start_pos
         self.episode_num += 1
         total_critic_loss = np.array([0,0]).astype(np.float32)
         total_policy_loss = np.array([0,0]).astype(np.float32)
         episode_reward = np.zeros((self.param_dict["NUM_DRONE"]+self.param_dict["NUM_CHARGER"]))
         self.step_num = 0

         for i_step in range(self.param_dict["MAX_STEPS"]):
            self.step_num += 1

            actions = self.controller.act(current_state, self.episode_num, self.param_dict["EPISODES_BEFORE_TRAIN"], True)
            last_actions = actions
            for robot in self.env.robot:
               p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
            for charger in self.env.charger:
               p.resetBaseVelocity(charger.robot, linearVelocity=[0., 0., 0.])
            state_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(self.trajectory))
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
            current_distance_np = np.array([sqrt(sum([(x - y) * (x - y) for x, y in zip(sp, ep)])) for sp, ep in zip(self.last_pos, current_pos)])
            self.total_distance_np += current_distance_np
            self.last_pos = current_pos
            self.trajectory.append(current_pos)
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

      for i in range(2):
         self.controller.UAVAgent.actor[i].eval()
      self.controller.chargerAgent.actor[0].eval()
      current_state = self.env.reset()
      data_orig = list(self.env.scene.signalPointId2data.values())
      start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
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
               break
   

   def random(self):
      current_state = self.env.reset()
      actions = torch.zeros((3, 2), device = self.device)
      total_reward = np.zeros(3)
      actions[0] = torch.tensor([1,1])
      actions[1] = torch.tensor([-1,-1])
      actions[2] = torch.tensor([1, -1])
      start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot]
      self.trajectory = [start_pos]
      self.last_pos = start_pos
      with torch.no_grad():
         for i_step in range(self.param_dict["MAX_STEPS"]):
            for robot in self.env.robot:
               p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
            state_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(self.trajectory))
            total_reward += reward
            if i_step != self.param_dict["MAX_STEPS"] - 1:
               next_state = state_
            else:
               next_state = None
            if next_state is not None:
               # Move to the next state
               current_state = next_state
            current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot]
            self.last_pos = current_pos
            self.trajectory.append(current_pos)
            if sum(done)>0:
               print(total_reward)
               break
         print(total_reward)


if __name__ == "__main__":


   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

   # train test random

   train_mode = "train"
   if train_mode=="train":
      runner = PybulletRunner(resume_run=False, if_render= False, device=device)
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
   elif train_mode=="test":
      runner = PybulletRunner(resume_run=True, if_render=False, device=device)
      try:
         runner.evaluate()
      except (Exception, KeyboardInterrupt) as e:
         raise e
   elif train_mode=="random":
      runner = PybulletRunner(resume_run=False, if_render=True, device=device)
      try:
         runner.random()
      except (Exception, KeyboardInterrupt) as e:
         raise e
