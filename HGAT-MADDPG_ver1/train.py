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
      for file in os.listdir("./config"):
         paths = "./config/" + file
         param_dict_current = load(open(paths, "r", encoding="utf-8"), Loader=Loader)
         self.param_dict.update(param_dict_current)
      
      self.device = device
      self.checkpoint_file = path.join(self.param_dict["LOG_DIR"], 'logs/')
      # resume_run = path.isfile(self.checkpoint_file)
      self.step_num = 0
      self.episode_num = 0
      previous_step_num = 0

      if resume_run:
         resume_path = get_load_path(self.checkpoint_file)
         print(f"Loading model from: {resume_path}")
         checkpoint = torch.load(resume_path, map_location=self.device)
         # previous_step_num = checkpoint['total_steps']
         # self.episode_num = checkpoint['n_episodes']
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
      save_path = "./env/data_signalPoint.npy"
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
      self.node_types = [0, 0, 0, 1, 1] #3架UAV+2架移动充电站
      self.checkpoint_save_episodes = 200
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
         current_state, cur_adj = self.env.reset()
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

            actions = self.controller.act(current_state, cur_adj, self.episode_num, self.param_dict["EPISODES_BEFORE_TRAIN"], True)
            # for index, robot in enumerate(self.env.robot):
            #    if robot.status == -1:
            #       actions[index] = torch.tensor([0., 0.], device = self.device)
            last_actions = actions
            for robot in self.env.robot:
               p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
            for charger in self.env.charger:
               p.resetBaseVelocity(charger.robot, linearVelocity=[0., 0., 0.])
            state_, adj_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(self.trajectory))
            # print("state", state_[0].x)
            # print(reward)
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
                # Move to the next state
                current_state = next_state
                cur_adj = next_adj

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

               if dataCollected_percentage >= 0.9:
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
         po_lo = total_policy_loss / (self.step_num * 2)
         cr_lo = total_critic_loss / self.step_num
         self.writer.add_scalar('Reward_sum', np.sum(episode_reward), self.episode_num)
         self.writer.add_scalar('Reward_1', episode_reward[0], self.episode_num)
         self.writer.add_scalar('Reward_2', episode_reward[1], self.episode_num)
         self.writer.add_scalar('Reward_3', episode_reward[2], self.episode_num)
         self.writer.add_scalar('Reward_4', episode_reward[3], self.episode_num)
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
      # for i in range(3):
      self.controller.UAVAgent.actor.eval()
      # self.controller.UAVAgent.critic.eval()
      self.controller.chargerAgent.actor.eval()
      # self.controller.chargerAgent.critic.eval()
      current_state, cur_adj = self.env.reset()
      start_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
      self.trajectory = [start_pos]
      self.last_pos = start_pos
      with torch.no_grad():
         for i_step in range(self.param_dict["MAX_STEPS"]):
            actions = self.controller.act(current_state, cur_adj, self.episode_num, self.param_dict["EPISODES_BEFORE_TRAIN"], False)
            # print(actions[0])
            for robot in self.env.robot:
               p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
            for charger in self.env.charger:
               p.resetBaseVelocity(charger.robot, linearVelocity=[0., 0., 0.])
            state_, adj_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(self.trajectory))
            # print('reward', reward)
            # print(state_[0].x)
            if i_step != self.param_dict["MAX_STEPS"] - 1:
               next_state = state_
               next_adj = adj_
            else:
               next_state = None
            if next_state is not None:
               # Move to the next state
               current_state = next_state
               cur_adj = next_adj
            current_pos = [list(p.getBasePositionAndOrientation(robot.robot)[0]) for robot in self.env.robot] + [list(p.getBasePositionAndOrientation(charger.robot)[0]) for charger in self.env.charger]
            self.last_pos = current_pos
            self.trajectory.append(current_pos)
            if sum(done)>0:
               robot_data_sensed = 0.
               for robot in self.env.robot:
                  robot_data_sensed += robot.dataSensed
               dataCollected_percentage = robot_data_sensed / self.env.scene.data_total
               print("dataCollected_percentage", dataCollected_percentage)
               # json.dump(self.trajectory, open("./data/tra/trajectory_2400.json", "w"), ensure_ascii=False)
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
            # for i in range(3): self.param_dict["MAX_STEPS"]
               # actions[i] =  torch.tensor(2*np.random.rand(self.param_dict["DIMENSION_ACTION"][0])-1).to(self.device)
            # actions[0] = torch.tensor([1, 1], [-1, -1], [1, -1])
            # print("state", current_state[3].x[0])
            for robot in self.env.robot:
               p.resetBaseVelocity(robot.robot, linearVelocity=[0., 0., 0.])
            state_, reward, done, energy_consumption = self.env.step(actions, i_step, np.array(self.trajectory))
            total_reward += reward
            # print('state', state_[0].x)
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

      # if trainer.action_module in TrainingConfig.OFF_POLICY_METHODS:
   # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
   # 使用CUDA_VISIBLE_DEVICES环境变量获取可见的GPU ID
   # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   # gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '')
   # # 将GPU ID拆分为列表
   # gpu_id_list = gpu_ids.split(',')
   # print("可见的GPU ID:", gpu_id_list)

   
   # elif trainer.action_module == TrainingConfig.ActionModuleType.RANDOM:
   #    runner = RandomSMACRunner(trainer)
   train_mode = "test"
   if train_mode=="train":
      runner = PybulletRunner(resume_run=False, if_render= False, device=device)
      try:
         runner.run()
      except (Exception, KeyboardInterrupt) as e:
         if isinstance(runner.controller, MADDPGController):
            print('Something happened, saving checkpoint...')
            runner.controller.save_checkpoint(runner.step_num, runner.episode_num)

         if not isinstance(e, KeyboardInterrupt):
            with open(path.join('data/', 'log.txt'), 'a') as f:
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
