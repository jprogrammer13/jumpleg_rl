#!/usr/bin/env python3

import joblib
from jumpleg_rl.srv import *
import rospy
import numpy as np
import os
import argparse
import pandas as pd

from PPO import PPO
from utils import *
from std_msgs.msg import Float32
from torch.utils.tensorboard import SummaryWriter


class JumplegAgent:
    def __init__(self, _mode, _data_path, _model_name, _restore_train):

        self.node_name = "JumplegAgent"

        self.mode = _mode
        self.data_path = _data_path
        self.model_name = _model_name
        # convert string back to bool
        self.restore_train = eval(_restore_train)

        # Service proxy
        self.get_action_srv = rospy.Service(os.path.join(self.node_name, "get_action"), get_action,
                                            self.get_action_handler)
        self.get_target_srv = rospy.Service(os.path.join(self.node_name, "get_target"), get_target,
                                            self.get_target_handler)
        self.set_reward_srv = rospy.Service(os.path.join(self.node_name, "set_reward"), set_reward,
                                            self.set_reward_handler)

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        self.main_folder = os.path.join(self.data_path, self.mode)

        if not os.path.exists(self.main_folder):
            os.mkdir(self.main_folder)

        if self.mode == 'test':
            self.main_folder = os.path.join(
                self.data_path, self.mode, f'model_{self.model_name}')

            if not os.path.exists(self.main_folder):
                os.mkdir(self.main_folder)

        if not os.path.exists(os.path.join(self.main_folder, 'logs')):
            os.mkdir(os.path.join(self.main_folder, 'logs'))

        if self.mode == 'train':
            if not os.path.exists(os.path.join(self.main_folder, 'partial_weights')):
                os.mkdir(os.path.join(self.main_folder, 'partial_weights'))

        self.log_writer = SummaryWriter(
            os.path.join(self.main_folder, 'logs'))

        self.data = pd.DataFrame(
            columns=['state', 'action', 'next_state', 'reward', 'done'])

        self.state_dim = 6
        self.action_dim = 5

        # Action limitations
        self.max_time = 1
        self.min_time = 0.2
        self.max_velocity = 4
        self.min_velocity = 0.1
        self.max_extension = 0.32
        self.min_extension = 0.25
        self.min_phi = np.pi/4.
        self.min_phi_d = np.pi/6.

        # Domain of targetCoM
        self.exp_rho = [-np.pi, np.pi]
        self.exp_z = [0.25, 0.5]
        self.exp_r = [0., 0.65]

        # RL
        self.layer_dim = 256

        self.action_std = 0.6 if self.mode == 'train' else 0.1

        self.action_std_decay_rate = 0.05
        self.min_action_std = 0.1
        self.action_std_decay_freq = int(5000)

        self.update_timestep = 30      # update ppo_agent every n timesteps
        K_epochs = 10               # update ppo_agent for K epochs
        eps_clip = 0.2              # clip parameter for PPO
        gamma = 0.99                # discount factor

        lr_actor = 3e-4       # learning rate for actor network
        lr_critic = 1e-3       # learning rate for critic network

        self.ppo_agent = PPO(self.state_dim, self.action_dim, lr_actor,
                             lr_critic, gamma, K_epochs, eps_clip, self.action_std)

        self.max_episode_target = 5
        self.episode_counter = 0
        self.iteration_counter = 0

        self.test_points = []

        if self.mode == 'test':
            self.test_points = np.loadtxt(
                os.environ["LOCOSIM_DIR"] + "/robot_control/jumpleg_rl/src/"+'test_points.txt')

        if self.mode != 'train' or self.restore_train:
            self.ppo_agent.load(self.data_path, self.model_name)
            if self.restore_train:
                pw = os.listdir(os.path.join(
                    self.data_path, 'train', 'partial_weights'))
                # restore the episode number from the latest partial weight
                self.episode_counter = np.array(
                    [int(i.split('_')[-1].split('.pt')[0]) for i in pw]).max()

        self.episode_transition = {
            "state": None,
            "action": None,
            "next_state": None,
            "reward": None,
            "done": 1
        }

        self.targetCoM = self.generate_target()

        # Start ROS node
        rospy.init_node(self.node_name)
        rospy.loginfo(f"JumplegAgent is lissening: {self.mode}")
        rospy.loginfo(
            f'restore_train: {self.restore_train}, episode_counter: {self.episode_counter}')

        rospy.spin()

    def generate_target(self):
        rho = np.random.uniform(self.exp_rho[0], self.exp_rho[1])
        z = np.random.uniform(self.exp_z[0], self.exp_z[1])
        r = np.random.uniform(self.exp_r[0], self.exp_r[1])
        x = r * np.cos(rho)
        y = r * np.sin(rho)
        return [-x, y, z]

    def get_target_handler(self, req):
        resp = get_targetResponse()

        if self.mode == 'inference':
            self.targetCoM = self.generate_target()

        elif self.mode == 'test':
            if self.iteration_counter < self.test_points.shape[0]:
                self.targetCoM = self.test_points[self.iteration_counter]
            else:  # send stop signal
                self.targetCoM = [0, 0, -1]
                # save test results in csv
                self.data.to_csv(os.path.join(
                    self.main_folder, 'test.csv'),index=None)

        elif self.mode == 'train':
            if self.episode_counter > self.max_episode_target:
                self.episode_counter = 0
                self.targetCoM = self.generate_target()

        resp.target_CoM = self.targetCoM

        return resp

    def get_action_handler(self, req):
        state = np.array(req.state)
        self.episode_transition['state'] = state

        action = self.ppo_agent.select_action(state)
        self.episode_transition['action'] = action

        # Performs action composition from [-1,1]

        theta, _, _ = cart2sph(state[3], state[4], state[5])

        T_th = (self.max_time - self.min_time) * \
            0.5*(action[0]+1) + self.min_time

        phi = (np.pi/2 - self.min_phi) * (0.5*(action[1]+1)) + self.min_phi

        r = (self.max_extension - self.min_extension) * \
            0.5*(action[2]+1) + self.min_extension

        ComF_x, ComF_y, ComF_z = sph2cart(theta, phi, r)

        phi_d = (np.pi/2 - self.min_phi_d)*(0.5*(action[3]+1))+self.min_phi_d

        r_d = (self.max_velocity - self.min_velocity) * \
            (0.5*(action[4]+1)) + self.min_velocity

        ComFd_x, ComFd_y, ComFd_z = sph2cart(theta, phi_d, r_d)

        final_action = np.concatenate(([T_th],
                                       [ComF_x],
                                       [ComF_y],
                                       [ComF_z],
                                       [ComFd_x],
                                       [ComFd_y],
                                       [ComFd_z]))

        resp = get_actionResponse()
        resp.action = final_action
        return resp

    def set_reward_handler(self, req):
        next_state = np.array(req.next_state)
        reward = np.array(req.reward)
        self.episode_transition['next_state'] = next_state
        self.episode_transition['reward'] = reward
        done = 1

        self.ppo_agent.buffer.rewards.append(reward)
        self.ppo_agent.buffer.is_terminals.append(done)

        self.log_writer.add_scalar(
            'Reward', req.reward, self.iteration_counter)
        self.log_writer.add_scalar(
            'Target Cost(Distance)', req.target_cost, self.iteration_counter)
        self.log_writer.add_scalar(
            'Unilateral', req.unilateral, self.iteration_counter)
        self.log_writer.add_scalar(
            'Friction', req.friction, self.iteration_counter)
        self.log_writer.add_scalar(
            'Singularity', req.singularity, self.iteration_counter)
        self.log_writer.add_scalar(
            'Joint range', req.joint_range, self.iteration_counter)
        self.log_writer.add_scalar(
            'Joint torque', req.joint_torques, self.iteration_counter)
        self.log_writer.add_scalar(
            'Error liftoff vel', req.error_vel_liftoff, self.iteration_counter)
        self.log_writer.add_scalar(
            'Error liftoff pos', req.error_pos_liftoff, self.iteration_counter)
        self.log_writer.add_scalar(
            'Unfeasible vertical velocity', req.unfeasible_vertical_velocity, self.iteration_counter)
        self.log_writer.add_scalar(
            'Touchdown penality', req.no_touchdown, self.iteration_counter)
        self.log_writer.add_scalar(
            'Total cost', req.total_cost, self.iteration_counter)
        rospy.loginfo(
            f"Reward[it {self.iteration_counter}]: {reward}")

        self.iteration_counter += 1
        self.episode_counter += 1

        if self.mode == 'train':
            if self.iteration_counter % self.update_timestep == 0:
                self.ppo_agent.update()

                if (self.episode_counter) % 1000 == 0:

                    rospy.loginfo(
                        f"Saving RL agent networks, epoch {self.episode_counter}")

                    self.ppo_agent.save(os.path.join(
                        self.main_folder, 'partial_weights'), str(self.episode_counter))

                self.ppo_agent.save(self.data_path, 'latest')

            if self.iteration_counter % self.action_std_decay_freq == 0:
                self.ppo_agent.decay_action_std(
                    self.action_std_decay_rate, self.min_action_std)

        if self.mode == 'test':
            self.data = pd.concat(
                [self.data, pd.DataFrame([self.episode_transition])], ignore_index=True)

        # reset the episode transition
        self.episode_transition = {
            "state": None,
            "action": None,
            "next_state": None,
            "reward": None,
            "done": 1
        }

        resp = set_rewardResponse()
        resp.ack = np.array(req.reward)

        return resp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JumplegAgent arguments')
    parser.add_argument('--mode', type=str,
                        default="inference", nargs="?", help='Agent mode')
    parser.add_argument('--data_path', type=str,
                        default=None, nargs="?", help='Path of RL data')
    parser.add_argument('--model_name', type=str,
                        default='latest', nargs="?", help='Iteration of the model')
    parser.add_argument('--restore_train', default=False,
                        nargs="?", help='Restore training flag')

    args = parser.parse_args(rospy.myargv()[1:])

    jumplegAgent = JumplegAgent(
        args.mode, args.data_path, args.model_name, args.restore_train)
