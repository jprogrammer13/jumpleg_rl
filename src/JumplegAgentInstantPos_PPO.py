#!/usr/bin/env python3

import joblib
from jumpleg_rl.srv import *
import rospy
import numpy as np
import os
import argparse

from PPO import PPO
from utils import *
from std_msgs.msg import Float32
from torch.utils.tensorboard import SummaryWriter


class JumplegAgentInstantPos:
    def __init__(self, _mode, _data_path, _model_name, _restore_train):

        self.node_name = "JumplegAgentInstantPos"

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
        self.set_reward_srv = rospy.Service(os.path.join(self.node_name, "set_reward"), set_reward_original,
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

        self.state_dim = 25
        self.action_dim = 3

        # Action limitations
        self.max_q = np.array([np.pi/2, np.pi/2, np.pi/2])

        # Curriculum learning (increese train domain)
        self.curr_learning = 0.5

        # Domain of targetCoM
        self.exp_rho = [-np.pi, np.pi]

        # RL
        avg_ep_len = 1000

        self.layer_dim = 256

        self.action_std = 0.6 if self.mode == 'train' else 0.1

        self.action_std_decay_rate = 0.05
        self.min_action_std = 0.1
        self.action_std_decay_freq = int(2.5e5)

        self.update_timestep = avg_ep_len * 5  # update ppo_agent every n timesteps
        K_epochs = 100               # update ppo_agent for K epochs
        eps_clip = 0.2              # clip parameter for PPO
        gamma = 0.99                # discount factor

        lr_actor = 3e-4       # learning rate for actor network
        lr_critic = 1e-3       # learning rate for critic network

        self.ppo_agent = PPO(self.state_dim, self.action_dim, lr_actor,
                             lr_critic, gamma, K_epochs, eps_clip, self.action_std)

        self.max_episode_target = 20

        self.n_curriculum_episode = 2500
        self.curriculum_step = 0.5 / \
            (self.n_curriculum_episode/self.max_episode_target)

        self.target_episode_counter = 0

        self.episode_counter = 0
        self.iteration_counter = 0
        self.net_iteration_counter = 0

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
                self.net_iteration_counter = np.array(
                    [int(i.split('_')[-1].split('.pt')[0]) for i in pw]).max()

        self.targetCoM = self.generate_target()

        # Start ROS node
        rospy.init_node(self.node_name)
        rospy.loginfo(f"JumplegAgent is listening: {self.mode}")
        rospy.loginfo(
            f'restore_train: {self.restore_train}, net_iteration: {self.net_iteration_counter}')

        rospy.spin()

    def generate_target(self):

        # update the train domain
        self.exp_z = [0.25, self.curr_learning*0.5]
        self.exp_r = [0., self.curr_learning*0.65]

        rho = np.random.uniform(self.exp_rho[0], self.exp_rho[1])
        z = np.random.uniform(self.exp_z[0], self.exp_z[1])
        r = np.random.uniform(self.exp_r[0], self.exp_r[1])
        x = r * np.cos(rho)
        y = r * np.sin(rho)

        # Update training domain while upper bound isn't reached
        if self.curr_learning > 1:
            self.curr_learning = 1
        else:
            self.curr_learning += self.curriculum_step

        return [-x, y, z]

    def get_target_handler(self, req):
        # print('TARGET HANDLER')
        resp = get_targetResponse()

        if self.mode == 'inference':
            self.targetCoM = self.generate_target()

        elif self.mode == 'test':
            if self.episode_counter < self.test_points.shape[0]:
                self.targetCoM = self.test_points[self.episode_counter]
            else:  # send stop signal
                self.targetCoM = [0, 0, -1]
                # TODO: replace buffer with csv save
                # self.replayBuffer.dump(
                #     os.path.join(self.main_folder), self.mode)

        elif self.mode == 'train':
            if self.target_episode_counter > self.max_episode_target:
                self.target_episode_counter = 0
                self.targetCoM = self.generate_target()

        resp.target_CoM = self.targetCoM

        return resp

    def get_action_handler(self, req):
        state = np.array(req.state)

        action = self.ppo_agent.select_action(state)

        action = (action*self.max_q)  # .clip(-np.pi,np.pi)
        resp = get_actionResponse()
        resp.action = action
        return resp

    def set_reward_handler(self, req):
        state = np.array(req.next_state)
        reward = np.array(req.reward)
        done = req.done

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
            'No touchdown', req.no_touchdown, self.iteration_counter)
        self.log_writer.add_scalar(
            'Smoothness', req.smoothness, self.iteration_counter)
        self.log_writer.add_scalar(
            'Straight', req.straight, self.iteration_counter)

        if req.done:
            rospy.loginfo(
                f"Reward[it {self.iteration_counter}]: {reward}")

        if self.mode == 'test':
            # Save results only on the end of the episode (avoid buffer overflow and data loss)
            if req.done:
                # TODO: store into the csv with the final state
                pass

        self.iteration_counter += 1

        if self.mode == 'train':
            if self.iteration_counter % self.update_timestep == 0:
                print('########## Training PPO ################')
                self.ppo_agent.update()
                self.net_iteration_counter += 1

                if (self.net_iteration_counter) % 1000 == 0:
                    rospy.loginfo(
                        f"Saving RL agent networks, net_iteration {self.net_iteration_counter}")
                    self.ppo_agent.save(os.path.join(
                        self.main_folder, 'partial_weights'), str(self.net_iteration_counter))

            self.ppo_agent.save(self.data_path, 'latest')
            if self.iteration_counter % self.action_std_decay_freq == 0:
                self.ppo_agent.decay_action_std(
                    self.action_std_decay_rate, self.min_action_std)

        resp = set_reward_originalResponse()
        resp.ack = np.array(req.reward)

        if req.done:
            # episode is done only when done is 1
            self.target_episode_counter += 1
            self.episode_counter += 1
            print(self.iteration_counter, self.episode_counter)

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

    jumplegAgentTorque = JumplegAgentInstantPos(
        args.mode, args.data_path, args.model_name, args.restore_train)
