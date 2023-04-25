#!/usr/bin/env python3

import joblib
from jumpleg_rl.srv import *
import rospy
import numpy as np
import os
import argparse

from ReplayBufferOriginal import ReplayBuffer
from TD3Original import TD3
import time
import matplotlib.pyplot as plt
from utils import *
from std_msgs.msg import Float32
from torch.utils.tensorboard import SummaryWriter


class JumplegAgent:
    def __init__(self, _mode, _data_path, _model_name, _restore_train):

        self.node_name = "JumplegAgentTorque"

        self.mode = _mode
        self.data_path = _data_path
        self.model_name = _model_name
        # convert string back to bool
        self.restore_train = eval(_restore_train)
        rospy.loginfo(f'restore_train: {self.restore_train}')

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

        self.state_dim = 15
        self.action_dim = 3

        # Action limitations
        self.max_torqe = 3

        # Domain of targetCoM
        self.exp_rho = [-np.pi, np.pi]
        self.exp_z = [0.25, 0.5]
        self.exp_r = [0., 0.65]

        # RL
        self.layer_dim = 128

        self.replayBuffer = ReplayBuffer(self.state_dim, self.action_dim)
        self.policy = TD3(self.log_writer, self.state_dim,
                          self.action_dim, self.layer_dim)

        self.batch_size = 128
        self.exploration_noise = 0.4

        self.max_episode_target = 1
        self.episode_counter = 0
        self.real_episode_counter = 0
        self.iteration_counter = 0
        self.random_episode = 1280

        self.test_points = []
        self.rb_dump_it = 100 if self.mode == 'train' else 10

        if self.mode == 'test':
            self.test_points = np.loadtxt(
                os.environ["LOCOSIM_DIR"] + "/robot_control/jumpleg_rl/src/"+'test_points.txt')

        # restore train
        if self.restore_train:
            # del self.replayBuffer
            self.replayBuffer = joblib.load(os.path.join(
                self.main_folder, 'ReplayBuffer_train.joblib'))
            self.iteration_counter = self.replayBuffer.get_number_episodes()

        # if mode is only train the model weights are not restore
        if self.mode != 'train' or self.restore_train:

            if self.restore_train:

                net_iteration_counter = self.iteration_counter - self.random_episode

                # chech if TD3 was already trained
                if net_iteration_counter > 0:
                    self.policy.load(
                        self.data_path, self.model_name, net_iteration_counter)

            else:
                # load pre-trained TD3
                self.policy.load(self.data_path, self.model_name, 0)

        self.episode_transition = {
            "state": None,
            "action": None,
            "next_state": None,
            "reward": None,
            "done": None
        }

        self.targetCoM = self.generate_target()

        # Start ROS node

        rospy.init_node(self.node_name)
        rospy.loginfo(f"JumplegAgent is lissening: {self.mode}")

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

        elif self.mode == 'train':
            if self.episode_counter > self.max_episode_target:
                self.episode_counter = 0
                self.targetCoM = self.generate_target()

        resp.target_CoM = self.targetCoM

        return resp

    def get_action_handler(self, req):
        state = np.array(req.state)
        self.episode_transition['state'] = state

        if self.mode == 'inference' or self.mode == 'test':
            # Get action from policy
            action = self.policy.select_action(state)

        elif self.mode == 'train':
            # Check if we have enought iteration to start the training
            if self.iteration_counter >= self.random_episode:
                # Get action from policy and apply exploration noise
                action = (
                    self.policy.select_action(state) +
                    np.random.normal(
                        0, 1*self.exploration_noise,
                        size=self.action_dim)
                ).clip(-1, 1)
            else:
                # If we don't have enought iteration, genreate random action
                action = np.random.uniform(-1, 1, self.action_dim)

        self.episode_transition['action'] = action
        
        action = action*self.max_torqe

        resp = get_actionResponse()
        resp.action = action
        return resp

    def set_reward_handler(self, req):
        self.episode_transition['next_state'] = np.array(req.next_state)

        self.episode_transition['reward'] = np.array(req.reward)
        self.episode_transition['done'] = np.array(req.done)

        if req.done:
            self.log_writer.add_scalar(
                'Reward', req.reward, self.real_episode_counter)
            self.log_writer.add_scalar(
                'Target Cost(Distance)', req.target_cost, self.real_episode_counter)
            self.log_writer.add_scalar(
                'Unilateral', req.unilateral, self.real_episode_counter)
            self.log_writer.add_scalar(
                'Friction', req.friction, self.real_episode_counter)
            self.log_writer.add_scalar(
                'Singularity', req.singularity, self.real_episode_counter)
            self.log_writer.add_scalar(
                'Joint range', req.joint_range, self.real_episode_counter)
            self.log_writer.add_scalar(
                'Joint torque', req.joint_torques, self.real_episode_counter)
            rospy.loginfo(
                f"Reward[it {self.real_episode_counter}]: {self.episode_transition['reward']}")
            rospy.loginfo(f"Episode transition:\n {self.episode_transition}")

        self.replayBuffer.store(self.episode_transition['state'],
                                self.episode_transition['action'],
                                self.episode_transition['next_state'],
                                self.episode_transition['reward'],
                                self.episode_transition['done']
                                )

        # reset the episode transition
        self.episode_transition = {
            "state": None,
            "action": None,
            "next_state": None,
            "reward": None,
            "done" : None
        }

        if self.mode == 'train':
            if self.real_episode_counter > self.random_episode:
                if req.done:
                    self.policy.train(self.replayBuffer, self.batch_size)
                    net_iteration_counter = self.real_episode_counter - self.random_episode

                    if (net_iteration_counter + 1) % 1000 == 0:

                        rospy.loginfo(
                            f"Saving RL agent networks, epoch {net_iteration_counter}")

                        self.policy.save(os.path.join(
                            self.main_folder, 'partial_weights'), str(net_iteration_counter))

                    self.policy.save(self.data_path, 'latest')

        if req.done:
            if (self.real_episode_counter + 1) % self.rb_dump_it == 0:
                self.replayBuffer.dump(os.path.join(
                    self.main_folder), self.mode)

        resp = set_reward_originalResponse()
        resp.ack = np.array(req.reward)

        if req.done:
            # episode is done only when done is 1
            self.episode_counter += 1
            self.real_episode_counter += 1

        self.iteration_counter += 1

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