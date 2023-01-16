#!/usr/bin/env python3

import joblib
from jumpleg_rl.srv import *
import rospy
import numpy as np
import os
import argparse

from ReplayBuffer import ReplayBuffer
from TD3 import TD3
import time
import matplotlib.pyplot as plt
from utils import *
from std_msgs.msg import Float32
from torch.utils.tensorboard import SummaryWriter


class JumplegAgent:
    def __init__(self, _mode, _data_path, _model_name, _restore_train):

        self.node_name = "JumplegAgent"

        self.mode = _mode
        self.data_path = _data_path
        self.model_name = _model_name
        self.restore_train = _restore_train
        rospy.loginfo(f'restore_train: {self.restore_train}')

        # Service proxy
        self.get_action_srv = rospy.Service(os.path.join(self.node_name, "get_action"), get_action,
                                            self.get_action_handler)
        self.get_target_srv = rospy.Service(os.path.join(self.node_name, "get_target"), get_target,
                                            self.get_target_handler)
        self.set_reward_srv = rospy.Service(os.path.join(self.node_name, "set_reward"), set_reward,
                                            self.set_reward_handler)

        if not os.path.exists(os.path.join(self.data_path, self.mode)):
            os.mkdir(os.path.join(self.data_path, self.mode))

        if not os.path.exists(os.path.join(self.data_path, self.mode, 'logs')):
            os.mkdir(os.path.join(self.data_path, self.mode, 'logs'))

        if not os.path.exists(os.path.join(self.data_path, self.mode, 'partial_weights')):
            os.mkdir(os.path.join(self.data_path, self.mode, 'partial_weights'))

        self.log_writer = SummaryWriter(
            os.path.join(self.data_path, self.mode, 'logs'))

        self.state_dim = 6
        self.action_dim = 5

        # Action limitations
        self.max_time = 1
        self.min_time = 0.1
        self.max_velocity = 3
        self.min_velocity = 0.1
        self.max_extension = 0.32
        self.min_extension = 0.15
        self.min_phi = np.pi/6.
        self.min_phi_d = np.pi/6.

        # Domain of exploration
        self.exp_az = [0, 2*np.pi]
        # self.exp_el = [0, np.pi]
        self.exp_el = [0, np.pi/4]
        # self.exp_r = [0.1, 0.6]
        self.exp_r = [0.35, 0.6]

        # RL
        self.layer_dim = 128

        self.replayBuffer = ReplayBuffer(self.state_dim, self.action_dim)
        self.policy = TD3(self.log_writer, self.state_dim,
                          self.action_dim, self.layer_dim)

        self.batch_size = 128
        self.exploration_noise = 0.4

        self.max_episode_target = 1
        self.episode_counter = 0
        self.iteration_counter = 0
        self.random_episode = 128

        # restore train
        if self.restore_train:
            # del self.replayBuffer
            self.replayBuffer = joblib.load(os.path.join(
                self.data_path, self.mode, 'ReplayBuffer_train.joblib'))
            self.iteration_counter = self.replayBuffer.get_number_episodes()

        # if mode is only train the model weights are note restore
        if self.mode != 'train' or self.restore_train:

            if self.restore_train:

                net_iteration_counter = self.iteration_counter - self.random_episode

                # chech if TD3 was already trained
                if net_iteration_counter > 0:
                    self.policy.load(os.path.join(
                        self.data_path, self.model_name, net_iteration_counter))

            else:
                # load pre-trained TD3
                self.policy.load(os.path.join(
                    self.data_path, self.model_name, 0))

        self.episode_transition = {
            "state": None,
            "action": None,
            "next_state": None,
            "reward": np.zeros(8)
        }

        self.targetCoM = self.generate_target()

        # Start ROS node

        rospy.init_node(self.node_name)
        rospy.loginfo(f"JumplegAgent is lissening: {self.mode}")

        rospy.spin()

    def generate_target(self):
        # generate 3d point in sperical coordinates
        az = np.random.uniform(self.exp_az[0], self.exp_az[1])
        el = np.random.uniform(self.exp_el[0], self.exp_el[1])
        r = np.random.uniform(self.exp_r[0], self.exp_r[1])

        # convert it to cartesian coordinates
        x, y, z = sph2cart(az, el, r)

        return [x, y, z+0.25252]


    def get_target_handler(self, req):
        resp = get_targetResponse()

        if self.mode == 'inference':
            self.targetCoM = self.generate_target()

        elif self.mode == 'test':
            # TODO Generate test semisphere
            pass

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
            if self.iteration_counter > self.random_episode:
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
        self.episode_transition['next_state'] = np.array(req.next_state)

        self.episode_transition['reward'][0] = req.reward
        self.episode_transition['reward'][1] = req.target_cost
        self.episode_transition['reward'][2] = req.unilateral
        self.episode_transition['reward'][3] = req.friction
        self.episode_transition['reward'][4] = req.singularity
        self.episode_transition['reward'][5] = req.joint_range
        self.episode_transition['reward'][6] = req.joint_torques
        self.episode_transition['reward'][7] = req.error_vel_liftoff

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

        rospy.loginfo(
            f"Reward[it {self.iteration_counter}]: {self.episode_transition['reward'][0]}")
        rospy.loginfo(f"Episode transition:\n {self.episode_transition}")

        self.replayBuffer.store(self.episode_transition['state'],
                                self.episode_transition['action'],
                                self.episode_transition['next_state'],
                                self.episode_transition['reward'])

        # reset the episode transition
        self.episode_transition = {
            "state": None,
            "action": None,
            "next_state": None,
            "reward": np.zeros(8)
        }

        if self.mode == 'train':
            if self.iteration_counter > self.random_episode:

                self.policy.train(self.replayBuffer, self.batch_size)

                net_iteration_counter = self.iteration_counter - self.random_episode

                if (net_iteration_counter + 1) % 1000 == 0:

                    rospy.loginfo(
                        f"Saving RL agent networks, epoch {net_iteration_counter}")

                    self.policy.save(os.path.join(
                        self.data_path, self.mode, 'partial_weights'), str(net_iteration_counter))

                self.policy.save(self.data_path, 'latest')

        self.replayBuffer.dump(os.path.join(
            self.data_path, self.mode), self.mode)

        resp = set_rewardResponse()
        resp.ack = req.reward

        self.episode_counter += 1
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
    parser.add_argument('--restore_train', type=bool,
                        default=False, nargs="?", help='Restore training flag')

    # args = parser.parse_args(rospy.myargv()[1:])
    rospy.loginfo(f'haloooo: rospy.myargv()')

    # jumplegAgent = JumplegAgent(args.mode, args.data_path, False)
    jumplegAgent = JumplegAgent(
        'train', '/home/riccardo/jumpleg_agent', 'latest', False)