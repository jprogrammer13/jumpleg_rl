#!/usr/bin/env python3

from email import policy

import joblib
from jumpleg_rl.srv import *
import rospy
import numpy as np
import os
import argparse

from ReplayBuffer import ReplayBuffer
from TD3 import TD3
from utils import *
import time
import matplotlib.pyplot as plt
from std_msgs.msg import Float32


class JumplegAgent:
    def __init__(self, _mode):

        # Class attribute
        self.node_name = "JumplegAgent"
        self.mode = _mode

        # Service proxy
        self.get_action_srv = rospy.Service(os.path.join(self.node_name, "get_action"), get_action,
                                            self.get_action_handler)
        self.get_target_srv = rospy.Service(os.path.join(self.node_name, "get_target"), get_target,
                                            self.get_target_handler)
        self.set_reward_srv = rospy.Service(os.path.join(self.node_name, "set_reward"), set_reward,
                                            self.set_reward_handler)

        # np.random.seed(613)

        # RL
        self.state_dim = 6
        self.action_dim = 5
        self.max_time = 1.
        self.min_time = 0.1
        self.max_velocity = 2
        self.min_velocity = 0.1
        self.max_extension = 0.32
        self.min_extension = 0.2
        self.min_phi = np.pi/4.
        self.min_phi_d = np.pi/6.
        self.layer_dim = 128

        self.replayBuffer = ReplayBuffer(self.state_dim, self.action_dim)
        self.policy = TD3(self.state_dim, self.action_dim, self.layer_dim)

        oldExperiemt = os.path.exists('/home/riccardo/ReplayBuffer.joblib')

        self.max_episode = 200
        self.episode_counter = 0
        self.first_episode_batch = True
        self.episode_transition = {
            "state": None, "action": None, "next_state": None, "reward": None}
        self.CoM0 = np.array([-0.01303,  0.00229,  0.25252])
        self.targetCoM = self.generate_target(self.CoM0)
        self.batch_size = 128
        self.exploration_noise = 0.3


        if self.mode == 'inferencce':
            # TODO: load model
            pass

        # Start the node
        rospy.init_node(self.node_name)
        rospy.loginfo(f"JumplegAgent is lissening: {self.mode}")
        # Load old saved data
        rospy.loginfo(f"Old data exist: {oldExperiemt}")
        if oldExperiemt:
            rospy.loginfo("Restoring old run")
            self.replayBuffer = joblib.load('/home/riccardo/ReplayBuffer.joblib')
            self.policy.load('/home/riccardo/TD3')
            self.first_episode_batch = False
        rospy.spin()

    def generate_target(self, CoM):
        # rho = np.random.uniform(0, 2 * np.pi)
        # z = 0.25 #np.random.uniform(0.25,0.5)
        # r = np.random.uniform(0.5, 0.6)
        # x = r * np.cos(rho)
        # y = 0  # r * np.sin(rho)

        x = -np.random.uniform(0.4,0.5)
        y = 0
        z = np.random.uniform(0.25,0.4)

        return [x, y, z]

    def get_target_handler(self, req):
        resp = get_targetResponse()
        if self.episode_counter > self.max_episode:
            self.episode_counter = 0
            self.first_episode_batch = False
            self.targetCoM = self.generate_target(self.CoM0)
        resp.target_CoM = self.targetCoM
        return resp

    def get_action_handler(self, req):

        state = np.array(req.state)
        self.episode_transition['state'] = state

        # Generate random action only at the firt episode batch and when the episode number < batch size
         
        if self.episode_counter < self.batch_size and self.first_episode_batch:
            action = np.random.uniform(-1, 1, self.action_dim)
        else:
            # We add exploration noise
            action = (
                self.policy.select_action(state)
                + np.random.normal(0, 1*self.exploration_noise,
                                   size=self.action_dim)
            ).clip(-1, 1)

        self.episode_transition['action'] = action

        # Performing action conversion
        theta, _, _ = cart2sph(state[3], state[4], state[5])

        T_th = (self.max_time - self.min_time) * \
            0.5*(action[0]+1) + self.min_time

        phi = (np.pi/2 - self.min_phi) * (0.5*(action[1]+1)) + self.min_phi

        r = (self.max_extension - self.min_extension) * \
            0.5*(action[2]+1) + self.min_extension

        ComF_x, ComF_y, ComF_z = sph2cart(theta, phi, r)

        phi_d = (np.pi/2 - self.min_phi_d)*(0.5*(action[3]+1))+self.min_phi_d

        r_d =  (self.max_velocity - self.min_velocity) * (0.5*(action[4]+1)) + self.min_velocity

        ComFd_x, ComFd_y, ComFd_z = sph2cart(theta, phi_d, r_d)

        final_action = np.concatenate(( [T_th],
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
        self.episode_transition['next_state'] = next_state

        reward = np.array(req.reward)
        self.episode_transition['reward'] = reward

        rospy.loginfo(
            f"Episode {self.episode_counter}: {self.episode_transition['reward']}")

        rospy.loginfo(f"Transition: {self.episode_transition}")
        self.replayBuffer.store(self.episode_transition['state'],
                                self.episode_transition['action'],
                                self.episode_transition['next_state'],
                                self.episode_transition['reward'])

        # reset the partial transition
        self.episode_transition = {
            "state": None, "action": None, "next_state": None, "reward": None}

        # Train
        if self.episode_counter > (self.batch_size):

            rospy.loginfo(f"TRAINING: {self.episode_counter}")

            # Save models and replaybuffer
            if ((self.episode_counter + 1) % 100) == 0:
                rospy.loginfo("SAVING")
                self.policy.save('/home/riccardo/TD3')
                self.replayBuffer.dump('/home/riccardo/ReplayBuffer.joblib')

            self.policy.train(self.replayBuffer, self.batch_size)

        resp = set_rewardResponse()
        resp.ack = reward
        # print(f"Reward: {reward}")
        self.episode_counter += 1

        return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JumplegAgent arguments')
    parser.add_argument('--mode', type=str,
                        default="inference", nargs="?", help='Agent mode')
    args = parser.parse_args(rospy.myargv()[1:])

    jumplegAgent = JumplegAgent(args.mode)
