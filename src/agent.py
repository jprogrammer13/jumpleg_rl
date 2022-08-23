#!/usr/bin/env python3

from email import policy
from netrc import netrc

import joblib
from jumpleg_rl.srv import *
import rospy
import numpy as np
import os
import argparse

from utils import *
import time
import matplotlib.pyplot as plt
from std_msgs.msg import Float32
from JumplegGA import JumplegGA


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

        # Action param
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

        self.CoM0 = np.array([-0.01303,  0.00229,  0.25252])
        self.targetCoM = self.generate_target(self.CoM0)

        # Genetic Algorithm
        self.pop_size = 10
        self.score = np.zeros(self.pop_size)
        self.score_hist = []
        self.generation_n = 0
        self.pop_index = 0

        self.jumplegGA = JumplegGA(
            self.state_dim, self.action_dim, self.pop_size, self.layer_dim, '/home/riccardo/pop_weights.joblib')

        if self.mode == 'inferencce':
            # TODO: load model
            pass

        # Start the node
        rospy.init_node(self.node_name)
        rospy.loginfo(f"JumplegAgent is lissening: {self.mode}")
        rospy.spin()

    def generate_target(self, CoM):

        x = CoM[0]+(-np.random.uniform(0.4, 0.5))
        y = CoM[1]+0
        z = np.random.uniform(0.25, 0.4)

        return [x, y, z]

    def get_target_handler(self, req):
        resp = get_targetResponse()
        resp.target_CoM = self.targetCoM
        return resp

    def get_action_handler(self, req):

        state = np.array(req.state)

        # Generate random action only at the firt episode batch and when the episode number < batch size

        rospy.loginfo(f"Using pop_index: {self.pop_index}")
        action = self.jumplegGA.predict(self.pop_index, state)

        # Performing action conversion
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

        self.score[self.pop_index] = np.array(req.reward)

        resp = set_rewardResponse()
        resp.ack = self.score[self.pop_index]

        self.pop_index += 1
        if (self.pop_index == self.pop_size):
            rospy.loginfo(
                "--------------------------------------------------------------------------")
            rospy.loginfo(f"Generation score: {self.score}")
            self.score_hist.append(self.score)
            joblib.dump(self.score_hist, 'score_hist.joblib')
            rospy.loginfo(
                "--------------------------------------------------------------------------")
            rospy.loginfo(
                f"Generate new generation, gen_numL: {self.generation_n}")
            self.generation_n += 1
            self.jumplegGA.new_generation(self.score)
            self.pop_index = 0
            self.targetCoM = self.generate_target(self.CoM0)

        return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JumplegAgent arguments')
    parser.add_argument('--mode', type=str,
                        default="inference", nargs="?", help='Agent mode')
    args = parser.parse_args(rospy.myargv()[1:])

    jumplegAgent = JumplegAgent(args.mode)
