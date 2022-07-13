#!/usr/bin/env python3

from jumpleg_rl.srv import *
import rospy
import numpy as np
import os
import argparse

from utils import ReplayBuffer
# import TD3

class JumplegAgent:
    def __init__(self, _mode):

        # Class attribute
        self.node_name = "JumplegAgent"
        self.mode = _mode

        # Service proxy
        self.get_action_srv = rospy.Service(os.path.join(self.node_name, "get_action"), get_action, self.get_action_handler)
        self.get_target_srv = rospy.Service(os.path.join(self.node_name, "get_target"), get_target, self.get_target_handler)
        self.set_reward_srv = rospy.Service(os.path.join(self.node_name, "set_reward"), set_reward, self.set_reward_handler)
        
        # Start the node
        rospy.init_node(self.node_name)
        rospy.loginfo(f"JumplegAgent is lissening: {self.mode}")
        rospy.spin()

    def get_target_handler(self, req):
        resp = get_targetResponse()
        resp.target_CoM = np.random.rand(3)
        return resp

    def get_action_handler(self, req):
        state = req.state

        resp = get_actionResponse()
        thrusting_duration = np.array([1.0])
        haa_coefficient = np.random.rand(6)
        hfe_coefficient = np.random.rand(6)
        kfe_coefficient = np.random.rand(6)

        action = np.concatenate((thrusting_duration, haa_coefficient, hfe_coefficient, kfe_coefficient))

        resp.action = action
        return resp
    
    def set_reward_handler(self, req):
        self.next_state = req.next_state
        self.reward = req.reward
        resp = set_rewardResponse()
        resp.ack = self.reward;
        return resp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JumplegAgent arguments')
    parser.add_argument('--mode', type=str,default="inference",nargs="?",help='Agent mode')
    args = parser.parse_args(rospy.myargv()[1:])

    jumplegAgent = JumplegAgent(args.mode)