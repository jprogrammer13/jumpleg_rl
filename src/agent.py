#!/usr/bin/env python3

from jumpleg_rl.srv import get_action, get_actionResponse, get_target, get_targetResponse, set_reward, set_rewardResponse
import rospy
import numpy as np
import os

class JumplegAgent:
    def __init__(self):

        # Class attribute
        self.node_name = "JumplegAgent"
        self.CoM = []
        self.distance_x = 0
        self.distance_y = 0
        self.distance_z = 0
        self.target_position = np.zeros(3)
        self.last_reward = 0
        self.haa_coefficient = np.zeros(6)
        self.hfe_coefficient = np.zeros(6)
        self.kfe_coefficient = np.zeros(6)

        # Service proxy
        self.get_action_srv = rospy.Service(os.path.join(self.node_name, "get_action"), get_action, self.get_action_handler)
        self.get_target_srv = rospy.Service(os.path.join(self.node_name, "get_target"), get_target, self.get_target_handler)
        self.set_reward_srv = rospy.Service(os.path.join(self.node_name, "set_reward"), set_reward, self.set_reward_handler)
        
        # Start the node
        rospy.init_node(self.node_name)
        rospy.loginfo("JumplegAgent is lissening")
        rospy.spin()

    def get_action_handler(self, req):
        rospy.loginfo("Request: %s",req)
        # Coping state recived from env
        self.CoM = req.CoM
        self.distance_x = req.distance_x
        self.distance_y = req.distance_y
        self.distance_z = req.distance_z
        resp = get_actionResponse()
        #TODO: Calculate network output
        resp.thrusting_duration = 1.0
        resp.haa_coefficient = self.haa_coefficient
        resp.hfe_coefficient = self.hfe_coefficient
        resp.kfe_coefficient = self.kfe_coefficient
        return resp
    
    def get_target_handler(self, req):
        resp = get_targetResponse()
        #TODO: Implement target control and generation based on reward or epoch
        resp.target_position =  self.target_position 
        rospy.loginfo("Target position: %s", self.target_position )
        return resp
    
    def set_reward_handler(self, req):
        resp = set_rewardResponse()
        self.last_reward = req.reward
        resp.ack = self.last_reward
        return resp

if __name__ == "__main__":
    jumplegAgent = JumplegAgent()