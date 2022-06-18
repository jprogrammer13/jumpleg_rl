#!/usr/bin/env python3

from jumpleg_rl.srv import get_action, get_actionResponse
import rospy
import numpy as np

class JumplegAgent:
    def __init__(self):
        rospy.init_node("JumplegAgent")
        service = rospy.Service("get_action", get_action, self.action_handler)
        rospy.loginfo("JumplegAgent is lissening")
        rospy.spin()

    def action_handler(self, req):
        rospy.loginfo("Request: %s",req)
        resp = get_actionResponse()
        resp.Tf = 1.0
        resp.haa_coefficient =  np.random.rand(6)
        resp.hfe_coefficient =  np.random.rand(6)
        resp.kfe_coefficient =  np.random.rand(6)
        return resp

if __name__ == "__main__":
    jumplegAgent = JumplegAgent()