#!/usr/bin/env python3

from jumpleg_rl.srv import *
import rospy
import numpy as np
import os
import argparse

from ReplayBuffer import ReplayBuffer
from TD3 import TD3

class JumplegAgent:
    def __init__(self, _mode):

        # Class attribute
        self.node_name = "JumplegAgent"
        self.mode = _mode

        # Service proxy
        self.get_action_srv = rospy.Service(os.path.join(self.node_name, "get_action"), get_action, self.get_action_handler)
        self.get_target_srv = rospy.Service(os.path.join(self.node_name, "get_target"), get_target, self.get_target_handler)
        self.set_reward_srv = rospy.Service(os.path.join(self.node_name, "set_reward"), set_reward, self.set_reward_handler)

        # RL
        self.state_dim = 6
        self.action_dim = 19
        self.max_time = 5
        self.max_action = 500
        self.replayBuffer = ReplayBuffer(self.state_dim,self.action_dim)
        self.policy = TD3(self.state_dim,self.action_dim,self.max_time,self.max_action)

        self.max_episode = 1000
        self.episode_counter = 0
        self.episode_transition = {"state":None, "action": None, "nxt_state": None, "reward": None}
        self.CoM0 = [0,0,0.25]
        self.targetCoM = self.generate_target(self.CoM0)
        self.batch_size = 32


        if self.mode == 'inferencce':
            #TODO: load model
            pass
        
        # Start the node
        rospy.init_node(self.node_name)
        rospy.loginfo(f"JumplegAgent is lissening: {self.mode}")
        rospy.spin()

    def generate_target(self, CoM):
        rho = np.random.uniform(0, 2 * np.pi)
        # z = np.random.uniform(CoM[2],CoM[2]*3)
        z = CoM[2]
        r = np.random.uniform(0.5, CoM[2] * 4)
        x = r * np.cos(rho)
        y = r * np.sin(rho)

        # TODO: Remove the test
        x = 1.5
        y = 0
        z = 0.25
        return [x, y, z]

    def get_target_handler(self, req):
        resp = get_targetResponse()
        if self.episode_counter > self.max_episode:
            episode_counter = 0
            self.generate_target(self.CoM0)
        resp.target_CoM = self.targetCoM
        return resp

    def get_action_handler(self, req):
        state = np.array(req.state)
        self.episode_transition['state'] = state

        action = self.policy.get_action(state)
        self.episode_transition['action'] = action

        resp = get_actionResponse()
        resp.action = action
        return resp
    
    def set_reward_handler(self, req):
        next_state = np.array(req.next_state)
        self.episode_transition['nxt_state'] = next_state

        reward = req.reward
        self.episode_transition['reward'] = reward

        rospy.loginfo(f"Episode {self.episode_counter}: {self.episode_transition['reward']}")
        self.replayBuffer.store(self.episode_transition['state'],
                                self.episode_transition['action'],
                                self.episode_transition['nxt_state'],
                                self.episode_transition['reward'])

        # self.replayBuffer.plot_reward()
        # reset the partial transition
        self.episode_transition = {"state":None, "action": None, "nxt_state": None, "reward": None}

        # Train

        if self.episode_counter > self.batch_size:
            rospy.loginfo(f"TRAINING: {self.episode_counter}")
            if ((self.episode_counter + 1) % 100) == 0:
                rospy.loginfo("SAVING")
                self.policy.save('TD3.pt', '/home/riccardo/')
                self.replayBuffer.dump_to_csv('/home/riccardo/ReplayBuffer.csv')
            self.policy.train(self.replayBuffer,32)

        resp = set_rewardResponse()
        resp.ack = reward
        print(f"Reward: {reward}")
        self.episode_counter += 1
        return resp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JumplegAgent arguments')
    parser.add_argument('--mode', type=str,default="inference",nargs="?",help='Agent mode')
    args = parser.parse_args(rospy.myargv()[1:])

    jumplegAgent = JumplegAgent(args.mode)