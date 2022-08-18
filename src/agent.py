#!/usr/bin/env python3

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
        # Reward publisher

        #TODO fix this
        #self.reward_pub = rospy.Publisher(os.path.join(self.node_name, "reward"), Float32, queue_size=10)

        # RL
        self.state_dim = 6
        self.action_dim = 7
        self.max_time = 1.
        self.min_time  = 0.1
        self.max_velocity = 1.5
        self.max_extension = 0.32
        self.min_extension = 0.12
        self.min_phi = np.pi/6.

        self.replayBuffer = ReplayBuffer(self.state_dim, self.action_dim)
        self.policy = TD3(self.state_dim, self.action_dim, self.max_time, self.min_time, self.max_velocity, self.max_extension, self.min_extension, self.min_phi)

        self.max_episode = 10000
        self.episode_counter = 0
        self.episode_transition = {"state": None, "action": None, "nxt_state": None, "reward": None}
        self.CoM0 = np.array([-0.01303,  0.00229,  0.25252])
        self.targetCoM = self.generate_target(self.CoM0)
        self.batch_size = 32

        # plot
        self.x = []
        self.y = []

        plt.ion()

        self.figure = plt.figure(figsize=(15, 10))

        if self.mode == 'inferencce':
            # TODO: load model
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

        # # TODO: Remove the test
        x = 0.51303
        y = 0.0229
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

        if self.episode_counter > (self.batch_size):
            action = self.policy.get_action(state)
        else:
            theta, _, _ = cart2sph(state[3], state[4], state[5])

            tmp_action = np.random.uniform(-1,1,self.action_dim-2)

            T_th = (self.max_time- self.min_time) * 0.5*(tmp_action[0]+1) + self.min_time

            phi = (np.pi/2 - self.min_phi) * ( 0.5*(tmp_action[1]+1) ) + self.min_phi
            r = (self.max_extension - self.min_extension) * 0.5*(tmp_action[0]+1) + self.min_extension
            ComF_x, ComF_y, ComF_z = sph2cart(theta, phi, r)

            phi_d =  ( 0.5*(tmp_action[3]+1) )*(np.pi/2)
            r_d = ( 0.5*(tmp_action[4]+1) ) * self.max_velocity

            ComFd_x, ComFd_y, ComFd_z = sph2cart(theta, phi_d, r_d)

            action = np.concatenate(([T_th], [ComF_x], [ComF_y], [ComF_z], [ComFd_x], [ComFd_y], [ComFd_z] ))

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
        self.episode_transition = {"state": None, "action": None, "nxt_state": None, "reward": None}

        # Train

        if self.episode_counter > (self.batch_size):
            rospy.loginfo(f"TRAINING: {self.episode_counter}")
            if ((self.episode_counter + 1) % 100) == 0:
                rospy.loginfo("SAVING")
                self.policy.save('TD3.pt', '/home/riccardo/')
                self.replayBuffer.dump_to_csv('/home/riccardo/ReplayBuffer.csv')
            self.policy.train(self.replayBuffer, self.batch_size)

        resp = set_rewardResponse()
        resp.ack = reward
        print(f"Reward: {reward}")
        #TODO fix this
        #self.reward_pub(reward)
        self.episode_counter += 1
        # TODO fix this
        # self.x = np.arange(0,self.episode_counter,1)
        # self.y.append(reward)
        # plt.cla()
        #
        # plt.title("Jumpleg RL reward")
        # plt.xlabel("Epoch")
        # plt.ylabel("Reward")
        #
        # plt.plot(self.x, self.y, color="orange")
        # self.figure.canvas.draw()
        # self.figure.canvas.flush_events()
        return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JumplegAgent arguments')
    parser.add_argument('--mode', type=str, default="inference", nargs="?", help='Agent mode')
    args = parser.parse_args(rospy.myargv()[1:])

    jumplegAgent = JumplegAgent(args.mode)