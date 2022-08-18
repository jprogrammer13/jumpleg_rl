import rospy

import ReplayBuffer
from Actor import Actor
from Critic import Critic

import torch
import torch.nn.functional as F
import numpy as np

class TD3(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_time,
                 min_time,
                 max_velocity,
                 max_extension,
                 min_extension,
                 min_phi,
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_time = max_time
        self.min_time = min_time
        self.max_velocity = max_velocity
        self.max_extension = max_extension
        self.min_extension = min_extension
        self.min_phi = min_phi
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor network
        self.actor = Actor(self.state_dim,self.action_dim-2,self.max_time,self.min_time, self.max_velocity, self.max_extension, self.min_extension, self.min_phi, 128).to(self.device)
        self.actor_target =  Actor(self.state_dim,self.action_dim-2,self.max_time,self.min_time, self.max_velocity, self.max_extension, self.min_extension, self.min_phi, 128).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        # Critic network
        self.critic = Critic(self.state_dim, self.action_dim, 128).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim, 128).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

    def get_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        # action = (action + np.random.normal(0, noise, size=self.action_dim))
        return action

    def train(self, replay_buffer, iteration = 50,  batch_size=32):
        loss_values = []
        for it in range(iteration):
            # get batch and convert to tensors
            state, action, next_state, reward = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)

            # TODO: This is implemented but not used, our state is always d=1
            # ---------------------------------------------------------------
            done = torch.FloatTensor(1 - 1).to(self.device)

            with torch.no_grad():
                noise = torch.FloatTensor(action).data.normal_(0,self.policy_noise).to(self.device)
                # rospy.loginfo(f"Noise: {noise}")
                #TODO: Fix noise clamp

                # noise = noise.clamp(100,100)
                next_action = (self.actor_target(next_state)+noise)
                # next_action = (self.actor_target(next_state))

                # We have only one step env -> we compare the current critic with only the real reward
                # target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                # target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward.reshape(-1,1) #+ (self.discount * done * target_Q).detach()
            # ---------------------------------------------------------------

            current_Q1, current_Q2 = self.critic(state,action)

            critic_loss = F.mse_loss(current_Q1,target_Q)+F.mse_loss(current_Q2,target_Q)

            loss_values.append(critic_loss.item())


            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if it % self.policy_freq == 0:
                rospy.loginfo("updating the net")
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
                    target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

                for param, target_param in zip(self.actor.parameters(),self.actor.parameters()):
                    target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

        rospy.loginfo(f"Loss: {sum(loss_values)/len(loss_values)}")

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))