import numpy as np
import rospy
import torch
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import copy

from Actor import Actor
from Critic import Critic
from ReplayBuffer import ReplayBuffer


class TD3(object):
    def __init__(
        self,
        log_writer,
        state_dim,
        action_dim,
        layer_dim,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.log_writer = log_writer
        self.lr = 3e-4
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading TD3 on {self.device}")
        self.actor = Actor(state_dim, action_dim, layer_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr)

        self.critic = Critic(state_dim, action_dim, layer_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        self.actor.eval()
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        state, action, next_state, reward = replay_buffer.sample(batch_size)
        self.actor.train()
        with torch.no_grad():
            noise = (torch.randn_like(action) *
                     self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            # This becouse we want to have an action {-1,1} to scale after
            next_action = (self.actor_target(next_state)+noise).clamp(-1, 1)

            target_Q = reward  # + not_done * self.discount * target_Q

        # Get the current Q estimates
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.log_writer.add_scalar("Critic loss", critic_loss, self.total_it)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates

        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = - self.critic(state, self.actor(state)).mean()
            rospy.loginfo("Updating the networks")
            self.log_writer.add_scalar("Actor loss", actor_loss, self.total_it)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path, name):
        torch.save(self.critic.state_dict(), os.path.join(
            path, f"TD3_{name}_critic.pt"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(
            path, f"TD3_{name}_critic_optimizer.pt"))

        torch.save(self.actor.state_dict(), os.path.join(
            path, f"TD3_{name}_actor.pt"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(
            path, f"TD3_{name}_actor_optimizer.pt"))

    def load(self, path, name, iteration):
        self.total_it = iteration
        
        self.critic.load_state_dict(torch.load(
            os.path.join(path, f"TD3_{name}_critic.pt"), map_location=self.device))
        self.critic_optimizer.load_state_dict(
            torch.load(os.path.join(path, f"TD3_{name}_critic_optimizer.pt"), map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(
            os.path.join(path, f"TD3_{name}_actor.pt"), map_location=self.device))
        self.actor_optimizer.load_state_dict(
            torch.load(os.path.join(path, f"TD3_{name}_actor_optimizer.pt"), map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)
