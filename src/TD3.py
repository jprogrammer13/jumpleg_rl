import imp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import copy

from Actor import Actor
from Critic import Critic
from ReplayBuffer import ReplayBuffer


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        layer_dim,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.writer = SummaryWriter('/home/riccardo/TD3')
        self.lr = 1e-4
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
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
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        state, action, next_state, reward = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) *
                     self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            # This becouse we want to have an action {-1,1} to scale after
            next_action = (self.actor_target(next_state)+noise).clamp(-1, 1)

            # Compute target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            # Our system end in 1 step, but we can consider the jump action repetable from the landing point
            not_done = torch.FloatTensor(np.ones((batch_size,1))).to(self.device)
            target_Q = reward + not_done * self.discount * target_Q

        # Get the current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)

        self.writer.add_scalar("Critic loss",critic_loss,self.total_it)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates

        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = - self.critic.Q1(state, self.actor(state)).mean()
            self.writer.add_scalar("Actor loss", actor_loss, self.total_it)

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

    def save(self, filename="/home/TD3"):
        torch.save(self.critic.state_dict(), filename+"_critic.pt")
        torch.save(self.critic_optimizer.state_dict(),
                   filename+"_critic_optimizer.pt")

        torch.save(self.actor.state_dict(), filename+"_actor.pt")
        torch.save(self.actor_optimizer.state_dict(),
                   filename+"_actor_optimizer.pt")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic.pt"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer.pt"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor.pt"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer.pt"))
        self.actor_target = copy.deepcopy(self.actor)
