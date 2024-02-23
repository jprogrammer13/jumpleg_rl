from turtle import forward
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, layer_dim=128):
        super(ActorCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_var = torch.full(
            (action_dim,), action_std_init * action_std_init)

        self.actor = nn.Sequential(
            nn.Linear(state_dim, layer_dim),
            nn.Tanh(),
            nn.Linear(layer_dim, layer_dim),
            nn.Tanh(),
            nn.Linear(layer_dim, action_dim),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, layer_dim),
            nn.Tanh(),
            nn.Linear(layer_dim, layer_dim),
            nn.Tanh(),
            nn.Linear(layer_dim, 1)
        )

    def set_action_std(self, new_action_std):

        self.action_var = torch.full(
            (self.action_dim,), new_action_std * new_action_std)
        

    def forward(self):
        pass

    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action, action_logprob, state_val
    
    def evaluate(self, state, action):

        action_mean = self.actor(state)
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy
