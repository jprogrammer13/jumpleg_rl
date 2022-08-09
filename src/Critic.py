import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, layer_dim=128):
        super(Critic, self).__init__()

        self.critic_model_Q1 = nn.Sequential(
            nn.Linear(state_dim+action_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, 1)
        )

        self.critic_model_Q2 = nn.Sequential(
            nn.Linear(state_dim+action_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, 1)
        )

    def forward(self, state, action):
        s_a = torch.cat([state,action],1)
        q1 = self.critic_model_Q1(s_a)
        q2 = self.critic_model_Q2(s_a)
        return q1, q2

    def Q1(self, state, action):
        s_a = torch.cat([state,action],1)
        q1 = self.critic_model_Q1(s_a)
        return q1
