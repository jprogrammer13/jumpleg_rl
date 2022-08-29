from turtle import forward
import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, layer_dim = 256):
        super(Critic, self).__init__()

        # Q1 critic    
        self.q1_fc1 = nn.Linear(state_dim+action_dim, layer_dim)
        self.q1_fc2 = nn.Linear(layer_dim,layer_dim)
        self.q1_fc3 = nn.Linear(layer_dim, 1)

        # Q2 critic
        # self.q2_fc1 = nn.Linear(state_dim+action_dim, layer_dim)
        # self.q2_fc2 = nn.Linear(layer_dim,layer_dim)
        # self.q2_fc3 = nn.Linear(layer_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        sa_cat = torch.cat([state, action], 1)

        q1 = self.relu(self.q1_fc1(sa_cat))
        q1 = self.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)
        #
        # q2 = self.relu(self.q2_fc1(sa_cat))
        # q2 = self.relu(self.q2_fc2(q2))
        # q2 = self.q2_fc3(q2)

        return q1#, q2
    
    # def Q1(self, state, action):
    #
    #     sa_cat = torch.cat([state, action], 1)
    #
    #     q1 = self.relu(self.q1_fc1(sa_cat))
    #     q1 = self.relu(self.q1_fc2(q1))
    #     q1 = self.q1_fc3(q1)
    #
    #     return q1