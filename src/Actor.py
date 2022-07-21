import rospy
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_time_value,
                 max_action_value,
                 layer_dim=256):
        
        super(Actor, self).__init__()
        self.max_action_value = max_action_value
        self.max_time_value = max_time_value

        self.actor_model = nn.Sequential(
            nn.Linear(state_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        action = self.actor_model(state)
        # return self.max_action_value * action
        T_th = torch.abs(self.max_time_value * action[:,0])
        coef = torch.flatten(self.max_action_value * action[:,1:])
        action = torch.cat((T_th[None,:],coef[None,:]),axis=1)
        return action
        
