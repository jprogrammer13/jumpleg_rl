# import rospy
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
        tmp_action = self.actor_model(state)
        action_dim = len(tmp_action.shape)
        if action_dim > 1:
            T_th = torch.abs(self.max_time_value * tmp_action[:,0]).reshape(-1,1)
            coef = self.max_action_value * tmp_action[:,1:]
            action = torch.cat((T_th,coef),axis=1)
        else:
            T_th = torch.flatten(torch.abs(self.max_time_value * tmp_action[0]))
            coef = torch.flatten(self.max_action_value * tmp_action[1:])
            action = torch.cat((T_th[None,:],coef[None,:]),axis=1)
        return action