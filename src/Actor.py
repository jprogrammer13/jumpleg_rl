# import rospy
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_time,
                 max_extension,
                 max_acceleration,
                 layer_dim=256):
        
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.max_acceleration = max_acceleration
        self.max_extension = max_extension
        self.max_time = max_time

        self.actor_model = nn.Sequential(
            nn.Linear(state_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, layer_dim),
            nn.ReLU(),
            nn.Linear(layer_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        tmp_action = torch.abs(self.actor_model(state))
        action_shape = list(tmp_action.shape)[0]
        action_dim = len(tmp_action.shape)
        if action_dim > 1:
            T_th = self.max_time * tmp_action[:,0].reshape(-1,1)
            k = tmp_action[:,1].reshape(-1,1)
            ComF_xy = state[:,3:5] * torch.cat((k,k),axis=1)
            ComF_z =  self.max_extension * tmp_action[:,2].reshape(-1,1)
            ComFd = self.max_acceleration * tmp_action[:,3:]
            action = torch.reshape(torch.cat((T_th,ComF_xy,ComF_z,ComFd),axis=1), [action_shape,self.action_dim+1])
        else:
            print(tmp_action)
            T_th = torch.flatten(torch.abs(self.max_time * tmp_action[0]))
            k = tmp_action[1]
            ComF_xy = torch.flatten(state[3:5] * torch.Tensor([k,k]))
            ComF_z = torch.flatten(self.max_extension * tmp_action[2])
            ComFd = torch.flatten(self.max_acceleration * tmp_action[3:])
            action = torch.cat((T_th[None,:], ComF_xy[None,:], ComF_z[None,:], ComFd[None,:]), axis=1)

        return action
