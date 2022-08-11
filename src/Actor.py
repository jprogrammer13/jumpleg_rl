# import rospy
import rospy
import torch
import torch.nn as nn
import numpy as np
from utils import *

class Actor(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_time,
                 max_velocity,
                 layer_dim=256):

        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.max_time = max_time
        self.max_velocity = max_velocity

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
        action_shape = list(tmp_action.shape)[0]
        action_dim = len(tmp_action.shape)

        if action_dim > 1:
            az, _, _ = cart2sph(state[:, 3].reshape(-1, 1), state[:, 4].reshape(-1, 1), state[:, 5].reshape(-1, 1))
            az = np.full((action_shape, 1), az)

            T_th = (self.max_time - 0.5) * (torch.abs(tmp_action[:, 0]).reshape(-1, 1)) + 0.5

            el = (torch.abs(tmp_action[:, 1]).reshape(-1, 1)) * (np.pi / 2)
            r = (torch.abs(tmp_action[:, 2]).reshape(-1, 1) * 0.3) + 0.1

            ComF_x, ComF_y, ComF_z = sph2cart(az, el.detach().numpy(), r.detach().numpy())
            ComF_x = torch.Tensor(ComF_x)
            ComF_y = torch.Tensor(ComF_y)
            ComF_z = torch.Tensor(ComF_z)

            el_d = (torch.abs(tmp_action[:, 3]).reshape(-1, 1) * (np.pi / 2))
            r_d = (torch.abs(tmp_action[:, 4])).reshape(-1, 1) * self.max_velocity

            ComFd_x, ComFd_y, ComFd_z = sph2cart(az, el_d.detach().numpy(), r_d.detach().numpy())
            ComFd_x = torch.Tensor(ComFd_x)
            ComFd_y = torch.Tensor(ComFd_y)
            ComFd_z = torch.Tensor(ComFd_z)

            action = torch.reshape(torch.cat((T_th, ComF_x, ComF_y, ComF_z, ComFd_x, ComFd_y, ComFd_z), axis=1),
                                   [action_shape, self.action_dim + 2])
        else:
            az, _, _ = cart2sph(state[3], state[4], state[5])

            T_th = torch.flatten((self.max_time - 0.5) * torch.abs(tmp_action[0]) + 0.5)

            el = torch.flatten((torch.abs(tmp_action[1]) * (np.pi / 2)))
            r = torch.flatten((torch.abs(tmp_action[2]) * 0.3) + 0.1)

            ComF_x, ComF_y, ComF_z = sph2cart(az.detach().numpy(), el.detach().numpy(), r.detach().numpy())
            ComF_x = torch.Tensor(ComF_x)
            ComF_y = torch.Tensor(ComF_y)
            ComF_z = torch.Tensor(ComF_z)

            el_d = torch.flatten((torch.abs(tmp_action[3]) * (np.pi / 2)))
            r_d = torch.flatten((torch.abs(tmp_action[4]) * self.max_velocity))

            ComFd_x, ComFd_y, ComFd_z = sph2cart(az.detach().numpy(), el_d.detach().numpy(), r_d.detach().numpy())
            ComFd_x = torch.Tensor(ComFd_x)
            ComFd_y = torch.Tensor(ComFd_y)
            ComFd_z = torch.Tensor(ComFd_z)

            action = torch.cat((T_th[None, :], ComF_x[None, :],ComF_y[None, :],ComF_z[None, :], ComFd_x[None, :], ComFd_y[None, :], ComFd_z[None, :]),
                               axis=1)

        return action