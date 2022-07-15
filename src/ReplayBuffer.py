import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, saved_buffer=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        if saved_buffer is not None:
            self.data = pd.read_csv(open(saved_buffer))
        else:
            self.data = pd.DataFrame(columns=['state', 'action', 'next_state', 'reward'])

    def store(self, state, action, next_state, reward):
        assert len(state) == self.state_dim,\
            f"state dimension is wrong: expected size {self.state_dim}, get size {len(state)} instead"
        assert len(action) == self.action_dim,\
            f"action dimension is wrong: expected size {self.action_dim}, get size {len(action)} instead"
        assert len(next_state) == self.state_dim,\
            f"next_state dimension is wrong: expected size {self.state_dim}, get size {len(next_state)} instead"

        self.data.loc[len(self.data.index)] = [
            np.array(state).astype(float),
            np.array(action).astype(float),
            np.array(next_state).astype(float),
            float(reward)]

    def sample(self, batch_size):
        sampled_indx = np.random.randint(0, self.data.shape[0], size=batch_size)
        sampled_data = self.data.filter(items=sampled_indx, axis=0)

        return sampled_data['state'].to_numpy(), sampled_data['action'].to_numpy(), sampled_data[
            'next_state'].to_numpy(), sampled_data['reward'].to_numpy()

    def plot_reward(self):
        plt.title("Reward over episode")
        plt.xlabel("Episode#")
        plt.ylabel("Reward")
        plt.plot(self.data['reward'], color="orange", linewidth=2.5)

    def dump(self):
        return self.data

    def dump_to_csv(self, filename="ReplayBuffer.csv"):
        print(f"#Dumping ReplayBuffer to {filename}#")
        self.data.to_csv(filename, index=False)

    def __str__(self):
        return f"ReplayBufer: {self.data.shape[0]} rows\n---------------------\n" \
               f"{self.data.head().to_string()}\n{self.data.tail().to_string()}"
