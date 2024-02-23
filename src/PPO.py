import torch
import torch.nn as nn
import os

from RolloutBuffer import RolloutBuffer
from ActorCritic import ActorCritic


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(
            state_dim, action_dim, action_std_init).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(
            state_dim, action_dim, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss_mse = nn.MSELoss()

    def set_action_std(self, action_std):
        self.action_std = action_std
        self.policy.set_action_std(action_std)
        self.policy_old.set_action_std(action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):

        self.action_std -= action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        # TODO verify this formulation
        self.action_std = min_action_std if self.action_std <= min_action_std else self.action_std
        self.set_action_std(self.action_std)

    def select_action(self, state):
        # TODO: verify the eval
        self.policy_old.eval()
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    def update(self):

        # return estmate using Monte Carlo
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            # TODO: verify expression
            # TODO: verify that reward is not substituted with 0
            discounted_reward = 0 if is_terminal else reward + \
                (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Rewards normalization
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Prepare tensors
        old_states = torch.squeeze(torch.stack(
            self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(
            self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(
            self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(
            self.buffer.state_values, dim=0)).detach().to(self.device)

        # Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs

        for _ in range(self.K_epochs):

            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            state_values = torch.squeeze(state_values)

            # Ration (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Surrogate Loss
            surr_loss_1 = ratios * advantages
            surr_loss_2 = torch.clamp(
                ratios, 1-self.eps_clip, 1+self.eps_clip)*advantages

            # Clipped PPO Loss
            loss = -torch.min(surr_loss_1, surr_loss_2) + 0.5 + \
                self.loss_mse(state_values, rewards) - 0.01 * dist_entropy

            # gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Clone updated policy in old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

    # TODO: implement
    def save(self, path, name):
        torch.save(self.policy.state_dict(), os.path.join(
            path, f"PPO_{name}.pt"))

    # TODO: implement
    def load(self, path, name):
        self.critic.load_state_dict(torch.load(
            os.path.join(path, f"PPO_{name}.pt"), map_location=self.device))
