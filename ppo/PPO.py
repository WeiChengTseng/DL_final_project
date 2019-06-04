import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np
import random


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, device):
        super(ActorCritic, self).__init__()
        self.device = device

        self.affine = nn.Linear(state_dim, n_latent_var)

        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var), nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var), nn.Tanh(),
            nn.Linear(n_latent_var, action_dim), nn.Softmax(dim=-1))

        # critic
        self.value_layer = nn.Sequential(nn.Linear(state_dim, n_latent_var),
                                         nn.Tanh(),
                                         nn.Linear(n_latent_var, n_latent_var),
                                         nn.Tanh(), nn.Linear(n_latent_var, 1))

    def forward(self):
        raise NotImplementedError

    def act_test(self, state, action_dim):

        model_state = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            model_action = self.action_layer(model_state[:8])
            index = torch.argmax(model_action, 1)

            model_index = index.detach().numpy()

            random_action = np.random.randint(0, action_dim,size=8)

            actions = np.concatenate((model_index, random_action),axis=None)
        return actions

    def act(self, state, memory=None):
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        if memory:
            if state.dim() >= 1:
                memory.update_state(state)
                memory.update_action(action)
                memory.update_logprobs(dist.log_prob(action))
                return action.numpy()
            else:
                memory.states.append(state)
                memory.actions.append(action)
                memory.logprobs.append(dist.log_prob(action))
                return action.item()

        return action

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 n_latent_var=64,
                 lr=1e-3,
                 gamma=0.99,
                 K_epochs=4,
                 eps_clip=0.2,
                 device='cpu',
                 ckpt_path=None):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var,
                                  device).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var,
                                      device).to(self.device)

        if ckpt_path:
            self.policy.load_state_dict(
                torch.load(ckpt_path, map_location=device))
            self.policy_old.load_state_dict(
                torch.load(ckpt_path, map_location=device))
            print('ppo restore')

        self.MseLoss = nn.MSELoss()

    def act(self, states):


        return self.policy.act(states.numpy()).view(8,1)

    def update(self, memory):
        rewards = memory.rewards

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        # print(memory.logprobs)
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(
                state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
