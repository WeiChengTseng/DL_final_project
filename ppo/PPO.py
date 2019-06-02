import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np




class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        return

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        return

    def update_action(self, action):
        # print(list(action))
        # print(action)

        self.actions += list(action)
        return

    def update_reward(self, reward):
        if isinstance(reward, float):
            self.actions.append(reward)
        else:
            self.rewards += list(reward)
        return

    def update_state(self, state):
        # print(list(state))
        self.states += list(state)
        return

    def update_logprobs(self, logprob):
        self.logprobs += list(logprob)
        return

    def sample(self, batch_size):
        pass


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

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(self.device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        if state.dim() > 1:
            memory.update_state(state)
            memory.update_action(action)
            memory.update_logprobs(dist.log_prob(action))
            return action.numpy()
        else:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
            return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, gamma,
                 K_epochs, eps_clip, device='cpu'):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.policy = ActorCritic(state_dim, action_dim,
                                  n_latent_var, device).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim,
                                      n_latent_var, device).to(self.device)

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        # rewards = []
        # discounted_reward = 0
        # for reward in reversed(memory.rewards):
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)
        rewards = memory.rewards
        # print(rewards)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
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
