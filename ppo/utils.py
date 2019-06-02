import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import numpy as np


class ReplayBuffer:
    def __init__(self, num_actor=16, gamma=0.99):
        self._n_actor = num_actor
        self._trags = [Trajectory(gamma) for i in range(num_actor)]
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

        if isinstance(action, int):
            self._trags[0].push_action(action)
        else:
            for i in range(self._n_actor):
                self._trags[i].push_action(action[i])
        return

    def update_reward(self, reward, done=None):
        if isinstance(reward, float):
            self._trags[0].push_reward(reward)
            
        else:
            for i in range(self._n_actor):
                self._trags[i].push_reward(reward[i])
        
        # print(done)
        for i in np.argwhere(done == True).flatten():
            s, l, a, r = self._trags[i].done()
            self.states += list(s)
            self.logprobs += list(l)
            self.actions += list(a)
            self.rewards += list(r)
            
            # if (len(np.argwhere(done== True).flatten())==len(done)):
            #     done[i] = False
            self._trags[i].clear()

        return

    def update_state(self, state):
        
        if state.dim() >=1:
            for i in range(self._n_actor):
                self._trags[i].push_state(state[i])
        else:
            self._trags[0].push_state(state)
        return

    def update_transition(self, state, action, reward, done):
        for i in range(self._n_actor):
            self._trags[i].push_transition(state[i], action[i], reward[i])
        for i in np.argwhere(done == True).flatten():
            s, l, a, r = self._trags[i].done()
            self.states += list(s)
            self.logprobs += list(l)
            self.actions += list(a)
            self.rewards += list(r)
            self._trags[i].clear()
            pass
        return

    def update_logprobs(self, logprob):
        if logprob.dim() > 0:
            for i in range(self._n_actor):
                self._trags[i].push_logprob(logprob[i])
        else:
            self._trags[0].push_logprob(logprob)

        return


class Trajectory:
    def __init__(self, gamma):
        self._gamma = gamma
        self._state = []
        self._action = []
        self._reward = []
        self._logprob = []
        self._acc_reward = []
        return

    def clear(self):
        del self._action[:]
        del self._state[:]
        del self._logprob[:]
        del self._reward[:]
        del self._acc_reward[:]
        return

    def done(self):
        self._acc_reward = []
        discounted_reward = 0
        for reward in reversed(self._reward):
            discounted_reward = reward + (self._gamma * discounted_reward)
            self._acc_reward.insert(0, discounted_reward)
        return self._state, self._logprob, self._action, self._acc_reward

    def push_transition(self, state, action, reward, done):
        self._state.append(state)
        self._action.append(action)
        self._reward.append(reward)
        return

    def push_state(self, state):
        self._state.append(state)
        return

    def push_logprob(self, logprob):
        self._logprob.append(logprob)
        return

    def push_action(self, action):
        self._action.append(action)
        return

    def push_reward(self, reward):
        self._reward.append(reward)
        return