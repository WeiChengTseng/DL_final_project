import numpy as np
from torch import Tensor
from torch.autograd import Variable


class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """

    def __init__(self,
                 max_steps,
                 num_agents,
                 obs_dims,
                 ac_dims):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        obs_dims = obs_dims
        ac_dims = ac_dims

        # self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim),
                                           dtype=np.float32))
            self.ac_buffs.append(np.zeros((max_steps, adim), dtype=np.float32))
            self.rew_buffs.append(np.zeros(max_steps, dtype=np.float32))
            self.next_obs_buffs.append(
                np.zeros((max_steps, odim), dtype=np.float32))
            self.done_buffs.append(np.zeros(max_steps, dtype=np.uint8))

        # index of first empty location in buffer (last index when full)
        self.filled_i = 0
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

        return

    def __len__(self):
        return self.filled_i

    def push(self,
             observations,
             actions,
             rewards,
             next_observations,
             dones,
             accumulate=True,
             gamma=0.95):

        # nentries = observations.shape[1] 
        nentries = 8

        if self.curr_i + nentries > self.max_steps:
            # num of indices to roll over
            rollover = self.max_steps - self.curr_i
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover,
                                                  axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover,
                                                 axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps

        for agent_i in range(self.num_agents):
            # print(observations[agent_i].shape)
            self.obs_buffs[agent_i][self.curr_i:self.curr_i +
                                    nentries] = observations[agent_i]
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i +
                                   nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i +
                                    nentries] = rewards[agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i +
                                         nentries] = next_observations[agent_i]
            self.done_buffs[agent_i][self.curr_i:self.curr_i +
                                     nentries] = dones[agent_i]

        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

        if accumulate:
            done_thread = np.argwhere(dones[0]).flatten()
            if len(done_thread) > 0:
                # print(done_thread)
                pass
            for thread in done_thread:
                accum_rwd, thd = np.zeros(self.num_agents), 16 - thread
                tmp = self.curr_i - thd - nentries
                accum_rwd += ([
                    self.rew_buffs[i][self.curr_i - thd]
                    for i in range(self.num_agents)
                ])

                while True:
                    # print(tmp)
                    # print(accum_rwd)
                    accum_rwd = np.array([
                        self.rew_buffs[i][tmp] for i in range(self.num_agents)
                    ]) + accum_rwd * gamma
                    for agent_i in range(self.num_agents):
                        self.rew_buffs[agent_i][tmp] = accum_rwd[agent_i]
                    tmp -= nentries
                    if self.done_buffs[0][tmp]:
                        break
                    if tmp < 0 and self.filled_i == self.max_steps:
                        tmp += self.max_steps
                # print(tmp)
                # print(nentries)
                # print(self.rew_buffs[0][tmp + np.arange(150) * nentries])
        return

    def sample(self, N, device='cpu', norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=True)
        # if to_gpu:
        #     cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        # else:
        cast = lambda x: Variable(Tensor(x), requires_grad=False).to(device)
        if norm_rews:
            ret_rews = [
                cast((self.rew_buffs[i][inds] -
                      self.rew_buffs[i][:self.filled_i].mean()) /
                     (self.rew_buffs[i][:self.filled_i].std() + 1e-7))
                for i in range(self.num_agents)
            ]
        else:
            ret_rews = [
                cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)
            ]

        obs_buffs = [
            cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)
        ]
        ac_buffs = [
            cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)
        ]
        next_obs_buffs = [
            cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)
        ]
        done_buffs = [
            cast(self.done_buffs[i][inds]) for i in range(self.num_agents)
        ]

        return (obs_buffs, ac_buffs, ret_rews, next_obs_buffs, done_buffs)

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N,
                             self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]
