import numpy as np
import torch
import gym
from ppo.PPO import PPO, Memory
from ppo.utils import ReplayBuffer
from env_exp import SocTwoEnv
import pickle
import os
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
except:
    pass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make('Breakout-ram-v0')
# env.render()

############## Hyperparameters Striker ##############
state_dim = 128
action_dim = 4
n_latent_var = 64  # number of variables in hidden layer
#############################################

max_episodes = 50000    # max training episodes
log_interval = 50      # print avg reward in the interval
update_episode = 20     # update policy every n timesteps 2000
lr = 2e-3
gamma = 0.99    # discount factor
K_epochs = 4    # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO

# memory_striker = Memory()
memory_striker = ReplayBuffer(1, gamma)
ppo_agent = PPO(state_dim, action_dim, n_latent_var, lr, gamma, K_epochs,
                eps_clip, device)

# logging variables
running_reward = 0
avg_length = 0
timestep = 0
reward_history = []

# training loop
i_episode = 1
while i_episode < (max_episodes + 1):
    state_striker = env.reset()
    while True:
        timestep += 1
        action_striker = ppo_agent.policy_old.act(state_striker,
                                                  memory_striker)
        states, reward, done, _ = env.step(action_striker)
        # env.render()
        memory_striker.update_reward(reward, done)

        running_reward += reward
        if done:
            break

    i_episode += 1
    if (i_episode) % update_episode == 0:
        ppo_agent.update(memory_striker)
        memory_striker.clear_memory()

    avg_length += timestep
    timestep = 0

    # logging
    if i_episode % log_interval == 0:
        avg_length = avg_length / log_interval
        running_reward = (running_reward / log_interval)
        reward_history.append(running_reward)
        print('Episode {} \t avg length: {} \t reward: {}'.format(
            i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0
        pickle.dump(reward_history, open('reward_atari.pkl', 'wb'))