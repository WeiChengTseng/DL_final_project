import numpy as np
import torch
import gym
from ppo.PPO import PPO
from ppo.utils import ReplayBuffer
from tensorboardX import SummaryWriter
import pickle
import os


device = torch.device("cpu")
env = gym.make('Breakout-ram-v0')
env.render()

############## Hyperparameters Striker ##############
state_dim = 128
action_dim = 4
n_latent_var = 64  # number of variables in hidden layer
#############################################

max_episodes = 50000    # max training episodes
log_interval = 100      # print avg reward in the interval
update_episode = 20    # update policy every n timesteps 2000
lr = 2e-3
gamma = 0.99    # discount factor
K_epochs = 8    # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO

# memory_striker = Memory()
memory_striker = ReplayBuffer(1, gamma)
ppo_agent = PPO(state_dim, action_dim, n_latent_var, lr, gamma, K_epochs,
                eps_clip, device)
name = "epoch_update_{}".format(K_epochs)
writer = SummaryWriter('./ppo/logs/'+name)

f = open("./ppo/logs/" + name +"/hyper-param.txt", "a")
print('state_dim = ',state_dim,file=f)
print('action_dim = ',action_dim,file=f)
print("n_latent_var = ", n_latent_var,file=f)
print("max_episode = ", max_episodes,file=f)
print('log_interval = ',log_interval,file=f)
print('update_episode = ', update_episode,file=f)
print('learning_rate = ', lr,file=f)
print('discount =', gamma,file=f)
print("every {} epoch update off-policy".format(K_epochs),file=f)
print("weight clipping = ", eps_clip,file=f)
f.close()


# logging variables
running_reward = 0
avg_length = 0
timestep = 0

# training loop
i_episode = 1
while i_episode < (max_episodes + 1):
    state_striker = env.reset()
    while True:
        timestep += 1
        action_striker = ppo_agent.policy_old.act(state_striker,
                                                  memory_striker)
        states, reward, done, _ = env.step(action_striker)
        env.render()
        memory_striker.update_reward(reward, done)

        running_reward += reward
        if done:
            break

    i_episode += 1
    if ((i_episode) % update_episode == 0):
        ppo_agent.update(memory_striker)
        memory_striker.clear_memory()

    avg_length += timestep
    timestep = 0

    # logging
    if i_episode % log_interval == 0:
        avg_length = avg_length / log_interval
        running_reward = (running_reward / log_interval)
        writer.add_scalar('average_reward', running_reward, i_episode)
        print('Episode {} \t avg length: {} \t reward: {}'.format(
            i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0