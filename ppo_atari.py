import numpy as np
import torch
import gym
from ppo.PPO import PPO, Memory, ReplayBuffer
from env_exp import SocTwoEnv
import pickle

env = gym.make('Breakout-ram-v0')
env.render()

############## Hyperparameters Striker ##############
state_dim_striker = 128
action_dim_striker = 3
n_latent_var_striker = 64  # number of variables in hidden layer
#############################################

max_episodes = 50000  # max training episodes
log_interval = 10  # print avg reward in the interval
update_episode = 10  # update policy every n timesteps 2000
lr = 0.001
gamma = 0.99  # discount factor
K_epochs = 4  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
random_seed = None

if random_seed:
    torch.manual_seed(random_seed)
    env.seed(random_seed)

# memory_striker = Memory()
memory_striker = ReplayBuffer(1, gamma)
ppo_striker = PPO(state_dim_striker, action_dim_striker, n_latent_var_striker,
                  lr, gamma, K_epochs, eps_clip)

# logging variables
running_reward = 0
avg_length = 0
timestep = 0
reward_history = []

# training loop
i_episode = 1
while i_episode < (max_episodes + 1):
    state_striker = env.reset()
    # for t in range(max_timesteps):
    while True:
        timestep += 1

        action_striker = ppo_striker.policy_old.act(state_striker,
                                                    memory_striker)
        states, reward, done, _ = env.step(action_striker)
        env.render()
        memory_striker.update_reward(reward)

        running_reward += reward
        if done:
            # print(i_episode)
            break
        # if np.argwhere(done):
        #     i_episode += len(np.argwhere(done))
        #     print(i_episode)
        #     break
    i_episode += 1
    if i_episode + 1 % update_episode == 0:
        ppo_striker.update(memory_striker)
        memory_striker.clear_memory()

    avg_length += timestep
    timestep = 0
    # logging
    if i_episode % log_interval == 0:
        avg_length = int(avg_length / log_interval)
        running_reward = int((running_reward / log_interval))
        reward_history.append(running_reward)
        print('Episode {} \t avg length: {} \t reward: {}'.format(
            i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0
pickle.dumps(reward_history, open('reward_atari.pkl', 'wb'))