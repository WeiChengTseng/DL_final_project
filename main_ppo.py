import numpy as np
import torch
import gym
from ppo.PPO import PPO
from ppo.utils import ReplayBuffer
from env_exp import SocTwoEnv
import random
import tensorboardX
import os

env_path = './env/linux/soccer_test.x86_64'
env = SocTwoEnv(env_path, worker_id=0, train_mode=True)

############## Hyperparameters Striker ##############
state_dim_striker = 112
action_dim_striker = 7
n_latent_var_striker = 64  # number of variables in hidden layer
#############################################

############## Hyperparameters Goalie ##############
state_dim_goalie = 112
action_dim_goalie = 5
n_latent_var_goalie = 64  # number of variables in hidden layer
#############################################

max_episodes = 50000  # max training episodes
update_episode = 400  # max timesteps in one episode
log_interval = 100  # print avg reward in the interval
interval = 100

update_timestep = 200  # update policy every n timesteps 2000
lr = 0.001
gamma = 0.99  # discount factor
K_epochs = 4  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
random_seed = None

if random_seed:
    torch.manual_seed(random_seed)
    env.seed(random_seed)

# memory_striker = Memory()
memory_striker = ReplayBuffer(16, gamma)
ppo_striker = PPO(state_dim_striker, action_dim_striker, n_latent_var_striker,
                  lr, gamma, K_epochs, eps_clip)

# memory_goalie = Memory()
memory_goalie = ReplayBuffer(16, gamma)
ppo_goalie = PPO(state_dim_goalie, action_dim_goalie, n_latent_var_goalie, lr,
                 gamma, K_epochs, eps_clip)

# logging variables
running_reward_striker = 0
running_reward_goalie = 0

avg_length_goalie = 0
avg_length_striker = 0

timestep_striker = np.zeros(16, dtype=int)
timestep_goalie = np.zeros(16, dtype=int)

i_old = 0
i_episode = 0
count = 1

writer = tensorboardX.SummaryWriter()


# training loop
state_striker, state_goalie = env.reset()
while i_episode < (max_episodes):
    while True:
        timestep_striker += 1
        timestep_goalie += 1

        # Running policy_old:
        action_striker = ppo_striker.policy_old.act(state_striker,
                                                    memory_striker)
        action_goalie = ppo_goalie.policy_old.act(state_goalie, memory_goalie)
        states, reward, done, _ = env.step(action_striker, action_goalie)
        
        # Saving reward:
        memory_striker.update_reward(reward[0], done[0])
        memory_goalie.update_reward(reward[1], done[1])

        running_reward_striker += reward[0]
        running_reward_goalie += reward[1]

        if (len(np.argwhere(done).flatten()) != 0):
            if ((i_episode + len(np.argwhere(done).flatten())) > log_interval):
                i_episode = log_interval
                
            else:
                i_episode += len(np.argwhere(done).flatten())

            break

    if (i_episode) % update_episode == 0:
        ppo_striker.update(memory_striker)
        ppo_goalie.update(memory_goalie)
        memory_striker.clear_memory()
        memory_goalie.clear_memory()

    avg_length_goalie += timestep_goalie
    avg_length_striker += timestep_striker

    timestep_goalie = 0
    timestep_striker = 0

    # logging
    if ((i_episode % log_interval) == 0):
        avg_length_goalie = np.sum(avg_length_goalie) / (
            i_episode - i_old) / 16
        avg_length_striker = np.sum(avg_length_striker) / (
            i_episode - i_old) / 16
        avg_running_reward_striker = np.sum(running_reward_striker) / (
            i_episode - i_old) / 16
        avg_running_reward_goalie = np.sum(running_reward_goalie) / (
            i_episode - i_old) / 16

        print('Episode {} \t avg striker length: {} \t reward: {}'.format(
            i_episode, avg_length_striker, avg_running_reward_striker))

        
        print('Episode {} \t avg goalie length: {} \t reward: {}'.format(
            i_episode, avg_length_goalie, avg_running_reward_goalie))

        # writer.add_scalars('reward',{'striker': avg_running_reward_striker},epo)
        

        running_reward_striker = 0
        running_reward_goalie = 0

        avg_length_goalie = 0
        avg_length_striker = 0

        torch.save(ppo_striker.policy.state_dict(),
                   './PPO_striker{}_{}.pth'.format('SoccerTwos',i_episode))
        torch.save(ppo_goalie.policy.state_dict(),
                   './PPO_goalie{}_{}.pth'.format('SoccerTwos',i_episode))

        log_interval += interval
        i_old = i_episode