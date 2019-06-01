import numpy as np
import torch
import gym
from ppo.PPO import PPO
from ppo.utils import ReplayBuffer
from env_exp import SocTwoEnv

env_path = './env/macos/SoccerTwosFast.app'
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
max_timesteps = 300  # max timesteps in one episode
log_interval = 100  # print avg reward in the interval
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
running_reward = 0
avg_length = 0
timestep = 0

# training loop
state_striker, state_goalie = env.reset()
for i_episode in range(1, max_episodes + 1):
    for t in range(max_timesteps):
        timestep += 1

        # Running policy_old:
        action_striker = ppo_striker.policy_old.act(state_striker,
                                                    memory_striker)
        action_goalie = ppo_goalie.policy_old.act(state_goalie, memory_goalie)
        states, reward, done, _ = env.step(action_striker, action_goalie)
        print(np.argwhere(done))
        # Saving reward:
        memory_striker.update_reward(reward[0])
        memory_goalie.update_reward(reward[1])

        # update if its time
        if timestep % update_timestep == 0:
            ppo_striker.update(memory_striker)
            memory_striker.clear_memory()

            ppo_goalie.update(memory_goalie)
            memory_goalie.clear_memory()

            timestep = 0

        running_reward += max(reward[0])

    avg_length += t

    # logging
    if i_episode % log_interval == 0:
        avg_length = int(avg_length / log_interval)
        running_reward = int((running_reward / log_interval))

        print('Episode {} \t avg length: {} \t reward: {}'.format(
            i_episode, avg_length, running_reward))
        running_reward = 0
        avg_length = 0
        torch.save(ppo_striker.policy.state_dict(),
                   './PPO_{}.pth'.format('SoccerTwos'))