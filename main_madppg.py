import numpy as np
import torch
import gym
from ppo.PPO import PPO, Memory
from ppo.utils import ReplayBuffer
from env_exp import SocTwoEnv
from MADDPG import Maddpg
env_path = r'env\windows\soccer_easy\Unity Environment.exe'
env = SocTwoEnv(env_path, worker_id=2, train_mode=True)

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


max_episodes = 50000    # max training episodes
max_timesteps = 300     # max timesteps in one episode
solved_reward = 230     # stop training if avg_reward > solved_reward
log_interval = 100      # print avg reward in the interval
update_timestep = 5        # update policy every n timesteps 2000
lr = 0.001
gamma = 0.99            # discount factor
K_epochs = 4            # update policy for K epochs
eps_clip = 0.2          # clip parameter for PPO
random_seed = None

if random_seed:
    torch.manual_seed(random_seed)
    env.seed(random_seed)

memory_striker = Memory()
Maddpg_ = Maddpg(n_striker = 16,n_goalie = 16, g_dim_act = 5, use_cuda = True, 
                dim_obs = 112, s_dim_act = 7, batchSize = 1024, episode_before_training = 0)
memory_goalie = Memory()

# logging variables
running_reward = 0
avg_length = 0
timestep = 0

def main():
# training loop
    state_striker, state_goalie = env.reset()
    episode = 0
    while episode < max_episodes:
        actions_striker = np.random.randint(7, size=16, dtype=int)
        actions_goalie = np.random.randint(5, size=16, dtype=int)
        states, reward, done, _ = env.step(actions_striker, actions_goalie)
        memory_striker.update_reward(reward[0])
        memory_goalie.update_reward(reward[1])
        if True in env.done_goalie:
            print("episode: ", episode, "*" * 10)
            arg_done_goalie = np.argwhere(env.done_goalie == True)
            for i in arg_done_goalie:
                # print("goalie %d"%(i[0]))
                # print("action", env.act_goalie_hist[i[0]])
                # print("Observation", env.observation_goalie_hist[i[0]])
                # print("reword", env.episode_goalie_rewards[i][0])
                pass
            arg_done_str = np.argwhere(env.done_striker == True)
            for i in arg_done_str:
                # print("str %d"%(i[0]))
                # print("action", env.act_striker_hist[i[0]])
                # print("Observation", env.observation_striker_hist[i[0]])
                # print("reword", env.episode_striker_rewards[i][0])
                pass
            # env.reset_some_agents(arg_done_str, arg_done_goalie)
            episode += 1
if __name__ == '__main__':
    main()