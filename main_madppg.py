import numpy as np
import torch
import gym
# from ppo.PPO import PPO, Memory
from ppo.utils import ReplayBuffer
from env_exp import SocTwoEnv
from MADDPG import Maddpg
from memory import ReplayMemory
env_path = r'env\windows\SoccerTwosBirdView\Unity Environment.exe'

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


max_episodes = 100        # max training episodes
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

# memory_striker = Memory()
# Maddpg_ = Maddpg(n_striker = 16,n_goalie = 16, g_dim_act = 5, use_cuda = True, 
#                 dim_obs = 112, s_dim_act = 7, batchSize_d2 = 1024, episode_before_training = 0)
# memory_goalie = Memory()

# logging variables
running_reward = 0
avg_length = 0
timestep = 0
capacity = int(1e7)
def main():
# training loop
    # s_memory = ReplayMemory(capacity)
    memory = ReplayMemory(capacity)
    state_striker, state_goalie = env.reset()
    episode = 0
    prev_states = np.concatenate([np.zeros([16, 112]), np.zeros([16, 112])]).reshape(-1, 4, 112)
    prev_reward = np.concatenate([np.zeros([16]), np.zeros([16])]).reshape(-1, 4, 1)
    prev_action_striker = np.zeros([16])
    prev_action_goalie = np.zeros([16])
    prev_action_striker = prev_action_striker.reshape(-1, 2, 1) 
    prev_action_goalie = prev_action_goalie.reshape(-1, 2, 1) 
    prev_action = np.concatenate([prev_action_striker, prev_action_goalie], axis=1)    

    while episode < max_episodes:

        action_striker = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        action_goalie = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        action_striker = np.random.randint(7, size = [16])
        action_goalie = np.random.randint(5, size = [16])
        action_striker = np.array(action_striker)
        action_goalie = np.array(action_goalie)
        
        states, reward, done, _ = env.step(action_striker, action_goalie, order = "field")
        # print(states[0].shape)
        # states[0] = states[0].reshape(2, -1, 112)
        # print(states[0].shape)
        # states[1] = states[1].reshape(2, -1, 112)
        # states = np.vstack([states[0], states[1]])
        # reward[0] = reward[0].reshape(2, -1)
        # reward[1] = reward[1].reshape(2, -1)
        # reward = np.vstack([reward[0], reward[1]])
        # done[0] = done[0].reshape(2, -1)
        # done[1] = done[1].reshape(2, -1)
        # done = np.vstack([done[0], done[1]])
        # print(states.shape)
        # print(reward.shape)
        # print(done.shape)
        # exit()
        memory.push(prev_states, states, prev_action, prev_reward)

        arg_done = np.argwhere(done[0] == True)
        prev_states[0][arg_done] = np.zeros([112])
        prev_states[1][arg_done] = np.zeros([112])
        prev_reward[0][arg_done] = 0
        prev_reward[1][arg_done] = 0
        prev_action_striker[arg_done] = 0
        prev_action_goalie[arg_done] = 0


        prev_states, prev_reward, prev_action_striker, prev_action_goalie = states, reward, action_striker, action_goalie
        prev_states[0] = prev_states[0].reshape(-1, 2, 112)
        prev_states[1] = prev_states[1].reshape(-1, 2, 112)
        prev_states = np.concatenate([prev_states[0], prev_states[1]], axis=1)


        prev_reward[0] = prev_reward[0].reshape(-1, 2, 1)
        prev_reward[1] = prev_reward[1].reshape(-1, 2, 1)
        prev_reward = np.concatenate([prev_reward[0], prev_reward[1]], axis=1)

        prev_action_striker = prev_action_striker.reshape(-1, 2, 1) 
        prev_action_goalie = prev_action_goalie.reshape(-1, 2, 1) 
        prev_action = np.concatenate([prev_action_striker, prev_action_goalie], axis=1)    

     
        if True in env.done_goalie:
        #     print("episode: ", episode, "*" * 10)
        #     # print(reward)
        #     # arg_done_goalie = np.argwhere(done_goa == True)
        #     if len(arg_done_goalie) == 2:
        #         print("arg_done_goalie", arg_done_goalie)

        #     for i in arg_done_goalie:
        #         # print("goalie %d"%(i[0]))
        #         # print("action", env.act_goalie_hist[i[0]])
        #         # print("Observation", env.observation_goalie_hist[i[0]])
        #         # print("reword", env.episode_goalie_rewards[i][0])
        #         pass
        #     arg_done_str = np.argwhere(done_goa == True)
        #     if len(arg_done_goalie) == 2:
        #         print("arg_done_str", arg_done_str)

        #     for i in arg_done_str:
        #         # print("str %d"%(i[0]))
        #         # print("action", env.act_striker_hist[i[0]])
        #         # print("Observation", env.observation_striker_hist[i[0]])
        #         # print("reword", env.episode_striker_rewards[i][0])
        #         pass
        #     # env.reset_some_agents(arg_done_str, arg_done_goalie)
            episode += 1
if __name__ == '__main__':
    main()