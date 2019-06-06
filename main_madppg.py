import numpy as np
import torch
import gym
# from ppo.PPO import PPO, Memory
from ppo.utils import ReplayBuffer
from env_exp import SocTwoEnv
from MADDPG import Maddpg
from memory import ReplayMemory
from copy import deepcopy
import time
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
 
 
max_episodes = 500        # max training episodes
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
Maddpg_ = Maddpg(n_striker = 1,n_goalie = 1, g_dim_act = 5, use_cuda = True,lr = 0.0001,
                dim_obs = 112, s_dim_act = 7, batchSize_d2 = 8, episode_before_training = 0, GAMMA = 1., scale_reward = 1.)
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
    states = env.reset()
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
 
        t1 = time.time()       
        # if episode < 20:
            # action_striker = np.random.randint(7, size = [16])
            # action_goalie = np.random.randint(5, size = [16])
            # action_striker = np.array(action_striker)
            # action_goalie = np.array(action_goalie)
        # else:
        action_striker, action_goalie = Maddpg_.select_action(states[0], states[1])
        action_striker = np.argmax(action_striker.cpu().detach().numpy(), axis = 1)
        action_goalie = np.argmax(action_goalie.cpu().detach().numpy(), axis = 1)
        t2 = time.time()
        print(action_striker)
        print('action require: %f s' % (t2-t1))
        states, reward, done, _ = env.step(action_striker, action_goalie, order = "field")
        
        if episode<500:
            for i in range(action_striker):
                if i == 0:
                    reward[i] -=0.005
              
        states_temp = deepcopy(states)
        states_temp[0] = states_temp[0].reshape(-1, 2, 112)
        states_temp[1] = states_temp[1].reshape(-1, 2, 112)
        states_temp = np.concatenate([states_temp[0], states_temp[1]], axis=1)

        memory.push(prev_states, states_temp, prev_action, prev_reward)
        t1 = time.time()
        loss_a , loss_c = Maddpg_.update_policy(memory)
        t2 = time.time()
        print(loss_a,loss_c)
        print('Update require: %f s' % (t2-t1))
        
        prev_states, prev_reward, prev_action_striker, prev_action_goalie = states, reward, action_striker, action_goalie

        arg_done = np.argwhere(done[0] == True)
        prev_states[0][arg_done] = np.zeros([112])
        prev_states[1][arg_done] = np.zeros([112])
        prev_reward[0][arg_done] = 0
        prev_reward[1][arg_done] = 0
        prev_action_striker[arg_done] = 0
        prev_action_goalie[arg_done] = 0

 
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
