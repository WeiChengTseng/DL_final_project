import numpy as np
import os
import torch
import gym
import pickle
# from ppo.PPO import PPO, Memory
# from ppo.utils import ReplayBuffer
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
 
 
max_episodes = 50000        # max training episodes
max_timesteps = 300     # max timesteps in one episode
solved_reward = 230     # stop training if avg_reward > solved_reward
log_interval = 100      # print avg reward in the interval
update_timestep = 20        # update policy every n timesteps 2000
lr = 0.0001
gamma = 0.99            # discount factor
scale_reward = 1.          
random_seed = None
episode_before_training = 256*10
tau = 0.01
batchSize_d2 = 512
if random_seed:
    
    torch.manual_seed(random_seed)
    env.seed(random_seed)
 
# memory_striker = Memory()
Maddpg_ = Maddpg(n_striker = 1,n_goalie = 1, g_dim_act = 5, use_cuda = True,lr = lr,
                dim_obs = 112, s_dim_act = 7, batchSize_d2 = batchSize_d2,
                GAMMA = gamma, scale_reward = scale_reward, tau = tau, update_timestep = update_timestep)
# memory_goalie = Memory()
outf = 'result/1/'
try:
    os.mkdir(outf)
except:
    pass
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
    reward_temp = np.zeros([8, 4, 1], dtype='double')
    reward_temp2 = np.zeros([2, 16], dtype='double')
    true_reward = np.zeros([2, 16,], dtype='double')
    prev_done = np.arange(16)
    while episode < max_episodes:
 
        action_striker = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        action_goalie = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
 
        t1 = time.time()       
        if episode < 200:
            action_striker = np.random.randint(7, size = [16])
            action_goalie = np.random.randint(5, size = [16])
            action_striker = np.array(action_striker)
            action_goalie = np.array(action_goalie)
        else:
            action_striker, action_goalie = Maddpg_.select_action(states[0], states[1])
            action_striker = np.argmax(action_striker.cpu().detach().numpy(), axis = 1)
            action_goalie = np.argmax(action_goalie.cpu().detach().numpy(), axis = 1)
        t2 = time.time()
        
        # print('action require: %f s' % (t2-t1))
        states, reward, done, _ = env.step(action_striker, action_goalie, order = "field")
        true_reward[0] += reward[0]
        true_reward[1] += reward[1]
        true_reward[0][prev_done] = 0
        true_reward[1][prev_done] = 0
        shaped_true_reward = [None] * 2
        shaped_true_reward[0] = (true_reward[0]).reshape(-1, 2, 1)
        shaped_true_reward[1] = (true_reward[1]).reshape(-1, 2, 1)
        shaped_true_reward = np.concatenate([shaped_true_reward[0], shaped_true_reward[1]], axis=1)
        states_temp = deepcopy(states)
        states_temp[0] = states_temp[0].reshape(-1, 2, 112)
        states_temp[1] = states_temp[1].reshape(-1, 2, 112)
        states_temp = np.concatenate([states_temp[0], states_temp[1]], axis=1)
        memory.push(prev_states, states_temp, prev_action, shaped_true_reward)
        t1 = time.time()
        f = open(outf + 'log.csv' , 'w')
        if len(memory) > 256*10:
            loss_a , loss_c = Maddpg_.update_policy(memory)
            if episode % 10 == 0 and episode > 0:
                print("epi", episode)
                torch.save(Maddpg_, outf +"model0_" + "episode_" + str(episode))
        t2 = time.time()
        # print(loss_a,loss_c)
        # print('Update require: %f s' % (t2-t1))
        # print(type(reward), type(reward[0]))
        # exit()
        prev_states, prev_reward, prev_action_striker, prev_action_goalie = states, reward, action_striker, action_goalie
        arg_done = np.argwhere(done[0] == True)
        
        prev_states[0][arg_done] = np.zeros([112])
        prev_states[1][arg_done] = np.zeros([112])
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
 
        prev_done = np.argwhere(done[0] == True)
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
