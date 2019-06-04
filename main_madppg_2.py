import numpy as np
import torch
import gym
from ppo.PPO import PPO, Memory
from ppo.utils import ReplayBuffer
from env_exp import SocTwoEnv
from MADDPG import Maddpg
env_path = r'env\windows\soccer_easy\Unity Environment.exe'
env = SocTwoEnv(env_path, worker_id=2, train_mode=True)

# do not render the scene
e_render = False

# food_reward = 10.
# poison_reward = -1.
# encounter_reward = 0.01
# n_coop = 2

env_path = r'env\windows\soccer_easy\Unity Environment.exe'
env = SocTwoEnv(env_path, worker_id=0, train_mode=True)


# vis = visdom.Visdom(port=5274)
reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
# world.seed(1234)
# n_agents = world.n_pursuers
n_states = 213
# n_actions = 2
# capacity = 1000000
batch_size = 1000

n_episode = 20000
max_steps = 1000
episodes_before_train = -1

win = None
param = None

# maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
#                 episodes_before_train)
Maddpg_ = Maddpg(n_striker = 16,n_goalie = 16, g_dim_act = 5, use_cuda = True, 
                dim_obs = 112, s_dim_act = 7, batchSize = 1024, episode_before_training = 0)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
obs_striker, obs_goalie = env.reset()
for i_episode in range(n_episode):
    action_size_str = soc_env.striker_brain.vector_action_space_size
    action_size_goalie = soc_env.goalie_brain.vector_action_space_size
    # randomly generate some actions for each agent.
    action_striker = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    action_goalie = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    action_striker = np.random.randint(7, size=16, dtype=int)
    action_goalie = np.random.randint(5, size=16, dtype=int)

    soc_env.step(action_striker, action_goalie)

    soc_env.done()
    if True in soc_env.done_goalie:
        soc_env.reward()
        print("episode: ", episode, "*" * 10)
        arg_done_goalie = np.argwhere(soc_env.done_goalie == True)
        for i in arg_done_goalie:
            # print("goalie %d"%(i[0]))
            # print("action", soc_env.act_goalie_hist[i[0]])
            # print("Observation", soc_env.observation_goalie_hist[i[0]])
            # print("reword", soc_env.episode_goalie_rewards[i][0])
            pass

        arg_done_str = np.argwhere(soc_env.done_striker == True)
        for i in arg_done_str:
            # print("str %d"%(i[0]))
            # print("action", soc_env.act_striker_hist[i[0]])
            # print("Observation", soc_env.observation_striker_hist[i[0]])
            # print("reword", soc_env.episode_striker_rewards[i][0])
            pass
        # soc_env.reset_some_agents(arg_done_str, arg_done_goalie)
        print("*" * 25)
        episode += 1
env.close()
