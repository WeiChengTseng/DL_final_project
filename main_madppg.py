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
from torch.distributions import Categorical
import tensorboardX
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import datetime
from torch import nn

env_path = 'env/linux/SoccerTwosFast.x86_64'


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

max_episodes = 10000  # max training episodes
max_timesteps = 5000  # max timesteps in one episode

log_interval = 10  # print avg reward in the interval
update_timestep = 400  # update policy every n timesteps 2000
lr = 5e-4
gamma = 0.95  # discount factor
scale_reward = 1
random_seed = None
update_param = 100
# episode_before_training = 256
tau = 0.01
batchSize_d2 = 1024
addition = 0.0005
if random_seed:

    torch.manual_seed(random_seed)
    env.seed(random_seed)

# memory_striker = Memory()
Maddpg_ = Maddpg(
    n_striker=1,
    n_goalie=1,
    g_dim_act=5,
    use_cuda=True,
    lr=lr,
    dim_obs=112,
    s_dim_act=7,
    batchSize_d2=batchSize_d2,
    GAMMA=gamma,
    scale_reward=scale_reward,
    tau=tau,
    update_timestep=update_timestep)
# memory_goalie = Memory()
outf = 'result/1/'
if not os.path.exists(outf):
    os.makedirs(outf)

torch.autograd.set_detect_anomaly(True)

try:
    os.mkdir(outf)
except:
    pass
# logging variables
running_reward = 0

avg_length = 0
timestep = 0

capacity = int(3e4)


def main():
    # training loop
    # s_memory = ReplayMemory(capacity)
    memory = ReplayMemory(capacity)
    reward_mapping = np.zeros(16)
    
    episode = 0
    writer = SummaryWriter()
    prev_states = np.concatenate([np.zeros([16, 112]),
                                  np.zeros([16, 112])]).reshape(-1, 4, 112)
    prev_reward = np.concatenate([np.zeros([16]),
                                  np.zeros([16])]).reshape(-1, 4, 1)
    init_matrix = np.zeros([16, 7])
    init_matrix[:, 0] = 1
    prev_action_striker = torch.tensor(init_matrix).float()
    prev_action_goalie = torch.tensor(init_matrix).float()
    prev_action_striker = prev_action_striker.reshape(-1, 2, 7)
    prev_action_goalie = prev_action_goalie.reshape(-1, 2, 7)

    prev_action = torch.cat([prev_action_striker, prev_action_goalie], dim=1)
    reward_temp = np.zeros([8, 4, 1], dtype='double')
    reward_temp2 = np.zeros([2, 16], dtype='double')
    true_reward = np.zeros([2, 16], dtype='double')  # accumulate reward
    prev_done = np.arange(16)
    prev_states_striker = np.zeros((16, 112))
    prev_states_goalie = np.zeros((16, 112))
    mask_striker = np.zeros(16, dtype=bool)
    mask_goalie = np.zeros(16, dtype=bool)
    iter_test = 0
    test_loop = 10
    trans = 0
    test_str_reward = np.zeros(16)
    win_prob = []
    lose_prob = []
    draw_prob = []
    order = 'field'
    states = env.reset(order)
    Logsoftmax = nn.LogSoftmax(-1)
    
    while episode < max_episodes:

        if (trans+1) %1600 == 0 and episode> 400 and (temp == 0):
            state_striker, state_goalie = env.reset(order)
            iter_test = 0
            print("being test!!!")
            while iter_test < test_loop:
                while True:
                    action_space_str = np.zeros((16,))
                    action_space_goalie = np.zeros((16,))
                    action_striker_distr, action_goalie_distr = Maddpg_.select_action(
                        state_striker[[0, 2, 4, 6, 8, 10, 12, 14]],
                        state_goalie[[0, 2, 4, 6, 8, 10, 12, 14]])

                    action_striker = Categorical(
                        action_striker_distr).sample()
                    action_goalie = Categorical(
                        action_goalie_distr).sample()
                    action_space_str[[0, 2, 4, 6, 8, 10, 12,
                                     14]] = action_striker

                    action_space_goalie[[0, 2, 4, 6, 8, 10, 12,
                                        14]] = action_goalie

                    random_action_striker = np.random.randint(7, size=8)
                    random_action_goalie = np.random.randint(5, size=8)

                    action_space_str[[1, 3, 5, 7, 9, 11, 13,
                                     15]] = random_action_striker
                    action_space_goalie[[1, 3, 5, 7, 9, 11, 13,
                                        15]] = random_action_goalie

                    action_space_goalie[mask_goalie] = 0
                    action_space_str[mask_striker] = 0
                    states, reward, done, _ = env.step(action_space_str,
                                                       action_space_goalie,order)

                    state_striker = states[0]
                    state_goalie = states[1]

                    done[0][mask_striker] = True
                    done[1][mask_goalie] = True

                    for i in np.argwhere(done[0] == True):
                        if i in np.argwhere(mask_striker == False):
                            mask_striker[i] = True
                            test_str_reward[i] = reward[0][i]

                    for i in np.argwhere(done[1] == True):
                        if i in np.argwhere(mask_goalie == False):
                            mask_goalie[i] = True
                    

                    if (len(np.argwhere(done[0]).flatten()) + len(
                            np.argwhere(done[1]).flatten()) == 32):
                        win = test_str_reward > 0
                        lose= test_str_reward < -0.5
                        red = len(np.argwhere(win[[0,2,4,6,8,10,12,14]] == True))
                        lose =len(np.argwhere(lose[[0,2,4,6,8,10,12,14]] == True))
                        win_prob += [((red / 8)) * 100]
                        lose_prob += [(lose / 8 )*100]
                        draw_prob += [(1-(red / 8+lose/ 8))*100]
                        mask_striker[:] = False
                        mask_goalie[:] = False
                        iter_test += 1
                        if (iter_test % 2 == 0):
                            print("draw prob: ", ((1-(red / 8+lose/ 8)))* 100, " iter: ",
                                  iter_test)
                            print("win prob: ", ((red / 8)) * 100, " iter: ",
                                  iter_test)
                            print("lose prob: ", (( lose/ 8)) * 100, " iter: ",
                                  iter_test)
                        if iter_test == test_loop:
                            result = np.mean(np.array(win_prob))
                            draw_r = np.mean(np.array(draw_prob))
                            lose_r = np.mean(np.array((lose_prob)))
                            f = open(outf + "/result.txt", "a")
                            print(
                                "Now time: ", datetime.datetime.now(), file=f)
                            print(
                                'episode: {} stocastic result: '.format(
                                    episode),
                                result," draw: ",draw_r, " lose: ",lose_r,
                                file=f)
                            print(
                                'episode: {} stocastic result: '.format(
                                    episode),
                                result," draw: ",draw_r, " lose: ",lose_r)

                            print("Now time: ", datetime.datetime.now())
                            print("======================", file=f)
                            f.close()
                            win_prob = []

                        states = env.reset(order)
                        temp =1
                        print("test end!!!")
                        torch.save(Maddpg_,
                           outf + "model0_" + "episode_" + str(episode))
                        break
            # del memory.memory[:-1000]  
            # memory = ReplayMemory(capacity) 

        action_striker = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        action_goalie = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        t1 = time.time()

        #reward shaping
        # if len(memory) < max_timesteps:
        #     print("episode: ", episode)
        #     print("memory length: ",len(memory))
        #     action_striker_distr = np.random.normal(0.5,0.5, size = [16,7])
        #     action_striker_distr =torch.tensor(action_striker_distr).float()
        #     action_goalie_distr = np.random.normal(0.5,0.5, size = [16,7])
        #     action_goalie_distr = torch.tensor(action_goalie_distr).float()
        #     action_striker = np.argmax(action_striker_distr.cpu().detach().numpy(), axis = 1)
        #     action_goalie  = np.argmax(action_goalie_distr.cpu().detach().numpy(), axis = 1)
        if trans % 100 ==0:
            print("episode: ", episode)
            print("memory_length: ", len(memory))


        action_striker_distr, action_goalie_distr = Maddpg_.select_action(states[0], states[1])
        
        action_striker = Categorical(
            action_striker_distr.detach()).sample()
        action_goalie = Categorical(
            action_goalie_distr.detach()).sample()
        states, reward, done, _ = env.step(
            action_striker, action_goalie, order="field")

        
        # if episode <= 100:
        #     for i in range(16):
        #         if states[0][i][2 * 7] == 1 or states[0][i][1 * 7] == 1 or states[0][i][3 * 7] == 1 or states[0][i][5 * 7]== 1or states[0][i][6 * 7]== 1:
        #             reward_mapping[i] += (
        #                 addition - (0.00001 * (episode * 0.001)))
        #         else:
        #             reward_mapping[i] -= (
        #                 addition - (0.00001 * (episode * 0.001)))

        #     list(reward)[0] = list(reward)[0] + reward_mapping
        #     reward = tuple(reward)

        true_reward[0] += reward[0]
        true_reward[1] += reward[1]
        true_reward[0][prev_done] = 0
        true_reward[1][prev_done] = 0

        shaped_true_reward = [None] * 2
        shaped_true_reward[0] = (true_reward[0]).reshape(-1, 2, 1)  # 8 * 2 * 1
        shaped_true_reward[1] = (true_reward[1]).reshape(-1, 2, 1)  #8 * 2 * 1
        shaped_true_reward = np.concatenate(
            [shaped_true_reward[0], shaped_true_reward[1]],
            axis=1)  # 8 * 4 * 1
        states_temp = deepcopy(states)
        states_temp[0] = states_temp[0].reshape(-1, 2, 112)  # 8 * 2 * 112
        states_temp[1] = states_temp[1].reshape(-1, 2, 112)  # 8 * 2 * 112
        states_temp = np.concatenate([states_temp[0], states_temp[1]],
                                     axis=1)  # 8 * 4 * 112
        memory.push(prev_states, states_temp, prev_action, shaped_true_reward)

        if episode> 200:
            if len(memory)% update_param ==0:
                loss_a, loss_c = Maddpg_.update_policy(memory, trans,
                                                       writer)
            
        t2 = time.time()

        prev_states, prev_reward, prev_action_striker, prev_action_goalie = states, reward, action_striker_distr.detach(
        ), action_goalie_distr.detach()
        arg_done = np.squeeze(np.argwhere(done[0] == True), -1)
        
        prev_action_striker[arg_done] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        prev_action_goalie[arg_done] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        prev_states_striker = prev_states[0]
        prev_states_goalie = prev_states[1]
        prev_states[0] = prev_states[0].reshape(-1, 2, 112)  # 8 * 2 * 112
        prev_states[1] = prev_states[1].reshape(-1, 2, 112)  # 8 * 2 * 112
        prev_states = np.concatenate([prev_states[0], prev_states[1]],
                                     axis=1)  # 8 * 4 * 112

        prev_action_striker = prev_action_striker.reshape(-1, 2,
                                                          7)  # 8 * 2 * 7
        prev_action_goalie = prev_action_goalie.reshape(-1, 2, 7)  # 8 * 2 * 7

        prev_action = torch.cat((prev_action_striker, prev_action_goalie),
                                dim=1)
        # print(prev_action.size())    # 8 * 4 * 7

        prev_done = np.argwhere(done[0] == True)
        trans +=1

        if len(np.argwhere(env.done_goalie == True)) > 0:
            episode += len(np.argwhere(env.done_goalie == True)) * 2
            temp = 0


if __name__ == '__main__':
    main()