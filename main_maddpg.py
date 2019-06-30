import numpy as np
import os
import torch
import gym
import pickle
from env_exp import SocTwoEnv
import time
from torch.distributions import Categorical
import tensorboardX
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import datetime
from torch import nn
import argparse
from MADDPG_ import Maddpg
from memory import ReplayBuffer
import random

ENVPATH = 'env/linux/SoccerTwosFast.x86_64'

STATEDIM = 112
ACTIONDIMSTRIKER = 7
HIDDENDIMSTRIKER = 64

STATEDIM = 112
ACTIONDIMGOALIE = 5
HIDDENDIMGOALIE = 64

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--Lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument(
        "--Ga", type=float, default=0.99, help="Discount coefficient")
    parser.add_argument("--Bz", type=int, default=512, help="Batch Size")
    parser.add_argument("--Tau", type=float, default=0.01, help="Tau")
    parser.add_argument("--UpdateEpisode",type=int,default=100,
        help="At least have 320 episode so you can update ")
    parser.add_argument("--UpdateParam",type=int,default=10,
        help="update parameter every UpdateParam step")
    parser.add_argument(
        "--ScaleReward", type=float, default=1.0, help="scale up Reward")
    parser.add_argument(
        "--TestEpisode", type=int, default=320, help="testing")
    parser.add_argument(
        "--LambdaScaler", type=float, default=0.01, help="scale up entropy")
    parser.add_argument(
        "--MaxEp", type=int, default=10000, help="Maximum Episode")
    parser.add_argument(
        "--Outf", type=str, default="./result/result_", help="saving model")
    parser.add_argument(
        "--Log", type=int, default=320, help="record every log episodes")
    parser.add_argument(
        "--capacity", type=int, default=2e5, help="capacity")
    opt = parser.parse_args()

    #initialize
    episode = 0
    order = 'field'
    Maddpg_object = Maddpg(
        tau=opt.Tau,
        GAMMA=opt.Ga,
        lr=opt.Lr,
        batchSize_d2=opt.Bz,
        update_timestep=opt.UpdateParam,
        lambda_scale=opt.LambdaScaler)
      
    Buffer = ReplayBuffer(capacity=opt.capacity)
    folder = None
    mask_striker = np.zeros(16, dtype=bool)
    mask_goalie = np.zeros(16, dtype=bool)
    mask = np.zeros(8, dtype=bool)
    test_loop = 2
    test_str_reward = np.zeros(16)

    #record episode rewards
    now = datetime.datetime.now()
    now = now.strftime("%Y_%m_%d-%H%M")
    if not os.path.exists(opt.Outf + str(now)):
        os.makedirs(opt.Outf + str(now))
        folder = opt.Outf + now
        f_h = open(folder + "/hyperparam.txt", "a")
        print("Learning rate: {}".format(opt.Lr), file=f_h)
        print("Discount Gamma: {}".format(opt.Ga), file=f_h)
        print("Tau: {}".format(opt.Tau), file=f_h)
        print(
            "Update Parameter every {} episode".format(opt.UpdateParam),
            file=f_h)
        print("Current time: ", datetime.datetime.now(), file=f_h)
        print("log_interval:", opt.Log, file=f_h)
        print("batch size: ", opt.Bz, file=f_h)
        print("folder: ", folder, file=f_h)
        print("LambdaScaler:", opt.LambdaScaler, file=f_h)
        f_h.close()
    writer = SummaryWriter(folder)

    #initize!!
    Env = SocTwoEnv(ENVPATH, worker_id=0, train_mode=True)
    time_step_0 = 0
    time_step_1 = 0
    time_step_2 = 0
    time_step_3 = 0
    time_step = 1
    temp = 0
    while episode < opt.MaxEp:
        
        #init 
        for i in range(8):
            Buffer._trags[i].clear()
        state_str, state_goal = Env.reset(order)
        done = np.zeros(8, dtype=bool)
        pre_sta_s = state_str
        pre_sta_g = state_goal
        pre_reward_s = np.zeros(16)
        pre_reward_g = np.zeros(16)
        prev_len = 0

        if (episode % opt.TestEpisode == 0) and (temp == 0):
            print("Being test!!")
            team_set =["blue","red"]
            team=random.choice(team_set)
            print("team_{} is testing!!!".format(team))
            f_h = open(folder + "/result.txt", "a")
            iter_test = 0
            while iter_test < test_loop:
                while True:
                    action_striker, action_goalie = Maddpg_object.select_action_test(pre_sta_s, pre_sta_g,team=team)
                    action_goalie[mask_goalie] = 0
                    action_striker[mask_striker] = 0
                    cur_states, cur_reward, cur_done, _ = Env.step(action_striker, action_goalie, order)
                    state_striker = cur_states[0]
                    state_goalie = cur_states[1]
                    reward_s = cur_reward[0]
                    reward_g = cur_reward[1]
                    cur_done[0][mask_striker] = True
                    cur_done[1][mask_goalie] = True
                    for i in np.argwhere(cur_done[0] == True):
                        if i is []:
                            break
                        if i in np.argwhere(mask_striker == False):
                            mask_striker[i] = True
                            test_str_reward[i] = cur_reward[0][i]

                    for i in np.argwhere(cur_done[1] == True):
                        if i is []:
                            break
                        if i in np.argwhere(mask_goalie == False):
                            mask_goalie[i] = True

                    pre_sta_s = cur_states[0]
                    pre_sta_g = cur_states[1]
                    if (len(np.argwhere(cur_done[0]).flatten()) + len(np.argwhere(cur_done[1]).flatten()) == 32):
                        win = test_str_reward > 0.5
                        lose= test_str_reward < -0.05
                        iter_test += 1
                        mask_striker[:] = False
                        mask_goalie[:] = False
                        if team =="red":
                            print("team_{} test result {} time on episode {}".format(team,iter_test,episode))
                            w_number = len(np.argwhere(win[[0,2,4,6,8,10,12,14]] == True))
                            print("win rate: " ,w_number/ 8 ,file=f_h)
                            print("win rate: " ,w_number/ 8)
                            l_number = len(np.argwhere(lose[[0,2,4,6,8,10,12,14]] == True))
                            print("lose rate: " ,l_number/ 8,file=f_h)
                            print("lose rate: " ,l_number/ 8)
                        else:
                            print("team_{} test result {} time on episode {}".format(team,iter_test,episode))
                            w_number = len(np.argwhere(win[[1,3,5,7,9,11,13,15]] == True))
                            print("win rate: " ,w_number/ 8 ,file=f_h)
                            print("win rate: " ,w_number/ 8)
                            l_number = len(np.argwhere(lose[[1,3,5,7,9,11,13,15]] == True))
                            print("lose rate: " ,l_number/ 8,file=f_h)
                            print("lose rate: " ,l_number/ 8)

                        if iter_test == test_loop:
                            temp =1
                            Maddpg_object.save_model(folder,episode)
                            st = datetime.datetime.now()
                            st = st.strftime("%Y_%m_%d-%H%M")
                            print("test end at :", st, file=f_h)
                            print("test end at :", st)
                            f_h.close()
                        pre_sta_s, pre_sta_g = Env.reset(order)
                        break
        
        while True:
            temp = 0
            action_striker_distr, action_goalie_distr = Maddpg_object.select_action_train(
                pre_sta_s, pre_sta_g)
            action_striker_distr=F.gumbel_softmax(action_striker_distr, -1)
            action_goalie_distr=F.gumbel_softmax(action_goalie_distr, -1)

            action_striker = torch.argmax(action_striker_distr,1).detach().numpy()
            action_goalie = torch.argmax(action_goalie_distr[:, :5],1).detach().numpy()

            action_goalie[mask_goalie] = 0
            action_striker[mask_striker] = 0

            cur_states, cur_reward, cur_done, _ = Env.step(
                action_striker, action_goalie, order)

            state_striker = cur_states[0]
            state_goalie = cur_states[1]
            reward_s = cur_reward[0]
            reward_g = cur_reward[1]

            cur_done[0][mask_striker] = True
            cur_done[1][mask_goalie] = True

            for i in np.argwhere(cur_done[0] == True):
                if i is []:
                    break
                if i in np.argwhere(mask_striker == False):
                    done[int(np.floor(i / 2))] = True

            for i in np.argwhere(done == False):
                if i is []:
                    break
                for j in i:
                    reward_s[j * 2:j * 2 + 1] += pre_reward_s[j * 2:j * 2 + 1]
                    reward_g[j * 2:j * 2 + 1] += pre_reward_g[j * 2:j * 2 + 1]

            for i in np.argwhere(cur_done[0] == True):
                if i is []:
                    break
                if i in np.argwhere(mask_striker == False):
                    done[int(np.floor(i / 2))] = False

            Buffer.update_transition(pre_sta_s, pre_sta_g, action_striker,
                                     action_goalie, action_striker_distr,
                                     action_goalie_distr, reward_s, reward_g,
                                     done, state_striker, state_goalie)
                   
            for i in np.argwhere(cur_done[0] == True):
                if i is []:
                    break
                if i in np.argwhere(mask_striker == False):
                    mask_striker[i] = True
                    done[int(np.floor(i / 2))] = True

            for i in np.argwhere(cur_done[1] == True):
                if i is []:
                    break
                if i in np.argwhere(mask_goalie == False):
                    mask_goalie[i] = True

            Buffer.clear_memory()
            time_step += 1
            if time_step % 1000 == 0:
                print("memory length: ",len(Buffer.states[0]))
                print("episode: ",episode)
                print("step: ",time_step)

            if episode > opt.UpdateEpisode:
                if (time_step % opt.UpdateParam==0):
                    seed = time_step
                    Maddpg_object.update_policy(Buffer,time_step,writer,seed)
                    seed = time_step*2
                    Maddpg_object.update_policy(Buffer,time_step,writer,seed)


            Buffer.update_rewards(done)

            pre_sta_s = cur_states[0]
            pre_sta_g = cur_states[1]
            pre_act_str_distr = action_striker_distr
            pre_act_goal_distr = action_goalie_distr
            pre_reward_s = reward_s
            pre_reward_g = reward_g
            prev_len = (len(np.argwhere(cur_done[0]).flatten()) + len(
                np.argwhere(cur_done[1]).flatten()))

            if (len(np.argwhere(cur_done[0]).flatten()) + len(
                    np.argwhere(cur_done[1]).flatten()) == 32):
                episode += 32
                mask_striker[:] = False
                mask_goalie[:] = False
                done[:] = False
                Buffer._record[:] =False
                print("episode: ",episode)
                # print("done: ",done)
                break

        
                
