import numpy as np
import torch
import gym
from PPO.PPO import PPO
from PPO.utils import ReplayBuffer
from env_exp import SocTwoEnv
import random
import tensorboardX
import os
import argparse
import datetime
from a2c.agent_wraper import A2CWraper

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=64, help='hidden layer dim')
parser.add_argument('--max_episode', type=int, default=9601, help='max episode number')
parser.add_argument('--update_episode', type=int, default=320, help='update episode number')
parser.add_argument('--log_interval', type=int, default=320, help='log_interval')
parser.add_argument('--lr',default = 1e-3, type=float, help='learning rate')
parser.add_argument('--gamma',default = 0.99, type=float,help = 'discount')
parser.add_argument('--K_epochs',default = 10, type=int, help='K_epochs')
parser.add_argument('--eps_clip',default = 0.2, type=float, help='clipping expilson')
parser.add_argument('--test_loop',default = 10, type=int , help='test loop')
parser.add_argument("--folder", default="./PPO_summary",type=str, help="saving_folder")
parser.add_argument("--rewards_add", default=False ,type=bool, help="addition rewards")
parser.add_argument("--reward_add_value", default=0.0004 ,type=float, help="addition rewards")

opt = parser.parse_args()

if not os.path.exists(opt.folder):
    os.makedirs(opt.folder)
env_path = r'.\env\windows\SoccerTwosBirdView\Unity Environment.exe'
env = SocTwoEnv(env_path, worker_id=2, train_mode=True)


f_h = open(opt.folder+"/hyperparam.txt","a")
print("Current time: ", datetime.datetime.now(), file=f_h)
print("hidden dim: ", opt.hidden, file=f_h)
print("max_episode: ", opt.max_episode, file=f_h)
print("log_interval:", opt.log_interval, file=f_h)
print("update_episode: ", opt.update_episode, file=f_h)
print("learning rate: ", opt.lr, file=f_h)
print("gamma: ", opt.gamma, file=f_h)
print("K_epoch: ",opt.K_epochs, file=f_h)
print("clipping: ", opt.eps_clip, file=f_h)
print("test_loop: ", opt.test_loop, file=f_h)
print("folder: ", opt.folder, file=f_h)
# print("reward_addition on 90 degree: ", str(addition), file=f_h)
f_h.close()
############## Hyperparameters Striker ##############
state_dim_striker = 112
action_dim_striker = 7
n_latent_var_striker = opt.hidden  # number of variables in hidden layer
#############################################

############## Hyperparameters Goalie ##############
state_dim_goalie = 112
action_dim_goalie = 5
n_latent_var_goalie = opt.hidden  # number of variables in hidden layer
#############################################

max_episodes = opt.max_episode  # max training episodes
update_episode = opt.update_episode  # max timesteps in one episode
log_interval = opt.log_interval  # print avg reward in the interval

lr = opt.lr
gamma = opt.gamma  # discount factor
K_epochs = opt.K_epochs  # update policy for K epochs
eps_clip = opt.eps_clip  # clip parameter for PPO
random_seed = None

reward_mapping = np.zeros(16)

addition = opt.reward_add_value

if random_seed:
    torch.manual_seed(random_seed)
    env.seed(random_seed)


ppo_striker_ckpt = r'.\PPO_summary\PPO_strikerSoccerTwos_9600.pth'
ppo_goa_ckpt = r'.\PPO_summary\PPO_goalieSoccerTwos_9600.pth'
# memory_striker = Memory()
memory_striker = ReplayBuffer(16, gamma)
ppo_striker = PPO(state_dim_striker, action_dim_striker, n_latent_var_striker,
                  lr, gamma, K_epochs, eps_clip, ckpt_path=ppo_striker_ckpt)

# memory_goalie = Memory()
memory_goalie = ReplayBuffer(16, gamma)
ppo_goalie = PPO(state_dim_goalie, action_dim_goalie, n_latent_var_goalie, lr,
                 gamma, K_epochs, eps_clip, ckpt_path=ppo_goa_ckpt)

ppo2_str_ckpt_path = r'.\PPO\baseline_K_10_reward_variance_0.0002\PPO_strikerSoccerTwos_9600.pth'
ppo2_goa_ckpt_path = r'.\PPO\baseline_K_10_reward_variance_0.0002\PPO_goalieSoccerTwos_9600.pth'

ppo2_striker = PPO(state_dim_striker, action_dim_striker, n_latent_var_striker,
    lr, gamma, K_epochs, eps_clip, ckpt_path=ppo2_str_ckpt_path)

ppo2_goalie = PPO(state_dim_goalie, action_dim_goalie, n_latent_var_goalie, lr,
                 gamma, K_epochs, eps_clip, ckpt_path=ppo2_goa_ckpt_path)





# logging variables
running_reward_striker = 0
running_reward_goalie = 0

avg_length_goalie = 0
avg_length_striker = 0

i_episode = 0
count = 0


timestep_striker = np.zeros(16, dtype=int)
timestep_goalie = np.zeros(16, dtype=int)

writer = tensorboardX.SummaryWriter(opt.folder)

mask_striker = np.zeros(16,dtype=bool)
mask_goalie = np.zeros(16,dtype=bool)

test_str_reward = np.zeros(16)
temp = 0

test_loop =opt.test_loop
iter_test = 0

win_prob = []

# training loop
while i_episode < (max_episodes):
    state_striker, state_goalie = env.reset()
    while True:

        # testing
        if (((i_episode) % log_interval == 0)and (temp == 0)):
        # if (((i_episode) % log_interval == 0)and (temp == 0) and (i_episode!=0)):
            iter_test = 0
            state_striker, state_goalie = env.reset()
            while iter_test < test_loop:
                while True:
                    action_striker1 = ppo_striker.policy.act_test(state_striker[:8], action_dim_striker)
                    action_goalie1 = ppo_goalie.policy.act_test(state_goalie[:8], action_dim_goalie)
                    action_striker2 = ppo2_striker.policy.act_test(state_striker[8:], action_dim_striker)
                    action_goalie2 = ppo2_goalie.policy.act_test(state_goalie[8:], action_dim_goalie)
                    action_striker = np.concatenate((action_striker1, action_striker2),axis=None)
                    action_goalie = np.concatenate((action_goalie1, action_goalie2),axis=None)
                    action_goalie[mask_goalie] = 0
                    action_striker[mask_striker] = 0
                    states, reward, done, _ = env.step(action_striker, action_goalie)

                    state_striker = states[0] 
                    state_goalie = states[1]
                    
                    done[0][mask_striker] = True
                    done[1][mask_goalie] = True
                    
            
                    for i in np.argwhere(done[0]==True):
                        if i in np.argwhere(mask_striker==False):
                            mask_striker[i] = True
                            test_str_reward[i] = reward[0][i]

                    for i in np.argwhere(done[1]==True):
                        if i in np.argwhere(mask_goalie==False):
                            mask_goalie[i] = True

                    if (len(np.argwhere(done[0]).flatten())+ len(np.argwhere(done[1]).flatten()) == 32):
                        win = test_str_reward >0
                        red=len(np.argwhere(win[:8]==True))
                        win_prob += [(red/8)*100]
                        mask_striker[:] = False
                        mask_goalie[:] = False
                        iter_test +=1
                        if (iter_test % 2 == 0):
                            print("win prob: ", (red/8)*100, " iter: ", iter_test)
                        if iter_test == test_loop:
                            result =np.mean(np.array(win_prob))
                            f = open(opt.folder+"/result.txt","a")
                            print("Now time: ", datetime.datetime.now(), file=f)
                            print('episode: {} stocastic result: '.format(i_episode),result, file= f)
                            print('episode: {} stocastic result: '.format(i_episode),result)
                            print("Now time: ", datetime.datetime.now())
                            print("======================", file=f)
                            f.close()
                            win_prob = []

                        state_striker, state_goalie = env.reset()
                        temp = 1
                        break

        timestep_striker += 1
        timestep_goalie += 1

        # Running policy_old:
        action_striker = ppo_striker.policy_old.act(state_striker,
                                                    memory_striker)
        action_goalie = ppo_goalie.policy_old.act(state_goalie, memory_goalie)

        action_goalie[mask_goalie] = 0
        action_striker[mask_striker] = 0

        if i_episode <= 6400:
            for i in range(16):
                if state_striker[i][2*7] == 1:
                    reward_mapping[i] += (addition -(0.00001*(i_episode*0.001)))
                else:
                    reward_mapping[i] -= (addition -(0.00001*(i_episode*0.001)))


        states, reward, done, _ = env.step(action_striker, action_goalie)
        
        if opt.rewards_add:
            list(reward)[0] = list(reward)[0] + reward_mapping
            reward = tuple(reward)
        
        state_striker = states[0]
        state_goalie = states[1]

        done[0][mask_striker] = True
        done[1][mask_goalie] = True

        # Saving reward:
        memory_striker.update_reward(reward[0], done[0])
        memory_goalie.update_reward(reward[1], done[1])


        running_reward_striker += reward[0]
        running_reward_goalie += reward[1]

        
        reward_mapping[i] = 0

        

        for i in np.argwhere(done[0]==True):
            if i in np.argwhere(mask_striker==False):
                mask_striker[i] = True
        
        for i in np.argwhere(done[1]==True):
            if i in np.argwhere(mask_goalie==False):
                mask_goalie[i] = True
        
        if (len(np.argwhere(done[0]).flatten())+ len(np.argwhere(done[1]).flatten()) == 32):
            i_episode += 32
            memory_striker.update_record()
            memory_goalie.update_record()
            
            mask_striker[:] = False
            mask_goalie[:] = False
            count +=1
            temp = 0

            break

    if ((i_episode) % update_episode == 0):
        ppo_striker.update(memory_striker)
        ppo_goalie.update(memory_goalie)
        memory_striker.clear_memory()
        memory_goalie.clear_memory()

    avg_length_goalie += timestep_goalie
    avg_length_striker += timestep_striker

    timestep_goalie[:] = 0
    timestep_striker[:] = 0

    # logging
    if (i_episode % 64 == 0):
        print("episode: ",i_episode)
    if ((i_episode % log_interval) == 0):
        
        count = 1
        avg_length_goalie = np.sum(avg_length_goalie) / log_interval / 16
        avg_length_striker = np.sum(avg_length_striker) / log_interval /16
        avg_running_reward_striker = np.sum(running_reward_striker) / log_interval /16
        avg_running_reward_goalie = np.sum(running_reward_goalie) /log_interval /16

        print('Episode {} \t avg striker length: {} \t reward: {}'.format(
            i_episode, avg_length_striker, avg_running_reward_striker))

        
        print('Episode {} \t avg goalie length: {} \t reward: {}'.format(
            i_episode, avg_length_goalie, avg_running_reward_goalie))        

        running_reward_striker[:] = 0
        running_reward_goalie[:] = 0

        avg_length_goalie = 0
        avg_length_striker = 0

        torch.save(ppo_striker.policy.state_dict(),
                   './'+opt.folder+'/PPO_striker{}_{}.pth'.format('SoccerTwos',i_episode))
        torch.save(ppo_goalie.policy.state_dict(),
                   './'+opt.folder+'/PPO_goalie{}_{}.pth'.format('SoccerTwos',i_episode))
