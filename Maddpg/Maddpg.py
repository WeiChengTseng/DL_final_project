import torch
import torch.nn as nn
from maddpg_model import Goalie_Critic,Striker_Critic,Goalie,Striker
from memory import Memory
import os
import winsound
class Maddpg:
    def __init__(self, n_striker = 2,n_goalie = 2, g_dim_act = 5,use_cuda = True, 
                dim_obs = 112, s_dim_act = 8, batchSize = 1024, episode_before_training = 0):
        if n_striker != n_goalie:
            winsound.Beep(800,2000)
            os.system('shutdown -s -t 0') 
            raise EnvironmentError("GAN")
            
        
        self.n_striker = n_striker
        self.n_goalie = n_goalie
        self.batchSize = batchSize
        self.memory = Memory
        self.dim_obs = dim_obs
        self.g_dim_act = g_dim_act
        self.s_dim_act = s_dim_act
        self.episode_before_training = episode_before_training
        self.s_actor = [Striker(self.dim_obs,self.s_dim_act) for i in range(self.n_striker)]
        self.s_critic = [Striker_Critic(2,self.dim_obs,self.s_dim_act) for i in range(self.n_striker)]
        self.g_actor = [Goalie(self.dim_obs,self.g_dim_act) for i in range(self.n_goalie)]
        self.g_critic = [Goalie_Critic(2,self.dim_obs,self.g_dim_act) for i in range(self.n_goalie)]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
        
        self.steps_done = 0
        self.episode_done = 0
    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        s_actions = torch.zeros(self.n_striker,self.s_dim_act)
        g_actions = torch.zeros(self.n_goalie,self.g_dim_act)

        # FloatTensor = torch.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_striker):
            sb = state_batch[i, :].detach()
            s_act = self.s_actor[i](sb.unsqueeze(0)).squeeze()
            g_act = self.g_actor[i](sb.unsqueeze(0)).squeeze()

            s_actions[i, :] = s_act
            g_actions[i, :] = g_act
        self.steps_done += 1

        return s_actions, g_actions


if __name__ == "__main__":
    Maddpg = Maddpg(1,2)
