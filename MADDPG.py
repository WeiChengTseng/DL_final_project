import torch
import torch.nn as nn
import numpy as np
from maddpg_model import Critic,Goalie,Striker
from memory import Experience
from copy import deepcopy
import os
from torch.optim import Adam
import winsound


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)

class Maddpg:
    def __init__(self, n_striker = 1,n_goalie = 1, g_dim_act = 5,use_cuda = True,lr = 0.0001,
                dim_obs = 112, s_dim_act = 7, batchSize_d2 = 1024, episode_before_training = 1024 * 10, GAMMA = 1., scale_reward = 1.):
        if n_striker != n_goalie:
            winsound.Beep(800,2000)
            # os.system('shutdown -s -t 0') 
            raise EnvironmentError("GAN")
        
        self.lr = lr
        self.GAMMA = GAMMA
        self.n_striker = n_striker
        self.n_goalie = n_goalie
        self.batchSize_d2 = batchSize_d2
        self.dim_obs = dim_obs
        self.scale_reward = scale_reward
        self.tau = 0.01
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

        self.g_dim_act = g_dim_act
        self.s_dim_act = s_dim_act
        self.episode_before_training = episode_before_training
        self.s_actor = [Striker(self.dim_obs,self.s_dim_act).to(self.device) for i in range(self.n_striker)]
        self.critic = [Critic(4,self.dim_obs,self.s_dim_act).to(self.device) for i in range(self.n_striker+self.n_goalie)]
        self.g_actor = [Goalie(self.dim_obs,self.g_dim_act).to(self.device) for i in range(self.n_goalie)]
        self.s_actors_target = deepcopy(self.s_actor)
        self.g_actors_target = deepcopy(self.g_actor)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=lr) for x in self.critic]
        self.s_actor_optimizer = [Adam(x.parameters(),
                                     lr=lr) for x in self.s_actor]
        self.g_actor_optimizer = [Adam(x.parameters(),
                                     lr=lr) for x in self.g_actor]
        
        
        self.steps_done = 0
        self.episode_done = 0
    def select_action(self, striker_batch, goalie_batch):
        # state_batch: n_agents x state_dim
        striker_batch = torch.from_numpy(striker_batch)
        goalie_batch = torch.from_numpy(goalie_batch)
        striker_batch = striker_batch.to(self.device)
        goalie_batch = goalie_batch.to(self.device)
        striker_batch = striker_batch.view(-1,112)
        
        goalie_batch = goalie_batch.view(-1,112)
        
        s_b = striker_batch.detach().float()
        g_b = goalie_batch.detach().float()
        s_act = self.s_actor[0](s_b.unsqueeze(0)).squeeze()
        g_act = self.g_actor[0](g_b.unsqueeze(0)).squeeze()
        self.steps_done += 1

        return s_act, g_act

    def update_policy(self, memory):

        # momory format is memory.push(prev_state, states, [prev_action_striker, prev_action_goalie], prev_reward)
        # do not train until exploration is enough
        # if self.episode_done <= self.episode_before_training:
            # return None, None

        c_loss = []
        a_loss = []
        
        if len(memory) < 1024*10:
            return None, None
        for agent_index in range(self.n_striker):

            #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   # 
            # batch sample is batch * N play ground * agents * state/next_state/action/reward/      #
            #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
            
            transitions = memory.sample(self.batchSize_d2)
            batch = Experience(*zip(*transitions))

            batch_state = np.asarray(batch.states)
            batch_action = np.asarray(batch.actions)
            batch_reward = np.asarray(batch.rewards)
            batch_reward = torch.from_numpy(batch_reward).to(self.device).float()
            batch_next_state = np.asarray(batch.next_state)
            
            state_batch = torch.from_numpy(batch_state)
            action_batch = torch.from_numpy(batch_action)
            next_state_batch = torch.from_numpy(batch_next_state)
            #   #   #   #   #   #   #   #   #
            # total numbers of data = batchsize * play ground   #
                                #   #   #   #   #   #   #   #   #
            total_numbers_of_data = batch_state.shape[0]* batch_state.shape[1]
            
        
            whole_state = state_batch.view(total_numbers_of_data, -1).to(self.device).float()
            whole_action = action_batch.view(total_numbers_of_data * 4,-1).long()

            #   #   #   #
            # translate action into one hot #
                                #   #   #   #    
            one_hot = (whole_action == torch.arange(7).reshape(1,7)).float()
            one_hot = one_hot.view(total_numbers_of_data , -1).to(self.device)
             
            self.critic_optimizer[0].zero_grad()
            self.critic_optimizer[1].zero_grad()
            
            s_current_Q = self.critic[0](whole_state, one_hot)
            g_current_Q = self.critic[1](whole_state, one_hot)
            s_whole_next_state = next_state_batch[:,:,0:2,:].to(self.device).float()
            s_whole_next_state = s_whole_next_state.view(total_numbers_of_data*2,-1)
            
            g_whole_next_state = next_state_batch[:,:,2:4,:].to(self.device).float()
            g_whole_next_state = g_whole_next_state.view(total_numbers_of_data*2,-1)
            #   #   #
            # Next_actions  #
                    #   #   #
            s_next_actions = [self.s_actors_target[0](s_whole_next_state)]
            g_next_actions = [self.g_actors_target[0](g_whole_next_state)]
            s_next_actions = torch.stack(s_next_actions)
            g_next_actions = torch.stack(g_next_actions)
            s_next_actions = (s_next_actions.transpose(0,1).contiguous())
            g_next_actions = (g_next_actions.transpose(0,1).contiguous())

            
            s_next_state = s_whole_next_state.view(-1,2,112).to(self.device)
            g_next_state = g_whole_next_state.view(-1,2,112).to(self.device)
            whole_next_stat = torch.cat([s_next_state,g_next_state], dim = -2)
            
            s_next_actions = s_next_actions.view(-1,14).to(self.device)
            g_next_actions = g_next_actions.view(-1,14).to(self.device)
            whole_next_action = torch.cat([s_next_actions,g_next_actions],dim = -1)
            
            whole_next_stat = whole_next_stat.view(-1,112*4)
            whole_next_action = whole_next_action.view(-1,7*4)
            
            s_target_Q = self.critic_target[0](whole_next_stat,whole_next_action)
            g_target_Q = self.critic_target[1](whole_next_stat,whole_next_action)
            # scale_reward: to scale reward in Q functions
            batch_reward = batch_reward.view(64,-1)
            
            s_1_target_Q = (s_target_Q * self.GAMMA) + (batch_reward[:, 0].unsqueeze(1) * self.scale_reward)
            s_2_target_Q = (s_target_Q * self.GAMMA) + (batch_reward[:, 1].unsqueeze(1) * self.scale_reward)
            g_1_target_Q = (g_target_Q * self.GAMMA) + (batch_reward[:, 2].unsqueeze(1) * self.scale_reward)
            g_2_target_Q = (g_target_Q * self.GAMMA) + (batch_reward[:, 3].unsqueeze(1) * self.scale_reward)
            # 64 *1 

            # # #
            # Update first striker #
                               # # #
            s_1_loss_Q = nn.MSELoss()(s_current_Q, s_1_target_Q.detach())
            s_1_loss_Q.backward(retain_graph = True)
            self.critic_optimizer[0].step()
            
            # # #
            # Update 2nd striker #
                             # # #
            # print(s_2_target_Q)
            self.critic_optimizer[0].zero_grad()
            s_2_loss_Q = nn.MSELoss()(s_current_Q, s_2_target_Q.detach())
            s_2_loss_Q.backward()
            self.critic_optimizer[0].step()
            
            # # #
            # Update first goalie #
                              # # #
            self.critic_optimizer[1].zero_grad()
            g_1_loss_Q = nn.MSELoss()(g_current_Q, g_1_target_Q.detach())
            g_1_loss_Q.backward(retain_graph = True)
            self.critic_optimizer[1].step()

            # # #
            # Update 2nd goalie #
                            # # #
            self.critic_optimizer[1].zero_grad()
            g_2_loss_Q = nn.MSELoss()(g_current_Q, g_2_target_Q.detach())
            g_2_loss_Q.backward()
            self.critic_optimizer[1].step()



            self.s_actor_optimizer[agent_index].zero_grad()
            self.g_actor_optimizer[agent_index].zero_grad()
            # state_i = state_batch[:, agent_index, :]
            # action_i = self.actors[agent_index](state_i)

            s_state = batch_state[:,:,0:2,:]
            s_state = torch.from_numpy(s_state).to(self.device).float()
            s_state = s_state.view(total_numbers_of_data*2,-1)
            g_state = batch_state[:,:,2:4,:]
            g_state = torch.from_numpy(g_state).to(self.device).float()
            g_state = g_state.view(total_numbers_of_data*2,-1)
            s_action = self.s_actor[agent_index](s_state)
            g_action = self.g_actor[agent_index](g_state)
            # # #
            # striker #
                    # #
            s_ac = one_hot.clone()# 8*8 * 7
            g_ac = one_hot.clone()
            s_action = s_action.view(-1,1,7).to(self.device)
            g_action = g_action.view(-1,1,7).to(self.device)
            # print(s_action.shape)
            s_ac = s_ac.view(-1,2,7)
            # print(s_ac.shape)
            s_ac[:, 0] = s_action.squeeze()
            g_ac = g_ac.view(-1,2,7)
            g_ac[:, 1] = g_action.squeeze()
            sup_action = s_ac.view(total_numbers_of_data, -1)
            gup_action = g_ac.view(total_numbers_of_data, -1)
            sactor_loss = -self.critic[0](whole_state, sup_action)
            sactor_loss = sactor_loss.mean()
            sactor_loss.backward()
            gactor_loss = -self.critic[1](whole_state, gup_action)
            gactor_loss = gactor_loss.mean()
            gactor_loss.backward()
            self.s_actor_optimizer[agent_index].step()
            self.g_actor_optimizer[agent_index].step()
            # # #
            # goalie #
                   # #
            c_loss.append(s_1_loss_Q + s_2_loss_Q + g_1_loss_Q + g_2_loss_Q)
            a_loss.append(sactor_loss + gactor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            soft_update(self.critic_target[0], self.critic[0], self.tau)
            soft_update(self.s_actors_target[0], self.s_actor[0], self.tau)
            soft_update(self.critic_target[1], self.critic[1], self.tau)
            soft_update(self.g_actors_target[0], self.g_actor[0], self.tau)
            
        return c_loss, a_loss

if __name__ == "__main__":
    Maddpg = Maddpg(1,2)
