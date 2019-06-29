import torch
import torch.nn as nn
import numpy as np
from maddpg_model import Critic,Goalie,Striker
from memory import Experience
from copy import deepcopy
import os
from torch.optim import Adam


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)        
        
class Maddpg:
    def __init__(self, n_striker = 1,n_goalie = 1, g_dim_act = 5,use_cuda = True,lr = 0.0001,
                dim_obs = 112, s_dim_act = 7, batchSize_d2 = 512, GAMMA = 0.99, scale_reward = 1., 
                tau = 0.01, update_timestep = 20):
        if n_striker != n_goalie:
            # winsound.Beep(800,2000)
            # os.system('shutdown -s -t 0') 
            raise EnvironmentError("GAN")
        
        self.lr = lr
        self.GAMMA = GAMMA
        self.n_striker = n_striker
        self.n_goalie = n_goalie
        self.batchSize_d2 = batchSize_d2
        self.dim_obs = dim_obs
        self.scale_reward = scale_reward
        self.tau = tau
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
        self.update_timestep = update_timestep
        self.g_dim_act = g_dim_act
        self.s_dim_act = s_dim_act
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
    def save_model(self):
        torch.save(self.s_actor)
        torch.save(self.g_actor)
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
        # print(s_b.size())
        s_act = self.s_actor[0](s_b.unsqueeze(0)).squeeze()
        g_act = self.g_actor[0](g_b.unsqueeze(0)).squeeze()

        self.steps_done += 1
        return s_act, g_act
    

    def update_policy(self, memory,step,writer):

        # do not train until exploration is enough
        # if self.episode_done <= self.episode_before_training:
            # return None, None

        c_loss = []
        a_loss = []

        for agent_index in range(self.n_striker):

            #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   # 
            # batch sample is batch * N play ground * agents * state/next_state/action/reward/      #
            #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
            
            transitions = memory.sample(self.batchSize_d2)
            batch = Experience(*zip(*transitions))

            batch_state = np.asarray(batch.states)
            # print("bs: ",batch_state)
            # print("----------")
            batch_action = torch.cat(batch.actions,dim=0)
            # print("----------")
            batch_reward = np.asarray(batch.rewards)
            # print("bw: ",batch_reward)
            # print("==========")
            batch_reward = torch.from_numpy(batch_reward).to(self.device).float()
            batch_next_state = np.asarray(batch.next_state)
            
            state_batch = torch.from_numpy(batch_state)
            action_batch = batch_action
            next_state_batch = torch.from_numpy(batch_next_state)
            
            #   #   #   #   #   #   #   #   #
            # total numbers of data = batchsize * play ground   #
                                #   #   #   #   #   #   #   #   #
            total_numbers_of_data = batch_state.shape[0]* batch_state.shape[1]
            whole_state_blue = state_batch.view(total_numbers_of_data,4, 112).to(self.device).float()
            index_order = [1,0,3,2]
            whole_state_red=whole_state_blue[:, index_order]

            # print("===============")
            # print(action_batch[0].size())

            # print("===============")
            whole_state_blue_flat  = whole_state_blue.view(total_numbers_of_data, -1)
            whole_state_red_flat   = whole_state_red.view(total_numbers_of_data, -1)

            #   #   #   #
            # translate action into one hot #
                                #   #   #   # 
            # print(action_batch.size())
            whole_action=action_batch.view(total_numbers_of_data,4,7)  
            whole_action_blue = whole_action
            whole_action_red = whole_action[:,index_order]

            whole_action_blue_flat = whole_action_blue.view(total_numbers_of_data, -1)
            whole_action_red_flat = whole_action_red.view(total_numbers_of_data, -1)

            self.critic_optimizer[0].zero_grad()
            self.critic_optimizer[1].zero_grad()
            
            s1_current_Q = self.critic[0](whole_state_blue_flat, whole_action_blue_flat.detach())
            s2_current_Q = self.critic[0](whole_state_red_flat,whole_action_red_flat.detach())
            g1_current_Q = self.critic[1](whole_state_blue_flat, whole_action_blue_flat.detach())
            g2_current_Q = self.critic[1](whole_state_red_flat, whole_action_red_flat.detach())
            
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
            whole_next_stat_blue = torch.cat([s_next_state,g_next_state], dim = -2)
            whole_next_stat_red = whole_next_stat_blue[:,index_order]

            
            s_next_actions = s_next_actions.view(-1,2,7).to(self.device)
            g_next_actions = g_next_actions.view(-1,2,7).to(self.device)
            whole_next_action_blue = torch.cat([s_next_actions,g_next_actions],dim = -2)
            whole_next_action_red = whole_next_action_blue[:,index_order]

            whole_next_stat_blue_flat = whole_next_stat_blue.view(-1,4*112)
            whole_next_stat_red_flat = whole_next_stat_red.view(-1,4*112)

            whole_next_action_blue_flat = whole_next_action_blue.view(-1,7*4)
            whole_next_action_red_flat = whole_next_action_red.view(-1,7*4)
            
            s1_target_Q = self.critic_target[0](whole_next_stat_blue_flat,whole_next_action_blue_flat.detach())
            g1_target_Q = self.critic_target[1](whole_next_stat_blue_flat,whole_next_action_blue_flat.detach())

            s2_target_Q = self.critic_target[0](whole_next_stat_red_flat,whole_next_action_red_flat.detach())
            g2_target_Q = self.critic_target[1](whole_next_stat_red_flat,whole_next_action_red_flat.detach())

            # scale_reward: to scale reward in Q functions
            batch_reward = batch_reward.view(total_numbers_of_data,-1)
            
            s_1_target_Q = (s1_target_Q * self.GAMMA) + (batch_reward[:, 0].unsqueeze(1) * self.scale_reward)
            s_2_target_Q = (s2_target_Q * self.GAMMA) + (batch_reward[:, 1].unsqueeze(1) * self.scale_reward)
            g_1_target_Q = (g1_target_Q * self.GAMMA) + (batch_reward[:, 2].unsqueeze(1) * self.scale_reward)
            g_2_target_Q = (g2_target_Q * self.GAMMA) + (batch_reward[:, 3].unsqueeze(1) * self.scale_reward)
            # 64 *1 

            # # #
            # Update first striker #
                                # # #
            s_1_loss_Q = nn.MSELoss()(s1_current_Q, s_1_target_Q.detach())
            writer.add_scalars('s_1_lossQ',{'train':(s_1_loss_Q.item())},step)
            s_1_loss_Q.backward(retain_graph = True)
            self.critic_optimizer[0].step()  
            
            # # #
            # Update 2nd striker #
                            # # #
            # print(s_2_target_Q)
            self.critic_optimizer[0].zero_grad()
            s_2_loss_Q = nn.MSELoss()(s2_current_Q, s_2_target_Q.detach())
            s_2_loss_Q.backward(retain_graph = True)
            writer.add_scalars('s_2_lossQ',{'train':(s_2_loss_Q.item())},step)
            self.critic_optimizer[0].step()
            
            # # #
            # Update first goalie #
                                # # #
            self.critic_optimizer[1].zero_grad()
            g_1_loss_Q = nn.MSELoss()(g1_current_Q, g_1_target_Q.detach())
            writer.add_scalars('g_1_lossQ',{'train':(g_1_loss_Q.item())},step)
            g_1_loss_Q.backward(retain_graph = True)
            self.critic_optimizer[1].step()

            # # #
            # Update 2nd goalie #
                            # # #
            self.critic_optimizer[1].zero_grad()
            g_2_loss_Q = nn.MSELoss()(g2_current_Q, g_2_target_Q.detach())
            writer.add_scalars('g_2_lossQ',{'train':(g_2_loss_Q.item())},step)
            g_2_loss_Q.backward()
            self.critic_optimizer[1].step()

            self.s_actor_optimizer[agent_index].zero_grad()
            self.g_actor_optimizer[agent_index].zero_grad()


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
            s_blue_actor_loss = -self.critic[0](whole_state_blue_flat, whole_action_blue_flat)
            s_blue_actor_loss = s_blue_actor_loss.mean()
            writer.add_scalars('s_blue_loss',{'train':(s_blue_actor_loss.item())},step)
            s_blue_actor_loss.backward(retain_graph=True)
            self.s_actor_optimizer[agent_index].step()

            g_blue_actor_loss = -self.critic[1](whole_state_blue_flat, whole_action_blue_flat)
            g_blue_actor_loss = g_blue_actor_loss.mean()
            writer.add_scalars('g_blue_loss',{'train':(g_blue_actor_loss.item())},step)
            g_blue_actor_loss.backward(retain_graph=True)
            self.g_actor_optimizer[agent_index].step()

            s_red_actor_loss = -self.critic[0](whole_state_red_flat, whole_action_red_flat)
            s_red_actor_loss = s_red_actor_loss.mean()
            writer.add_scalars('s_red_loss',{'train':(s_red_actor_loss.item())},step)
            s_red_actor_loss.backward(retain_graph=True)
            self.s_actor_optimizer[agent_index].step()

            g_red_actor_loss = -self.critic[1](whole_state_red_flat, whole_action_red_flat)
            g_red_actor_loss = g_red_actor_loss.mean()
            writer.add_scalars('g_red_loss',{'train':(g_red_actor_loss.item())},step)
            g_red_actor_loss.backward(retain_graph=True)
            self.g_actor_optimizer[agent_index].step()


            # # #
            # goalie #
                   # #
            c_loss.append(s_1_loss_Q+g_1_loss_Q)
            a_loss.append(s_red_actor_loss + g_red_actor_loss)
            writer.add_scalars('loss_a',{'train':((s_red_actor_loss + g_red_actor_loss).item())},step)
            writer.add_scalars('loss_c',{'train':(s_1_loss_Q+g_1_loss_Q)},step)


        if self.steps_done % self.update_timestep == 0 and self.steps_done > 0:
            soft_update(self.critic_target[0], self.critic[0], self.tau)
            soft_update(self.s_actors_target[0], self.s_actor[0], self.tau)
            soft_update(self.critic_target[1], self.critic[1], self.tau)
            soft_update(self.g_actors_target[0], self.g_actor[0], self.tau)
            
        return c_loss, a_loss

if __name__ == "__main__":
    Maddpg = Maddpg(1,2)