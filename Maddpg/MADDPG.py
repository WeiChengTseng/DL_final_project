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
                dim_obs = 112, s_dim_act = 7, batchSize_d2 = 8, episode_before_training = 0, GAMMA = 1., scale_reward = 1.):
        if n_striker != n_goalie:
            winsound.Beep(800,2000)
            os.system('shutdown -s -t 0') 
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
        self.critic = [Critic(4,self.dim_obs,self.s_dim_act).to(self.device) for i in range(self.n_striker)]
        self.g_actor = [Goalie(self.dim_obs,self.g_dim_act).to(self.device) for i in range(self.n_goalie)]
        # self.g_critic = [Critic(4,self.dim_obs,self.g_dim_act).to(self.device) for i in range(self.n_goalie)]
       
        self.s_actors_target = deepcopy(self.s_actor)
        self.g_actors_target = deepcopy(self.g_actor)
        self.critic_target = deepcopy(self.critic)
        # self.s_actors_target[0] = self.s_actors_target[0].to(self.device)
        # self.g_actors_target[0] = self.g_actors_target[0].to(self.device)
        
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=lr) for x in self.critic]
        self.s_actor_optimizer = [Adam(x.parameters(),
                                     lr=lr) for x in self.s_actor]
        # self.g_critic_optimizer = [Adam(x.parameters(),
                                    #   lr=lr) for x in self.g_critic]
        self.g_actor_optimizer = [Adam(x.parameters(),
                                     lr=lr) for x in self.g_actor]
        
        
        self.steps_done = 0
        self.episode_done = 0
    def select_action(self, striker_batch, goalie_batch):
        # state_batch: n_agents x state_dim
        s_actions = torch.zeros(self.n_striker,self.s_dim_act).to(self.device)
        g_actions = torch.zeros(self.n_goalie,self.g_dim_act).to(self.device)
        striker_batch = torch.from_numpy(striker_batch)
        goalie_batch = torch.from_numpy(goalie_batch)
        striker_batch = striker_batch.to(self.device)
        goalie_batch = goalie_batch.to(self.device)
        # FloatTensor = torch.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_striker):
            s_b = striker_batch[i, :].detach().float()
            g_b = goalie_batch[i, :].detach().float()
            self.s_actor[i] = self.s_actor[i].to(self.device)
            self.g_actor[i] = self.g_actor[i].to(self.device)
            s_act = self.s_actor[i](s_b.unsqueeze(0)).squeeze()
            g_act = self.g_actor[i](g_b.unsqueeze(0)).squeeze()

            s_actions[i, :] = s_act
            g_actions[i, :] = g_act
        self.steps_done += 1

        return s_actions, g_actions

    def update_policy(self, memory):

        # momory format is memory.push(prev_state, states, [prev_action_striker, prev_action_goalie], prev_reward)
        # do not train until exploration is enough
        # if self.episode_done <= self.episode_before_training:
        #     return None, None

        c_loss = []
        a_loss = []
        # print(len(memory))
            
        if len(memory) < 10:
            return
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
            # print(batch.next_state)
            batch_next_state = np.asarray(batch.next_state)
            
            state_batch = torch.from_numpy(batch_state)
            action_batch = torch.from_numpy(batch_action)
            next_state_batch = torch.from_numpy(batch_next_state)

            total_numbers_of_data = batch_state.shape[0]* batch_state.shape[1]
            whole_state = state_batch.view(total_numbers_of_data, -1).to(self.device).float()
            whole_action = action_batch.view(total_numbers_of_data * 4,-1).long()
            
            one_hot = (whole_action == torch.arange(7).reshape(1,7)).float()
            one_hot = one_hot.view(total_numbers_of_data , -1).to(self.device)
            # print(one_hot.shape)
             
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # PADDING ZERO                                            #
            # If enivronment > 5 -> 0                                 #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            self.critic_optimizer[agent_index].zero_grad()
            current_Q = self.critic[agent_index](whole_state, one_hot)
            # print(current_Q.shape)
            # print(current_Q)
            # non_final_next_actions = [
            #     self.actors_target[i](non_final_next_states[:,i,:])
            #                             for i in range(self.n_agents)]
            # non_final_next_actions = torch.stack(non_final_next_actions)
            # non_final_next_actions = (
            #     non_final_next_actions.transpose(0,1).contiguous())
            s_whole_next_state = next_state_batch[:,:,0:2,:].to(self.device).float()
            s_whole_next_state = s_whole_next_state.view(total_numbers_of_data*2,-1)
            
            g_whole_next_state = next_state_batch[:,:,2:4,:].to(self.device).float()
            g_whole_next_state = g_whole_next_state.view(total_numbers_of_data*2,-1)
            # # #
            # Next_actions  #
                        # # #
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
            
            # exit()
            target_Q = torch.zeros(self.batchSize_d2)
            whole_next_stat = whole_next_stat.view(-1,112*4)
            whole_next_action = whole_next_action.view(-1,7*4)
            
            target_Q = self.critic_target[agent_index](whole_next_stat,whole_next_action)
        #     # scale_reward: to scale reward in Q functions
            # print(target_Q.shape) # 64 * 1
            batch_reward = batch_reward.view(64,-1)
            # print(batch_reward[:,0].shape)
            # 64*4
            target_Q = (target_Q * self.GAMMA) + (batch_reward[:, 0].unsqueeze(1) * self.scale_reward)
            # 64 *1 
            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent_index].step()

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
            sactor_loss = -self.critic[agent_index](whole_state, sup_action)
            sactor_loss = sactor_loss.mean()
            sactor_loss.backward()
            gactor_loss = -self.critic[agent_index](whole_state, gup_action)
            gactor_loss = gactor_loss.mean()
            gactor_loss.backward()
            self.s_actor_optimizer[agent_index].step()
            self.g_actor_optimizer[agent_index].step()
            # # #
            # goalie #
                   # #
            c_loss.append(loss_Q)
            a_loss.append(sactor_loss + gactor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_striker):
                soft_update(self.s_critic_target[i], self.s_critic[i], self.tau)
                soft_update(self.s_actors_target[i], self.s_actor[i], self.tau)
            for i in range(self.n_goalie):
                soft_update(self.g_critics_target[i], self.g_critic[i], self.tau)
                soft_update(self.g_actors_target[i], self.g_actor[i], self.tau)
            
        return c_loss, a_loss

if __name__ == "__main__":
    Maddpg = Maddpg(1,2)
