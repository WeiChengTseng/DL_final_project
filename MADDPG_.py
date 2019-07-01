import torch
import torch.nn as nn
import numpy as np
from maddpg_model import Critic,Goalie,Striker
# from memory import Experience
from copy import deepcopy
import os
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)  

def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)

class Maddpg:
    def __init__(self, n_striker = 2,n_goalie = 2, g_dim_act = 5,use_cuda = True,lr = 0.0001,
                dim_obs = 112, s_dim_act = 7, batchSize_d2 = 512, GAMMA = 0.99, scale_reward = 1., 
                tau = 0.01, update_timestep = 20, field=8, lambda_scale= 0.01):
        if n_striker != n_goalie:
            raise EnvironmentError("Game should have the same teammates")
        
        self.lr = lr
        self.Gamma = GAMMA
        self.n_striker = n_striker
        self.n_goalie = n_goalie
        self.batchSize = batchSize_d2
        self.dim_obs = dim_obs
        self.scale_reward = scale_reward
        self.tau = tau 
        self.field = field
        self.g_dim_act =g_dim_act
        self.s_dim_act =s_dim_act
        self.lambda_scale = lambda_scale
        self.update_target = 10
        self.update_timestep = update_timestep

        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")  
        
        self.s_actor = [ Striker(self.dim_obs, self.s_dim_act).to(self.device) for i in range(self.n_striker)]
        self.s_actor_optimizer = [Adam(x.parameters(),
                                        lr=lr) for x in self.s_actor]
        self.s_actor_other = [ [Striker(self.dim_obs, self.s_dim_act).to(self.device),Goalie(self.dim_obs, self.g_dim_act).to(self.device),Goalie(self.dim_obs, self.g_dim_act).to(self.device)] for i in range(self.n_striker)]
        self.s_actor_other_target = [ [Striker(self.dim_obs, self.s_dim_act).to(self.device),Goalie(self.dim_obs, self.g_dim_act).to(self.device),Goalie(self.dim_obs, self.g_dim_act).to(self.device)] for i in range(self.n_striker)]
        self.s_actor_other_optimizer = [ [Adam(network.parameters(),lr=lr) for network in x] for x in self.s_actor_other]
        
        self.g_actor = [ Goalie(self.dim_obs, self.g_dim_act).to(self.device)  for i in range(self.n_goalie)]
        self.g_actor_optimizer = [Adam(x.parameters(),
                                        lr=lr) for x in self.g_actor]
        self.g_actor_other = [ [Goalie(self.dim_obs, self.g_dim_act).to(self.device),Striker(self.dim_obs, self.s_dim_act).to(self.device),Striker(self.dim_obs, self.s_dim_act).to(self.device)]  for i in range(self.n_goalie)]
        self.g_actor_other_target = [ [Goalie(self.dim_obs, self.g_dim_act).to(self.device),Striker(self.dim_obs, self.s_dim_act).to(self.device),Striker(self.dim_obs, self.s_dim_act).to(self.device)]  for i in range(self.n_goalie)]
        self.g_actor_other_optimizer = [ [Adam(network.parameters(),lr=lr) for network in x] for x in self.g_actor_other]

        self.s_actor_target = [ Striker(self.dim_obs, self.s_dim_act).to(self.device) for i in range(self.n_striker)]
        self.g_actor_target = [ Goalie(self.dim_obs, self.g_dim_act).to(self.device)  for i in range(self.n_goalie)]

        self.s_critic  = [ Critic(4, self.dim_obs, self.s_dim_act).to(self.device) for i in range(self.n_striker)]
        self.s_critic_optimizer = [Adam(x.parameters(),
                                        lr=lr*5) for x in self.s_critic]

        self.g_critic  = [ Critic(4, self.dim_obs, self.s_dim_act).to(self.device) for i in range(self.n_goalie)]
        self.g_critic_optimizer = [Adam(x.parameters(),
                                        lr=lr*5) for x in self.g_critic]

        self.s_critic_target  = [ Critic(4, self.dim_obs, self.s_dim_act).to(self.device) for i in range(self.n_striker)]
        self.g_critic_target  = [ Critic(4, self.dim_obs, self.s_dim_act).to(self.device) for i in range(self.n_goalie)]
        
        self.steps_done = 0
        self.episode_done = 0

    def save_model(self,folder,episode):
        # saving all models
        PATH = folder
        if not os.path.exists(PATH+"/episode_{}".format(episode)):
            os.makedirs(PATH+"/episode_{}".format(episode))
        cur_path=PATH+"/episode_{}".format(episode)

        for i in range(self.n_striker):
            torch.save(self.s_actor[i].state_dict(), cur_path+"/striker_actor_{}.pth".format(i))
            torch.save(self.s_critic[i].state_dict(),cur_path+"/striker_critic_{}.pth".format(i))
            torch.save(self.s_actor_target[i].state_dict(), cur_path+"/striker_actor_target_{}.pth".format(i))
            torch.save(self.s_critic_target[i].state_dict(),cur_path+"/striker_critic_target_{}.pth".format(i))
            for j in range(self.n_striker+self.n_goalie-1):
                torch.save(self.s_actor_other[i],cur_path+"/striker_{}_others_{}.pth".format(i,j))
        for i in range(self.n_goalie):
            torch.save(self.g_actor[i].state_dict(), cur_path+"/goalie_actor_{}.pth".format(i))
            torch.save(self.g_critic[i].state_dict(),cur_path+"/goalie_critic_{}.pth".format(i))
            torch.save(self.g_actor_target[i].state_dict(), cur_path+"/goalie_actor_target_{}.pth".format(i))
            torch.save(self.g_critic_target[i].state_dict(),cur_path+"/goalie_critic_target_{}.pth".format(i))
            for j in range(self.n_striker+self.n_goalie-1):
                torch.save(self.g_actor_other[i],cur_path+"/goalie_{}_others_{}.pth".format(i,j))
        return
    
    def select_action_train(self,strikers,goalies):
        s_act_distr = torch.zeros([16,7])
        g_act_distr = torch.zeros([16,7])
        odd = [0,2,4,6,8,10,12,14]
        even = [1,3,5,7,9,11,13,15]
        s_act_distr[odd]=(self.s_actor[0](torch.from_numpy(strikers[odd]).float())).clone()
        g_act_distr[odd]=(self.g_actor[0](torch.from_numpy(goalies[odd]).float())).clone()
        s_act_distr[even]=(self.s_actor[1](torch.from_numpy(strikers[even]).float())).clone()
        g_act_distr[even]=(self.g_actor[1](torch.from_numpy(goalies[even]).float())).clone()
        
        return s_act_distr, g_act_distr
    
    def select_action_test(self,strikers,goalies,team):
        s_act = np.zeros([16])
        g_act = np.zeros([16])
        even = [0,2,4,6,8,10,12,14]
        odd = [1,3,5,7,9,11,13,15]
        if team == "red":
            s_act[even]=torch.argmax(F.gumbel_softmax(self.s_actor[1](torch.from_numpy(strikers[even]).float()),-1),1).detach().numpy()
            g_act[even]=torch.argmax(F.gumbel_softmax(self.g_actor[1](torch.from_numpy(goalies[even]).float()),-1),1).detach().numpy()
            s_act[odd]= np.random.randint(7, size=8)
            g_act[odd]= np.random.randint(5, size=8)
        else:
            s_act[odd]=torch.argmax(F.gumbel_softmax(self.s_actor[0](torch.from_numpy(strikers[odd]).float()),-1),1).detach().numpy()
            g_act[odd]=torch.argmax(F.gumbel_softmax(self.g_actor[0](torch.from_numpy(goalies[odd]).float()),-1),1).detach().numpy()
            s_act[even]= np.random.randint(7, size=8)
            g_act[even]= np.random.randint(5, size=8)
        
        return s_act, g_act
    
    def update_policy(self, memory, step, writer,seed):

        # do not train until exploration is enough
        ##########################################
        
        c_loss = []
        a_loss = []

        self.steps_done += 1

        for agent_index in range(self.n_striker):
            if agent_index==0:
                index_order = [0,1,2,3] #(0, 1 -> striker),
                                        #(2 ,3 -> goalie)
            else:
                index_order = [1,0,3,2]

            # alert me that action need to be log()
            states, actions, probs, rewards, next_states = memory.sample(index_order,seed,self.batchSize)
            
            ###################################################
            # update critic
            for i in range(3):
                probs_stack= torch.stack(probs[i+1]).detach()
                output=self.s_actor_other[agent_index][i](torch.from_numpy(np.stack(states[i+1],axis=0)).float()).clone()
                dist_1 = Categorical(logits=output)
                dist_1_entropy = dist_1.entropy()
                self.s_actor_other_optimizer[agent_index][i].zero_grad()
                criterion = torch.nn.KLDivLoss(reduction="sum")
                loss_entropy_1 = criterion(F.gumbel_softmax(output,-1).log(),probs_stack)+self.lambda_scale*dist_1_entropy
                writer.add_scalars("s_actor_other_{}_number_{}".format(agent_index, i),{"loss":(loss_entropy_1.mean().item())},step)
                loss_entropy_1.mean().backward()
                self.s_actor_other_optimizer[agent_index][i].step()
                    
            output_itself = self.s_actor_target[agent_index](torch.from_numpy(np.stack(next_states[0],axis=0)).float())
            output = torch.zeros((output_itself.size(0),3,7))
            output[:,0] = self.s_actor_other_target[agent_index][0](torch.from_numpy(np.stack(next_states[1],axis=0)).float()).clone()
            output[:,1] = self.s_actor_other_target[agent_index][1](torch.from_numpy(np.stack(next_states[2],axis=0)).float()).clone()
            output[:,2] = self.s_actor_other_target[agent_index][2](torch.from_numpy(np.stack(next_states[3],axis=0)).float()).clone()
            critic_actions = torch.stack((output_itself.detach(),output[:,0].detach(),output[:,1].detach(),output[:,2].detach()),dim= 1).float().reshape(self.batchSize, -1)
            next_states_stack = torch.from_numpy(np.stack(next_states,axis=1)).float().reshape(self.batchSize, -1)
            target_Q_p=self.s_critic_target[agent_index](next_states_stack,critic_actions)
            states_stack = torch.from_numpy(np.stack(states,axis=0)).reshape(self.batchSize, -1).float().detach()
            probs_stack= torch.stack((torch.stack(probs[0]).detach(),torch.stack(probs[1]).detach(),torch.stack(probs[2]).detach(),torch.stack(probs[3]).detach())).reshape(self.batchSize, -1).float().detach().log()
            probs_stack_update= torch.stack((torch.stack(probs[0]),torch.stack(probs[1]).detach(),torch.stack(probs[2]).detach(),torch.stack(probs[3]).detach())).reshape(self.batchSize, -1).float().log()
            s_current_Q = self.s_critic[agent_index](states_stack,probs_stack)
            s_target_Q = target_Q_p.detach() * self.Gamma + torch.from_numpy(np.stack(rewards[0],axis=0) * self.scale_reward).unsqueeze_(-1).float()

            s_critic_loss_Q = nn.MSELoss()(s_current_Q, s_target_Q)
            writer.add_scalars("striker_{}".format(agent_index),{"critic":(s_critic_loss_Q.item())},step)

            ###################################################
            # update inferring policy of other agents
            # need to fix
            
            ###################################################
            self.s_critic_optimizer[agent_index].zero_grad()
            s_critic_loss_Q.backward()
            self.s_critic_optimizer[agent_index].step()
            ###################################################

            ###################################################
            # update actor policy
            s_actor_loss = -self.s_critic[agent_index](states_stack, probs_stack_update)
            self.s_actor_optimizer[agent_index].zero_grad()
            s_actor_loss =s_actor_loss.mean()
            s_actor_loss.backward(retain_graph=True)
            self.s_actor_optimizer[agent_index].step()
            ###################################################

            writer.add_scalars('striker_{}'.format(agent_index),{'actor':s_actor_loss},step)
            del states[:], actions[:], probs[:], rewards[:], next_states[:]
            ###################################################
        for agent_index in range(self.n_goalie):
            if agent_index==0:
                index_order = [2,3,0,1] #(0, 1 -> striker),
                                        #(2 ,3 -> goalie)
            else:
                index_order = [3,2,1,0]

            states, actions, probs, rewards, next_states = memory.sample(index_order,seed,self.batchSize)

            ###################################################
            # update critic
            
            
            for i in range(3):
                probs_stack= torch.stack(probs[i+1]).detach()
                output =self.g_actor_other[agent_index][i](torch.from_numpy(np.stack(states[i+1],axis=0)).float()).clone()
                wanted=F.gumbel_softmax(output,-1)
                dist_1 = Categorical(probs=wanted)
                dist_1_entropy = dist_1.entropy()
                self.g_actor_other_optimizer[agent_index][i].zero_grad()
                criterion = torch.nn.KLDivLoss(reduction="sum")
                loss_entropy_1 = criterion(wanted.log(), probs_stack)+self.lambda_scale*dist_1_entropy
                writer.add_scalars("g_actor_other_{}_number_{}".format(agent_index, i),{"loss":(loss_entropy_1.mean().item())},step)
                loss_entropy_1.mean().backward()
                self.g_actor_other_optimizer[agent_index][i].step() 
                    

            output_itself = self.g_actor_target[agent_index](torch.from_numpy(np.stack(next_states[0],axis=0)).float())
            output = torch.zeros((output_itself.size(0),3,7))
            output[:,0] = self.g_actor_other_target[agent_index][0](torch.from_numpy(np.stack(next_states[1],axis=0)).float()).clone()
            output[:,1] = self.g_actor_other_target[agent_index][1](torch.from_numpy(np.stack(next_states[2],axis=0)).float()).clone()
            output[:,2] = self.g_actor_other_target[agent_index][2](torch.from_numpy(np.stack(next_states[3],axis=0)).float()).clone()
            
            critic_actions = torch.stack((output_itself.detach(),output[:,0].detach(),output[:,1].detach(),output[:,2].detach()),dim= 1).float().reshape(self.batchSize, -1)
            next_states_stack = torch.from_numpy(np.stack(next_states,axis=1)).float().reshape(self.batchSize, -1)
            target_Q_p=self.g_critic_target[agent_index](next_states_stack,critic_actions)
            states_stack = torch.from_numpy(np.stack(states,axis=0)).reshape(self.batchSize, -1).float()
            probs_stack= torch.stack((torch.stack(probs[0]).detach(),torch.stack(probs[1]).detach(),torch.stack(probs[2]).detach(),torch.stack(probs[3]).detach())).reshape(self.batchSize, -1).float().log()
            probs_stack_update= torch.stack((torch.stack(probs[0]),torch.stack(probs[1]).detach(),torch.stack(probs[2]).detach(),torch.stack(probs[3]).detach())).reshape(self.batchSize, -1).float().log()
            
            g_current_Q = self.g_critic[agent_index](states_stack,probs_stack)
            g_target_Q = target_Q_p.detach() * self.Gamma + torch.from_numpy(np.stack(rewards[0],axis=0) * self.scale_reward).unsqueeze_(-1).float()
            g_critic_loss_Q = nn.MSELoss()(g_current_Q, g_target_Q)
            writer.add_scalars("goalie_{}".format(agent_index),{"critic":(g_critic_loss_Q.item())},step)

            ###################################################
            # update inferring policy of other agents
            # need to fix
            
            
            ###################################################
            self.g_critic_optimizer[agent_index].zero_grad()
            g_critic_loss_Q.backward()
            self.g_critic_optimizer[agent_index].step()
            ###################################################

            ###################################################
            # update actor policy
            self.g_actor_optimizer[agent_index].zero_grad()
            g_actor_loss = -self.g_critic[agent_index](states_stack,  probs_stack_update)
            g_actor_loss.mean().backward(retain_graph=True)
            self.g_actor_optimizer[agent_index].step()
            ###################################################

            writer.add_scalars('goalie_{}'.format(agent_index),{'actor':(g_actor_loss.mean())},step)
            del states[:], actions[:], probs[:], rewards[:], next_states[:]
            ###################################################
        # soft update targetNetwork
        if self.steps_done  % self.update_target == 0 and self.steps_done > 0:
            soft_update(self.g_critic_target[agent_index], self.g_critic[agent_index], self.tau)
            soft_update(self.g_actor_target[agent_index], self.g_actor[agent_index], self.tau)
            soft_update(self.s_critic_target[agent_index], self.s_critic[agent_index], self.tau)
            soft_update(self.s_actor_target[agent_index],  self.s_actor[agent_index], self.tau)  
            for i in range(3):
                soft_update(self.g_actor_other_target[agent_index][i], self.g_actor_other[agent_index][i], self.tau)
                soft_update(self.s_actor_other_target[agent_index][i], self.s_actor_other[agent_index][i], self.tau)  
                
            



            
            
            








            

            



            











        
        
            






        

    