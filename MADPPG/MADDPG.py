import torch
import torch.nn as nn
from maddpg_model import Goalie_Critic,Striker_Critic,Goalie,Striker
# from memory import Memory , Experience
import os
from torch.optim import Adam
import winsound
class Maddpg:
    def __init__(self, n_striker = 2,n_goalie = 2, g_dim_act = 5,use_cuda = True,lr = 0.0001,
                dim_obs = 112, s_dim_act = 8, batchSize = 1024, episode_before_training = 0):
        if n_striker != n_goalie:
            winsound.Beep(800,2000)
            os.system('shutdown -s -t 0') 
            raise EnvironmentError("GAN")
            
        self.lr = lr
        self.n_striker = n_striker
        self.n_goalie = n_goalie
        self.batchSize = batchSize
        self.dim_obs = dim_obs
        self.g_dim_act = g_dim_act
        self.s_dim_act = s_dim_act
        self.episode_before_training = episode_before_training
        self.s_actor = [Striker(self.dim_obs,self.s_dim_act) for i in range(self.n_striker)]
        self.s_critic = [Striker_Critic(2,self.dim_obs,self.s_dim_act) for i in range(self.n_striker)]
        self.g_actor = [Goalie(self.dim_obs,self.g_dim_act) for i in range(self.n_goalie)]
        self.g_critic = [Goalie_Critic(2,self.dim_obs,self.g_dim_act) for i in range(self.n_goalie)]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
        self.s_critic_optimizer = [Adam(x.parameters(),
                                      lr=lr) for x in self.s_critic]
        self.s_actor_optimizer = [Adam(x.parameters(),
                                     lr=lr) for x in self.s_actor]
        self.g_critic_optimizer = [Adam(x.parameters(),
                                      lr=lr) for x in self.g_critic]
        self.g_actor_optimizer = [Adam(x.parameters(),
                                     lr=lr) for x in self.g_actor]
        
        
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

    def update_policy(self, striker_memory, goalie_memory):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []
        for agent_index in range(self.n_striker):
            s_transitions = striker_memory.sample(self.batchSize)
            g_transitions = goalie_memory.sample(self.batchSize)
            s_batch = Experience(*zip(*s_stransitions))
            g_batch = Experience(*zip(*s_stransitions))
            
            s_non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                    s_batch.next_states)))
            g_non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                    g_batch.next_states)))
            
            # state_batch: batch_size x n_agents x dim_obs
            s_state_batch = torch.stack(s_batch.states).type(FloatTensor)
            s_action_batch = torch.stack(s_batch.actions).type(FloatTensor)
            s_reward_batch = torch.stack(s_batch.rewards).type(FloatTensor)
            g_state_batch = torch.stack(g_batch.states).type(FloatTensor)
            g_action_batch = torch.stack(g_batch.actions).type(FloatTensor)
            g_reward_batch = torch.stack(g_batch.rewards).type(FloatTensor)
            
            # : (batch_size_non_final) x n_agents x dim_obs
            s_non_final_next_states = torch.stack(
                [s for s in s_batch.next_states
                 if s is not None]).type(FloatTensor)
            g_non_final_next_states = torch.stack(
                [s for s in g_batch.next_states
                 if s is not None]).type(FloatTensor)
            
            # for current agent
            s_whole_state = s_state_batch.view(self.batchSize, -1)
            print(s_whole_state.shape)
            s_whole_action = s_action_batch.view(self.batchSize, -1)
            print(s_whole_action.shape)
            g_whole_state = g_state_batch.view(self.batchSize, -1)
            print(g_whole_state.shape)
            g_whole_action = g_action_batch.view(self.batchSize, -1)
            print(g_whole_action.shape)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # we need a discussion to define the meaning of act_dim   #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            self.s_critic_optimizer[agent_index].zero_grad()
            self.g_critic_optimizer[agent_index].zero_grad()
        #     current_Q = self.critics[agent_index](whole_state, whole_action)

        #     non_final_next_actions = [
        #         self.actors_target[i](non_final_next_states[:,
        #                                                     i,
        #                                                     :]) for i in range(
        #                                                         self.n_agents)]
        #     non_final_next_actions = th.stack(non_final_next_actions)
        #     non_final_next_actions = (
        #         non_final_next_actions.transpose(0,
        #                                          1).contiguous())

        #     target_Q = th.zeros(
        #         self.batch_size).type(FloatTensor)

        #     target_Q[non_final_mask] = self.critics_target[agent](
        #         non_final_next_states.view(-1, self.n_agents * self.n_states),
        #         non_final_next_actions.view(-1,
        #                                     self.n_agents * self.n_actions)
        #     ).squeeze()
        #     # scale_reward: to scale reward in Q functions

        #     target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
        #         reward_batch[:, agent].unsqueeze(1) * scale_reward)

        #     loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
        #     loss_Q.backward()
        #     self.critic_optimizer[agent].step()

        #     self.actor_optimizer[agent].zero_grad()
        #     state_i = state_batch[:, agent, :]
        #     action_i = self.actors[agent](state_i)
        #     ac = action_batch.clone()
        #     ac[:, agent, :] = action_i
        #     whole_action = ac.view(self.batch_size, -1)
        #     actor_loss = -self.critics[agent](whole_state, whole_action)
        #     actor_loss = actor_loss.mean()
        #     actor_loss.backward()
        #     self.actor_optimizer[agent].step()
        #     c_loss.append(loss_Q)
        #     a_loss.append(actor_loss)

        # if self.steps_done % 100 == 0 and self.steps_done > 0:
        #     for i in range(self.n_agents):
        #         soft_update(self.critics_target[i], self.critics[i], self.tau)
        #         soft_update(self.actors_target[i], self.actors[i], self.tau)

        # return c_loss, a_loss

if __name__ == "__main__":
    Maddpg = Maddpg(1,2)