import torch as torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, ns_agent,ng_agent, dim_observation = 112, s_dim_action = 7,g_dim_action=5):
        super(Critic, self).__init__()
        self.n_agent = ns_agent+ng_agent
        self.dim_observation = dim_observation
        obs_dim = dim_observation * self.n_agent
        act_dim = s_dim_action * ns_agent + g_dim_action *ng_agent
        self.FC = nn.Linear(obs_dim, 256)
        self.sq = nn.Sequential(
            nn.Linear(256+act_dim , 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
        )
    def forward(self, obs, acts):
        result = F.relu(self.FC(obs))
        
        combined = torch.cat([result, acts], 1)
        return self.sq(combined)

class Goalie(nn.Module):
    def __init__(self, dim_observation = 112, dim_action = 5):
        super(Goalie, self).__init__()
        self.sq = nn.Sequential(
            nn.Linear(dim_observation,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,dim_action),
            nn.LogSoftmax(-1)
        )

    # action output between -2 and 2
    def forward(self, obs):
        buf = self.sq(obs)
        buf = buf.squeeze()
        
        return F.gumbel_softmax(buf,tau=5,dim=-1)



class Striker(nn.Module):
    def __init__(self, dim_observation = 112, dim_action = 7):
        super(Striker, self).__init__()
        self.sq = nn.Sequential(
            nn.Linear(dim_observation,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,dim_action),
            nn.LogSoftmax(-1)
        )

    # action output between -2 and 2
    def forward(self, obs):
        return F.gumbel_softmax(self.sq(obs),tau=5,dim=-1)

