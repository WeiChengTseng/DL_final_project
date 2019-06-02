import torch as torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation = 112, dim_action = 7):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = dim_observation * n_agent
        act_dim = self.dim_action * n_agent
        self.FC = nn.Linear(obs_dim, 1024)
        self.sq = nn.Sequential(
            nn.Linear(1024+act_dim , 512),
            nn.ReLU(inplace= True),
            nn.Linear(512,300),
            nn.ReLU(inplace= True),
            nn.Linear(300,1)
        )
    def forward(self, obs, acts):
        result = F.relu(self.FC(obs))
        combined = torch.cat([result, acts], 1)
        return self.sq(combined)

class Goalie(nn.Module):
    def __init__(self, dim_observation = 112, dim_action = 5):
        super(Goalie, self).__init__()
        self.sq = nn.Sequential(
            nn.Linear(dim_observation,500),
            nn.ReLU(inplace= True),
            nn.Linear(500,128),
            nn.ReLU(inplace= True),
            nn.Linear(128,dim_action),
            nn.Tanh()
        )

    # action output between -2 and 2
    def forward(self, obs):
        output = (obs.size(0) , 7)
        buf = self.sq(obs)
        output[:,5] = buf
        return output

class Striker(nn.Module):
    def __init__(self, dim_observation = 112, dim_action = 8):
        super(Striker, self).__init__()
        self.sq = nn.Sequential(
            nn.Linear(dim_observation,500),
            nn.ReLU(inplace= True),
            nn.Linear(500,128),
            nn.ReLU(inplace= True),
            nn.Linear(128,dim_action),
            nn.Tanh()
        )

    # action output between -2 and 2
    def forward(self, obs):
        return self.sq(obs)
