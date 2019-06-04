import time
import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from a2c.agent_wraper import A2CWraper
from a2c.models import A2CLarge


def eval_agents_compete(strikers,
                        goalies,
                        env,
                        device,
                        order='team',
                        eval_epsoid=40):
    obs_striker, obs_goalie = env.reset(order)
    actions_strikers = [None, None]
    actions_goalies = [None, None]
    records = [0, 0, 0]

    epsoid = 0
    while epsoid < eval_epsoid:
        obs_striker = Variable(
            torch.from_numpy(obs_striker).float()).to(device)
        obs_goalie = Variable(torch.from_numpy(obs_goalie).float()).to(device)

        actions_strikers[0], _ = strikers[0](obs_striker[:8])
        actions_goalies[0], _ = goalies[0](obs_goalie[:8])
        actions_strikers[1], _ = strikers[1](obs_striker[8:])
        actions_goalies[1], _ = goalies[1](obs_goalie[8:])

        actions_striker = torch.cat(actions_strikers, 0)
        actions_goalie = torch.cat(actions_goalies, 0)

        obs, rewards, dones, _ = env.step(actions_striker, actions_goalie,
                                          order)
        obs_striker, obs_goalie = obs

        for i in np.argwhere(dones[0]).flatten():
            epsoid += 1
            if rewards[1][i]<0:
                records[0]+=1
            elif rewards[0][i]<0:
                records[1]+=1
            else:
                records[2]+=1
    
    return


if __name__ == '__main__':
    s1 = './a2c/a2cLargeStrikerstep39960000.pth'
    g1 = './a2c/a2cLargeGoaliestep39960000.pth'
    s2 = './a2c/a2cLargeStrikerstep10920000.pth'
    g2 = './a2c/a2cLargeGoaliestep10920000.pth'

    a = A2CWraper(5)
    a = torch.load(s1)
    # strikers = [torch.load(s1), torch.load(s2)]
    # goalies = [torch.load(g1), torch.load(g2)]
    pass