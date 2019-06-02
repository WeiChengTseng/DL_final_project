import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from a2c.models import AtariCNN, A2C
from a2c.envs import make_env, RenderSubprocVecEnv
from a2c.train_multi import train

from env_exp import SocTwoEnv


def eval_with_random_agent(net_striker,
                           net_goalie,
                           env,
                           device,
                           eval_epsoid=40):
    obs_striker, obs_goalie = env.reset('team')
    epsoid = 0
    while epsoid < eval_epsoid:
        obs_striker = Variable(
            torch.from_numpy(obs_striker).float()).to(device)
        obs_goalie = Variable(torch.from_numpy(obs_goalie).float()).to(device)

        policies_striker, values_striker = net_striker(obs_striker)
        policies_goalie, values_goalie = net_goalie(obs_goalie)

        probs_striker = F.softmax(policies_striker, dim=-1)
        probs_goalie = F.softmax(policies_goalie, dim=-1)

        actions_striker = probs_striker.multinomial(1).data
        actions_goalie = probs_goalie.multinomial(1).data

        # gather env data, reset done envs and update their obs
        actions_striker = torch.cat([
            actions_striker[:8],
            torch.LongTensor(np.random.randint(0, 7, (8, 1)))
        ],
                                    dim=0)
        actions_goalie = torch.cat([
            actions_goalie[:8],
            torch.LongTensor(np.random.randint(0, 7, (8, 1)))
        ],
                                   dim=0)

        # print(actions_striker)
        obs, rewards, dones, _ = env.step(actions_striker, actions_goalie,
                                          'team')
        obs_striker, obs_goalie = obs

        rewards_striker = torch.from_numpy(
            rewards[0]).float().unsqueeze(1).to(device)
        rewards_goalie = torch.from_numpy(
            rewards[1]).float().unsqueeze(1).to(device)

        for i in np.argwhere(dones[0]).flatten():
            epsoid += 1
            pass
    return


if __name__ == '__main__':
    env_path = './env/macos/SoccerTwosFast.app'
    env = SocTwoEnv(env_path, worker_id=0, train_mode=False, render=True)
    # net_path = './a2c/ckpt/a2c_step20320000.pth'
    net_path = './a2c/ckpt_reward_shaping/a2c_step400000.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    policy_striker, policy_goalie = A2C(7).to(device), A2C(5).to(device)

    checkpoint = torch.load(net_path, map_location=device)
    policy_striker.load_state_dict(checkpoint['striker_a2c'])
    policy_goalie.load_state_dict(checkpoint['goalie_a2c'])

    policy_striker.eval()
    policy_goalie.eval()
    eval_with_random_agent(policy_striker, policy_goalie, env, device)
    pass