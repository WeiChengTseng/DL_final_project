import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from a2c.models import A2CLarge
# from models import A2CLarge


class A2CWraper(nn.Module):
    def __init__(self, num_action, name=''):
        super().__init__()
        self._name = name
        self.policy = A2CLarge(num_action)
        # self.policy.load_state_dict(ckpt)
        return

    def act(self, states):
        policies, _ = self.policy(states)
        probs = F.softmax(policies, dim=-1)
        actions = probs.multinomial(1).data
        return actions.flatten()

    def forward(self, states):
        policies, _ = self.policy(states)
        probs = F.softmax(policies, dim=-1)
        actions = probs.multinomial(1).data
        return actions.flatten()

    def __str__(self):
        return self._name


if __name__ == '__main__':
    net_path = './ckpt_wors_2e/a2cLarge_step10920000.pth'
    # net_path = './ckpt_wors_2e/a2cLarge_step39960000.pth'
    ckpt = torch.load(net_path, map_location='cpu')
    policy_striker = A2CWraper(7)
    policy_goalie = A2CWraper(5)

    policy_striker.policy.load_state_dict(ckpt['striker_a2c'], net_path.split('/')[-1])
    policy_goalie.policy.load_state_dict(ckpt['goalie_a2c'], net_path.split('/')[-1])

    torch.save(policy_striker, net_path.split('/')[-1].replace("_", 'Striker'))
    torch.save(policy_goalie, net_path.split('/')[-1].replace("_", 'Goalie'))
