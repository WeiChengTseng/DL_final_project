import torch
from torch.distributions.categorical import Categorical
import numpy as np
import sys
from mlagents.envs import UnityEnvironment
from env_exp import SocTwoEnv

input_ = np.arange(5) + 2


def net(input_):
    mapping = [4, 1, 0, 3, 2]

    mapped = input_[mapping]
    out = mapped ** 2
    re_map = np.argsort(mapping)
    return out[re_map]

print(net(input_))