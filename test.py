import torch
from torch.distributions.categorical import Categorical

x = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
dist = Categorical(x)
print(dist.sample())
