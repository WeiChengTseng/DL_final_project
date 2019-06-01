import torch
from torch.distributions.categorical import Categorical
import numpy as np
import sys
from mlagents.envs import UnityEnvironment

x = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
dist = Categorical(x)
print(dist.sample())
env_path = './env/macos/SoccerTwosLearnerBirdView.app'
env = UnityEnvironment(file_name=env_path, worker_id=0)
striker_brain_name, goalie_brain_name = env.brain_names
striker_brain = env.brains[striker_brain_name]
goalie_brain = env.brains[goalie_brain_name]
env.reset(train_mode=True)
env_info = env.step(
    {
        striker_brain_name: [0] * 16,
        goalie_brain_name: [0] * 16
    },
    memory={
        striker_brain_name: [0] * 16,
        goalie_brain_name: [0] * 16
    }
)

print(env_info[striker_brain_name].memories)