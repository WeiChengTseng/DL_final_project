import matplotlib.pyplot as plt
import numpy as np
import sys
from gym_unity.envs import UnityEnv
from mlagents.envs import UnityEnvironment

env_Path = 'env/soccer_env_different_cams.x86_64'
train_mode = True
env = UnityEnvironment(env_Path, worker_id=1)

striker_brain_name, goalie_brain_name = env.brain_names

striker_brain = env.brains[striker_brain_name]
goalie_brain = env.brains[goalie_brain_name]

env_info = env.reset(train_mode=train_mode)[striker_brain_name]

print("Agent state looks like: \n{}".format(env_info.vector_observations[0]))
print(env_info.vector_observations.shape)
for observation in env_info.vector_observations:
    print("Agent observations look like:")
    if observation.shape[3] == 3:
        plt.imshow(observation[0, :, :, :])
        plt.show()
    else:
        plt.imshow(observation[0, :, :, 0])
        plt.show()

for episode in range(10):
    env_info_str = env.reset(train_mode=train_mode)[striker_brain_name]
    env_info_goalie = env.reset(train_mode=train_mode)[goalie_brain_name]

    done_str = False
    done_goalie = False
    episode_rewards = 0

    num_agent_str = len(env_info_str.agents)
    print("str_",num_agent_str)
    num_agent_goalie = len(env_info_goalie.agents)
    print("goalie_",num_agent_goalie)
    while not (done_str & done_goalie):
        action_size_str = striker_brain.vector_action_space_size
        # print('action_size_str=', action_size_str)
        action_size_goalie = goalie_brain.vector_action_space_size
        # print('action_size_goalie=', action_size_goalie)
        action_str = np.column_stack([
            np.random.randint(0, action_size_str[i], size=num_agent_str)
            for i in range(len(action_size_str))
        ])
        action_goalie = np.column_stack([
            np.random.randint(0, action_size_goalie[i], size=num_agent_goalie)
            for i in range(len(action_size_goalie))
        ])

        env_info = env.step({
            striker_brain_name: action_str,
            goalie_brain_name: action_goalie
        })

        episode_rewards += env_info[striker_brain_name].rewards[0] + env_info[
            goalie_brain_name].rewards[0]
        done_str = env_info[striker_brain_name].local_done[0]
        done_goalie = env_info[goalie_brain_name].local_done[0]
    print("Total reward this episode: {}".format(episode_rewards))

env.close()


class SocTwoEnv:
    def __init__(self, env_path, worker_id, train_mode=True):
        self.env = UnityEnvironment(file_name=env_path, worker_id=worker_id)
        self.striker_brain_name, self.goalie_brain_name = self.env.brain_names
        self.striker_brain = env.brains[self.striker_brain_name]
        self.goalie_brain = env.brains[self.goalie_brain_name]
        return
    # def num_

    def reset(self):
        self.env_info_str = self.env.reset(
            train_mode=self.train_mode)[self.striker_brain_name]
        self.env_info_goalie = self.env.reset(
            train_mode=self.train_mode)[self.goalie_brain_name]
        return {'str':self.env_info_str, 'goalie':self.env_info_goalie}

    def step(self):
        self.env.step
        return
