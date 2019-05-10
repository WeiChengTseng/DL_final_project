import matplotlib.pyplot as plt
import numpy as np
import sys
from gym_unity.envs import UnityEnv
from mlagents.envs import UnityEnvironment

train_mode = True
env_name = './env/macos/SoccerTwosLearner.app'

env = UnityEnvironment(file_name=env_name)

striker_brain_name, goalie_brain_name = env.brain_names
striker_brain = env.brains[striker_brain_name]
goalie_brain = env.brains[goalie_brain_name]

env_info = env.reset(train_mode=train_mode)[striker_brain_name]

print("Agent state looks like: \n{}".format(env_info.vector_observations[0]))

for observation in env_info.visual_observations:
    print("Agent observations look like:")
    if observation.shape[3] == 3:
        plt.imshow(observation[0, :, :, :])
        plt.show()
    else:
        plt.imshow(observation[0, :, :, 0])
        plt.show()

for episode in range(10):
    env_info = env.reset(train_mode=train_mode)[striker_brain_name]
    done = False
    episode_rewards = 0
    num_agent = len(env_info.agents)

    while not done:
        action_size = striker_brain.vector_action_space_size
        action = np.column_stack([
            np.random.randint(0, action_size[i], size=num_agent)
            for i in range(len(action_size))
        ])
        # env_info = env.step(action)[striker_brain_name]
        env_info = env.step({
            striker_brain_name: action,
            goalie_brain_name: action
        })

        # episode_rewards += env_info.rewards[0]
        episode_rewards += env_info[striker_brain_name].rewards[0] + env_info[
            goalie_brain_name].rewards[0]
        done = env_info[striker_brain_name].local_done[0]
        done = env_info[striker_brain_name].local_done[0]
    print("Total reward this episode: {}".format(episode_rewards))

env.close()