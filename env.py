import matplotlib.pyplot as plt
import numpy as np
import sys
from gym_unity.envs import UnityEnv
from mlagents.envs import UnityEnvironment

train_mode = True
env_name = "./env/SoccerTwosLearner"

env = UnityEnvironment(file_name=env_name, )

# Set the default brain to work with
default_brain = env.brain_names[0]
print(env.brain_names)
brain = env.brains[default_brain]



# Reset the environment
env_info = env.reset(train_mode=train_mode)[default_brain]

# Examine the state space for the default brain
print("Agent state looks like: \n{}".format(env_info.vector_observations[0]))

# Examine the observation space for the default brain
for observation in env_info.visual_observations:
    print("Agent observations look like:")
    if observation.shape[3] == 3:
        plt.imshow(observation[0,:,:,:])
        plt.show()
    else:
        plt.imshow(observation[0,:,:,0])
        plt.show()



for episode in range(10):
    env_info = env.reset(train_mode=train_mode)[default_brain]
    done = False
    episode_rewards = 0
    while not done:
        action_size = brain.vector_action_space_size
        if brain.vector_action_space_type == 'continuous':
            env_info = env.step(np.random.randn(len(env_info.agents), 
                                                action_size[0]))[default_brain]
        else:
            action = np.column_stack([np.random.randint(0, action_size[i], size=(len(env_info.agents))) for i in range(len(action_size))])
            env_info = env.step(action)[default_brain]
        episode_rewards += env_info.rewards[0]
        done = env_info.local_done[0]
    print("Total reward this episode: {}".format(episode_rewards))


env.close()