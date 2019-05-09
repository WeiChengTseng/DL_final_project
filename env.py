import matplotlib.pyplot as plt
import numpy as np
import sys
from gym_unity.envs import UnityEnv

env_name = "./env/SoccerTwos"
multi_env = UnityEnv(env_name, worker_id=1, 
                     use_visual=False, multiagent=True)

print(str(multi_env))


# Reset the environment
initial_observations = multi_env.reset()

if len(multi_env.observation_space.shape) == 1:
    # Examine the initial vector observation
    print("Agent observations look like: \n{}".format(initial_observations[0]))
else:
    # Examine the initial visual observation
    print("Agent observations look like:")
    if multi_env.observation_space.shape[2] == 3:
        plt.imshow(initial_observations[0][:,:,:])
    else:
        plt.imshow(initial_observations[0][:,:,0])

for episode in range(10):
    initial_observation = multi_env.reset()
    done = False
    episode_rewards = 0
    while not done:
        actions = [multi_env.action_space.sample() for agent in range(multi_env.number_agents)]
        observations, rewards, dones, info = multi_env.step(actions)
        episode_rewards += np.mean(rewards)
        done = dones[0]
    print("Total reward this episode: {}".format(episode_rewards))

multi_env.close()