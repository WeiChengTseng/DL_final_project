import numpy as np
import sys
from mlagents.envs import UnityEnvironment

SIZE_OBSERVATION = 112


class SocTwoEnv():
    """
    parameters:
        env_path: the binary file which is built by Unity.
        train_mode: It's True when u want to train the brain.
        n_striker: number of agent of striker in the scene.
        n_goalie: number of agent of goalie in the scene.
    
    **********************************************************
    Store "action" of each agent in act_striker_hist and act_goalie_hist
    respectively.

    Store "Observation" or "State" of each agent in observation_striker_hist
    and observation_goalie_hist respectively.
    """

    def __init__(self,
                 env_path,
                 worker_id,
                 train_mode=True,
                 n_striker=16,
                 n_goalie=16,
                 render=True):
        self._striker_map = {
            'field': [8, 0, 4, 2, 14, 10, 12, 6, 9, 1, 5, 3, 15, 11, 13, 7],
<<<<<<< HEAD
            'team': [12, 8, 10, 9, 15, 13, 14, 11, 4, 0, 2, 1, 7, 5, 6, 3],
            'test': [0, 7, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
        }
        self._goalie_map = {
            'field': [8, 0, 4, 2, 14, 10, 12, 6, 13, 7, 11, 3, 15, 9, 5, 1],
            'team': [12, 8, 10, 9, 15, 13, 14, 11, 6, 3, 5, 1, 7, 4, 2, 0],
            'test': [0, 13, 1, 15, 2, 14, 3, 11, 4, 12, 5, 10, 6, 8, 7, 9]
        }



        self._striker_inv_map = {
            'field': np.argsort(self._striker_map['field']),
            'team': np.argsort(self._striker_map['team']),
            'test': np.argsort(self._striker_map['test']),
        }
        self._goalie_inv_map = {
            'field': np.argsort(self._goalie_map['field']),
            'team': np.argsort(self._goalie_map['team']),
            'test': np.argsort(self._goalie_map['test'])

        self.env = UnityEnvironment(file_name=env_path,
                                    worker_id=0,
                                    no_graphics=not render)
        self.striker_brain_name, self.goalie_brain_name = self.env.brain_names
        self.striker_brain = self.env.brains[self.striker_brain_name]
        self.goalie_brain = self.env.brains[self.goalie_brain_name]
        self.done_striker = [False] * 16
        self.done_goalie = [False] * 16
        self.train_mode = train_mode
        self.done_hist_striker = [False] * 16
        self.done_hist_goalie = [False] * 16
        self.episode_striker_rewards = 0
        self.episode_goalie_rewards = 0
        self.n_striker = n_striker
        self.n_goalie = n_goalie

        self.observation_striker = None
        self.observation_goalie = None
        return

    def reset(self, order=None):
        """
        Reset the all environments and agents.
        """
        self.env_info_striker = self.env.reset(
            train_mode=self.train_mode)[self.striker_brain_name]
        self.env_info_goalie = self.env.reset(
            train_mode=self.train_mode)[self.goalie_brain_name]
        self.episode_rewards = 0
        self.done_striker = [False] * 16
        self.done_goalie = [False] * 16

        empty_action = np.zeros(16)
        return self.step(empty_action, empty_action, order)[0]

    def step(self, action_striker, action_goalie, order=None):
        """
        In each timestep, give each striker and goalie a instruction
        to do action. And then, get the current observation stored
        at observation_striker and observation_goalie.
        Input:
        - action_striker: a vector with shape [num_striker]
        - action_goalie: a vector with shape [num_goalie]
        """
        if order:
            action_striker = action_striker[self._striker_map[order]]
            action_goalie = action_goalie[self._goalie_map[order]]

        self.env_info = self.env.step({
            self.striker_brain_name: action_striker,
            self.goalie_brain_name: action_goalie
        })
        self.observation_striker = np.array(
            self.env_info[self.striker_brain_name].vector_observations)
        self.observation_goalie = np.array(
            self.env_info[self.goalie_brain_name].vector_observations)

        rewards_striker, rewards_goalie = self.reward()
        dones_striker, dones_goalie = self.done()
<<<<<<< HEAD
        if True in dones_goalie:
            print("before", dones_goalie)
        rewards_striker = rewards_striker[self._striker_inv_map['field']]
        rewards_goalie = rewards_goalie[self._goalie_inv_map['field']]
        dones_striker = dones_striker[self._striker_inv_map['field']]
        dones_goalie = dones_goalie[self._goalie_inv_map['field']]
        if True in dones_goalie:
            print("after", dones_goalie)

        return [[self.observation_striker, self.observation_goalie],
                [rewards_striker, rewards_goalie],
                [dones_striker, dones_goalie], None]

    def reward(self):
        """
        return the rewards of striker and goalie respectively.
        """
        self.episode_striker_rewards = np.array(
            self.env_info[self.striker_brain_name].rewards)
        self.episode_goalie_rewards = np.array(
            self.env_info[self.goalie_brain_name].rewards)
        return self.episode_striker_rewards, self.episode_goalie_rewards

    def close(self):
        """
        Close the simulation Unity environment.
        """
        self.env.close()
        return

    def done(self):
        """
        Check whether each agent has finished at their own episode,
        true means it has finished.
        """
        self.done_striker = np.array(
            self.env_info[self.striker_brain_name].local_done)
        self.done_goalie = np.array(
            self.env_info[self.goalie_brain_name].local_done)
        return self.done_striker, self.done_goalie

    def train(self):
        self.train_mode = True
        self.reset()
        return

    def eval(self):
        self.train_mode = False
        self.reset()
        return


if __name__ == "__main__":

    env_path = './env/macos/SoccerTwosLearnerBirdView.app'
    soc_env = SocTwoEnv(env_path, worker_id=0, train_mode=False)
    order = 'team'
    soc_env.reset(order)  # Don't touch me!
    episode = 0
    for i in range(16):
        for _ in range(40):
            action_size_str = soc_env.striker_brain.vector_action_space_size
            action_size_goalie = soc_env.goalie_brain.vector_action_space_size

            # randomly generate some actions for each agent.

            action_striker = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            action_goalie = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # action_striker = np.random.randint(7, size=16, dtype=int)
            # action_goalie = np.random.randint(5, size=16, dtype=int)

            action_striker[i] = np.random.randint(7)
            action_goalie[i] = np.random.randint(5)

            action_striker = np.array(action_striker)
            action_goalie = np.array(action_goalie)
            soc_env.step(action_striker, action_goalie, order)

    soc_env.close()

