import numpy as np
import sys
from mlagents.envs import UnityEnvironment

SIZE_OBSERVATION = 112


class SocTwoEnv():
    """
        parameters:
            env_path: the binary file which is built by Unity.
            train_mode: It's True when u want to train the brain.
            n_str: number of agent of striker in the scene.
            n_goalie: number of agent of goalie in the scene.
        
        **********************************************************
        Store "action" of each agent in act_str_hist and act_goalie_hist
        respectively.

        Store "Observation" or "State" of each agent in observation_str_hist
        and observation_goalie_hist respectively.
    """

    def __init__(self,
                 env_path,
                 worker_id,
                 train_mode=True,
                 n_str=16,
                 n_goalie=16):
        self.env = UnityEnvironment(file_name=env_path, worker_id=0)
        self.striker_brain_name, self.goalie_brain_name = self.env.brain_names
        self.striker_brain = self.env.brains[self.striker_brain_name]
        self.goalie_brain = self.env.brains[self.goalie_brain_name]
        self.done_str = [False] * 16
        self.done_goalie = [False] * 16
        self.train_mode = train_mode
        self.done_hist_str = [False] * 16
        self.done_hist_goalie = [False] * 16
        self.episode_str_rewards = 0
        self.episode_goalie_rewards = 0
        self.n_str = n_str
        self.n_goalie = n_goalie
        self.act_str_hist = [[] for x in range(n_str)]
        self.act_goalie_hist = [[] for x in range(n_goalie)]
        self.observation_str_hist = [[] for x in range(SIZE_OBSERVATION)]
        self.observation_goalie_hist = [[] for x in range(SIZE_OBSERVATION)]
        self.observation_str = None
        self.observation_goalie = None
        return

    def reset(self):
        """
            Reset the all environments and agents.
        """
        self.env_info_str = self.env.reset(
            train_mode=self.train_mode)[self.striker_brain_name]
        self.env_info_goalie = self.env.reset(
            train_mode=self.train_mode)[self.goalie_brain_name]
        

        self.episode_rewards = 0
        self.done_str = [False] * 16
        self.done_goalie = [False] * 16
        self.done_hist_str = np.array([False] * 16)
        self.done_hist_goalie = np.array([False] * 16)
        return {'str': self.env_info_str, 'goalie': self.env_info_goalie}

    def step(self, action_str, action_goalie):
        """
            In each timestep, give each striker and goalie a instruction
            to do action. And then, get the current observation stored
            at observation_str and observation_goalie.
        """
        self.env_info = self.env.step({
            self.striker_brain_name: action_str,
            self.goalie_brain_name: action_goalie
        })
        self.observation_str = np.array(
            self.env_info[self.striker_brain_name].vector_observations)
        self.observation_goalie = np.array(
            self.env_info[self.goalie_brain_name].vector_observations)
        return self.env_info

    def reward(self):
        """
            return the rewards of striker and goalie respectively.
        """
        self.episode_str_rewards = np.array(
            self.env_info[self.striker_brain_name].rewards)
            
        self.episode_goalie_rewards = np.array(
            self.env_info[self.goalie_brain_name].rewards)
        
        
        return self.episode_str_rewards, self.episode_goalie_rewards

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
        self.done_str = np.array(
            self.env_info[self.striker_brain_name].local_done)
        self.done_goalie = np.array(
            self.env_info[self.goalie_brain_name].local_done)
        return

    def reset_some_agents(self, str_arg, goalie_arg):
        """
            params:
                str_arg, mark which striker's history that wants to be cleared.
                goalie_arg, mark which goalie's history that wants to be cleared.
            Clear the history of specific agents.

        """
        for i in str_arg:
            self.act_str_hist[i[0]] = []
            self.observation_str_hist[i[0]] = []
        for i in goalie_arg:
            self.act_goalie_hist[i[0]] = []
            self.observation_goalie_hist[i[0]] = []


if __name__ == "__main__":

    env_Path = r'./env/linux/soccer_test.x86_64'
    env_Path = r'.\env\windows\soccer_easy\Unity Environment.exe'

    env_Path = './env/macos/SoccerTwosLearnerBirdView.app'
    
    soc_env = SocTwoEnv(env_Path, worker_id=0, train_mode=True)
    soc_env.reset()  # Don't touch me!
    episode = 0
    while episode < 10:
        action_size_str = soc_env.striker_brain.vector_action_space_size
        action_size_goalie = soc_env.goalie_brain.vector_action_space_size

        # randomly generate some actions for each agent.

        action_str = np.array([3]*7+[0]*1+[3]*8)
        action_goalie = np.array([4]*8+[4]*4+[0]*1+[4]*3)


        # store the action of agent in list
        for i in range(soc_env.n_goalie):
            (soc_env.act_goalie_hist[i]).append(action_goalie[i])
        for i in range(soc_env.n_str):
            (soc_env.act_str_hist[i]).append(action_str[i])
        soc_env.step(action_str, action_goalie)
        # store the observation(state) of agent in list
        for i in range(soc_env.n_goalie):
            (soc_env.observation_goalie_hist[i]).append(
                soc_env.observation_goalie[i])
        for i in range(soc_env.n_str):
            (soc_env.observation_str_hist[i]).append(
                soc_env.observation_str[i])

        soc_env.done()
        if True in soc_env.done_goalie:
            soc_env.reward()
            print("episode: ", episode, "*" * 10)
            arg_done_goalie = np.argwhere(soc_env.done_goalie == True)
            for i in arg_done_goalie:
                # print("goalie %d"%(i[0]))
                # print("action", soc_env.act_goalie_hist[i[0]])
                # print("Observation", soc_env.observation_goalie_hist[i[0]])
                # print("reword", soc_env.episode_goalie_rewards[i][0])
                pass

            arg_done_str = np.argwhere(soc_env.done_str == True)
            for i in arg_done_str:
                # print("str %d"%(i[0]))
                # print("action", soc_env.act_str_hist[i[0]])
                # print("Observation", soc_env.observation_str_hist[i[0]])
                # print("reword", soc_env.episode_str_rewards[i][0])
                pass
            soc_env.reset_some_agents(arg_done_str, arg_done_goalie)
            print("*" * 25)
            episode += 1
            c = 0
    soc_env.close()
