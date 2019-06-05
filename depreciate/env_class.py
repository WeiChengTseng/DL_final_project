import numpy as np
import sys
from mlagents.envs import UnityEnvironment

SIZE_OBSERVATION = 112

class SocTwoEnv():
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
        self.done_str = np.array(
            self.env_info[self.striker_brain_name].local_done)
        self.done_goalie = np.array(
            self.env_info[self.goalie_brain_name].local_done)

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

    def print_r(self, episode):
        print("Total reward this episode_{}: {}".format(
            episode, self.episode_rewards))
        return


if __name__ == "__main__":

    env_Path = r'.\env\windows\SoccerTwosBirdView\Unity Environment.exe'
    env_Path = './env/macos/SoccerTwosLearnerBirdView.app.app'
    soc_env = SocTwoEnv(env_Path, worker_id=1, train_mode=True)
    soc_env.reset()

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
            (soc_env.act_str_hist[i]).append(action_str[i][0])
        soc_env.step(action_str, action_goalie)
        soc_env.done()
        if True in soc_env.done_goalie:
            soc_env.reward()
            print("episode", episode, "*" * 10)
            arg_done_goalie = np.argwhere(soc_env.done_goalie == True)
            for i in arg_done_goalie:
                print("which goalie %d act" % (i[0]),
                      soc_env.act_goalie_hist[i[0]], "len",
                      len(soc_env.act_goalie_hist[i[0]]))
                print("reword", soc_env.episode_goalie_rewards[i][0])
            arg_done_str = np.argwhere(soc_env.done_str == True)
            for i in arg_done_str:
                print("which str %d act" % (i[0]), soc_env.act_str_hist[i[0]],
                      "len", len(soc_env.act_str_hist[i[0]]))
                print("reword", soc_env.episode_str_rewards[i][0])
            soc_env.reset_some_agents(arg_done_str, arg_done_goalie)
            print("*" * 10)
            episode += 1
            c = 0
    soc_env.close()
