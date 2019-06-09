import argparse
import torch
import os
import numpy as np
# from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
# from maac.utils.make_env import make_env
from maac.utils.buffer import ReplayBuffer
# from maac.utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from maac.attention_sac import AttentionSAC

from env_exp import SocTwoEnv

OBS_DIM = 112


def run(config):
    model_dir = Path('./maac/models') / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [
            int(str(folder.name).split('run')[1])
            for folder in model_dir.iterdir()
            if str(folder.name).startswith('run')
        ]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))
    device = 'cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu'

    torch.manual_seed(run_num)
    np.random.seed(run_num)

    # define the environment we need to train
    # env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
    env = SocTwoEnv(config.env_path,
                    worker_id=0,
                    train_mode=True,
                    render=config.render)
    obs = env.reset('team')

    # create the model
    model = AttentionSAC.init_from_env(
        env,
        tau=config.tau,
        pi_lr=config.pi_lr,
        q_lr=config.q_lr,
        gamma=config.gamma,
        pol_hidden_dim=config.pol_hidden_dim,
        critic_hidden_dim=config.critic_hidden_dim,
        attend_heads=config.attend_heads,
        reward_scale=config.reward_scale)
    # replay_buffer = ReplayBuffer(
    #     config.buffer_length, model.nagents,
    #     [obsp.shape[0] for obsp in env.observation_space], [
    #         acsp.shape[0] if isinstance(acsp, Box) else acsp.n
    #         for acsp in env.action_space
    #     ])
    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [OBS_DIM, OBS_DIM], [7, 5])

    t, ep_i = 0, 0
    # for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
    while ep_i < config.n_episodes:
        # print(
        #     "Episodes %i-%i of %i" %
        #     (ep_i + 1, ep_i + 1 + config.n_rollout_threads, config.n_episodes))
        if ep_i % 1000 == 0:
            print('Episode:', ep_i)
        # obs = env.reset('team')
        model.prep_rollouts(device=device)

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            # torch_obs = [
            #     Variable(torch.Tensor(np.vstack(obs[:, i])),
            #              requires_grad=False) for i in range(model.nagents)
            # ]

            obs = np.array(obs)
            torch_obs = [
                torch.FloatTensor(obs[i]).to(device)
                for i in range(model.nagents)
            ]

            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            # shape [(16, 7), (16, 5)]

            # convert actions to numpy arrays
            agent_actions = [
                ac.data.cpu().numpy() for ac in torch_agent_actions
            ]

            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions]
                       for i in range(config.n_rollout_threads)]

            actions = [np.argmax(action, axis=-1) for action in agent_actions]

            next_obs, rewards, dones, infos = env.step(actions[0], actions[1])
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                model.prep_training(device=device)
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  norm_rews=False,
                                                  to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device=device)

            for i in np.argwhere(dones):
                ep_i += 1
                pass

        # ep_rews = replay_buffer.get_average_rewards(config.episode_length *
        #                                             config.n_rollout_threads)

        # for a_i, a_ep_rew in enumerate(ep_rews):
        #     logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew,
        #                       ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device=device)
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' %
                                                  (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name",
                        help="Name of directory to store " +
                        "model/training contents",
                        default='maac')
    parser.add_argument('--env_path',
                        type=str,
                        default='./env/macos/SoccerTwosBeta.app',
                        help='path to the environment binary')
    parser.add_argument('--render',
                        type=bool,
                        default=False,
                        help='whether to render enviroment')
    parser.add_argument("--n_rollout_threads",
                        default=16,
                        type=int,
                        help='the number of environment in training process')
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=int(5e6), type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates",
                        default=4,
                        type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024,
                        type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    config = parser.parse_args()

    run(config)