import argparse
import torch
import os
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from maac_ac.utils.buffer import ReplayBuffer
from maac_ac.attention_sac import AttentionSACAC

from a2c.agent_wraper import A2CWraper

from env_exp import SocTwoEnv

OBS_DIM = 112
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse_double(obs):
    parsed_obs = [None] * 4
    parsed_obs[0] = obs[0][:8]
    parsed_obs[2] = obs[0][8:]
    parsed_obs[1] = obs[1][:8]
    parsed_obs[3] = obs[1][8:]
    return np.array(parsed_obs)


def run(config):
    model_dir = Path('./maac_ac/models') / config.model_name
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

    env = SocTwoEnv(config.env_path,
                    worker_id=0,
                    train_mode=True,
                    render=config.render)
    obs = parse_double(env.reset('team'))

    print('The training process use', device)

    # create the model
    model = AttentionSACAC.init_from_env(
        env,
        tau=config.tau,
        pi_lr=config.pi_lr,
        q_lr=config.q_lr,
        gamma=config.gamma,
        pol_hidden_dim=config.pol_hidden_dim,
        critic_hidden_dim=config.critic_hidden_dim,
        attend_heads=config.attend_heads,
        reward_scale=config.reward_scale,
        device=device)

    ac_ckpt = torch.load(config.ac_ckpt, map_location=device)
    ac_striker = A2CWraper(7).to(device)
    ac_goalie = A2CWraper(5).to(device)
    ac_striker.policy.load_state_dict(ac_ckpt['striker_a2c'])
    ac_goalie.policy.load_state_dict(ac_ckpt['goalie_a2c'])

    replay_buffer = ReplayBuffer(config.buffer_length, model.nagents,
                                 [OBS_DIM] * 4, [7, 5] * 2)

    t, ep_i = 0, 0
    model.prep_rollouts(device=device)
    # for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
    while ep_i < config.n_episodes:

        if ep_i % 100 == 0:
            print('Episode:', ep_i)

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable

            obs = np.array(obs)
            torch_obs = [
                torch.FloatTensor(obs[i]).to(device)
                for i in range(model.nagents)
            ]

            ac_striker_action = ac_striker(torch_obs[2])
            ac_goalie_action = ac_goalie(torch_obs[3])

            ac_striker_action_oh = np.zeros((8, 7), dtype=np.float32)
            ac_goalie_action_oh = np.zeros((8, 5), dtype=np.float32)
            ac_striker_action_oh[np.arange(8), ac_striker_action] = 1
            ac_goalie_action_oh[np.arange(8), ac_goalie_action] = 1

            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [
                ac.data.cpu().numpy() for ac in torch_agent_actions
            ]
            agent_actions[2] = ac_striker_action_oh
            agent_actions[3] = ac_goalie_action_oh

            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions]
                       for i in range(config.n_rollout_threads)]

            actions = [
                list(np.argmax(action, axis=-1)) for action in agent_actions
            ]

            next_obs, rewards, dones, infos = env.step(
                np.array(actions[0] + list(ac_striker_action)),
                np.array(actions[1] + list(ac_goalie_action)),
                order='team')
            next_obs = parse_double(next_obs)
            replay_buffer.push(obs, agent_actions, parse_double(rewards),
                               next_obs, parse_double(dones))
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                model.prep_training(device=device)
                # print('timestep:', t)
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  norm_rews=False,
                                                  device=device)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device=device)

            done_env = np.argwhere(dones[0])
            if len(done_env) != 0:
                ep_i += len(done_env)
                break

        if ep_i % config.save_interval == 0:
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
                        default=8,
                        type=int,
                        help='the number of environment in training process')
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=int(4e7), type=int)
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
    parser.add_argument("--save_interval", default=2000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--ac_ckpt",
                        default='./a2c/ckpt_wors_2e/a2cLarge_step39960000.pth',
                        type=str)

    config = parser.parse_args()

    run(config)