import argparse
import torch
import torch.optim as optim

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from a2c.models import AtariCNN, A2C
from a2c.envs import make_env, RenderSubprocVecEnv
from a2c.train_multi import train

from env_exp import SocTwoEnv

parser = argparse.ArgumentParser(description='A2C (Advantage Actor-Critic)')
# parser.add_argument('env_name', type=str, help='Gym environment id')
parser.add_argument('--no-cuda',
                    action='store_true',
                    help='use to disable available CUDA')
parser.add_argument('--num-workers',
                    type=int,
                    default=16,
                    help='number of parallel workers')
parser.add_argument('--rollout-steps',
                    type=int,
                    default=20,
                    help='steps per rollout')
parser.add_argument('--total-steps',
                    type=int,
                    default=int(4e7),
                    help='total number of steps to train for')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--gamma',
                    type=float,
                    default=0.99,
                    help='gamma parameter for GAE')
parser.add_argument('--lambd',
                    type=float,
                    default=1.00,
                    help='lambda parameter for GAE')
parser.add_argument('--value_coeff',
                    type=float,
                    default=0.5,
                    help='value loss coeffecient')
parser.add_argument('--entropy_coeff',
                    type=float,
                    default=0.01,
                    help='entropy loss coeffecient')
parser.add_argument('--grad_norm_limit',
                    type=float,
                    default=40.,
                    help='gradient norm clipping threshold')
parser.add_argument('--render',
                    action='store_true',
                    help='render training environments')
parser.add_argument('--render-interval',
                    type=int,
                    default=4,
                    help='number of steps between environment renders')
parser.add_argument('--plot-reward',
                    action='store_true',
                    help='plot episode reward vs. total steps')
parser.add_argument('--plot-group-size',
                    type=int,
                    default=80,
                    help='number of episodes grouped into a single plot point')
parser.add_argument('--seed', type=int, default=0, help='random seed')

args = parser.parse_args()

env_path = './env/macos/SoccerTwosFast.app'
env = SocTwoEnv(env_path, worker_id=0, train_mode=True)

# cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device(
    "cuda:0" if torch.cuda.is_available() and (not args.no_cuda) else "cpu")

policy_striker, policy_goalie = A2C(7).to(device), A2C(5).to(device)

optim_striker = optim.Adam(policy_striker.parameters(), lr=args.lr)
optim_goalie = optim.Adam(policy_goalie.parameters(), lr=args.lr)

# if cuda:
#     policy_striker = policy_striker.cuda()
#     policy_goalie = policy_goalie.cuda()

train(args, policy_striker, policy_goalie, optim_striker, optim_goalie, env,
      device)
