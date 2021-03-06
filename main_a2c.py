import argparse
import torch
import torch.optim as optim
import os

# from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from a2c.models import AtariCNN, A2C, A2CLarge
# from a2c.envs import make_env, RenderSubprocVecEnv
from a2c.train_multi import train

from env_exp import SocTwoEnv

parser = argparse.ArgumentParser(description='A2C (Advantage Actor-Critic)')

parser.add_argument('--env_path',
                    type=str,
                    default='./env/macos/SoccerTwosFast.app',
                    help='path to the environment binary')
parser.add_argument('--render',
                    type=bool,
                    default=False,
                    help='whether to render enviroment')
parser.add_argument('--reward_shaping',
                    type=bool,
                    default=False,
                    help='whether to use reward shaping')
parser.add_argument('--num-workers',
                    type=int,
                    default=16,
                    help='number of parallel workers')
parser.add_argument('--rollout-steps',
                    type=int,
                    default=300,
                    help='steps per rollout')
parser.add_argument('--total-steps',
                    type=int,
                    default=int(4e7),
                    help='total number of steps to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
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

args = parser.parse_args()

# env_path = './env/macos/SoccerTwosFast.app'
env = SocTwoEnv(args.env_path, worker_id=0, train_mode=True, render=args.render)

os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device(
    "cuda:0" if torch.cuda.is_available()else "cpu")
print(device)

# policy_striker, policy_goalie = A2C(7).to(device), A2C(5).to(device)
policy_striker, policy_goalie = A2CLarge(7).to(device), A2CLarge(5).to(device)

optim_striker = optim.Adam(policy_striker.parameters(), lr=args.lr)
optim_goalie = optim.Adam(policy_goalie.parameters(), lr=args.lr)

train(args, policy_striker, policy_goalie, optim_striker, optim_goalie, env,
      device, args.reward_shaping)
