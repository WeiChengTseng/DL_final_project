import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from a2c.utils import mean_std_groups


def train(args, net_striker, net_goalie, optim_striker, optim_goalie, env,
          device):
    obs_striker, obs_goalie = env.reset()

    steps_striker, steps_goalie = [], []
    total_steps = 0
    ep_rewards_striker = [0.] * args.num_workers
    ep_rewards_goalie = [0.] * args.num_workers

    # writer = SummaryWriter('./a2c/logs/')

    while total_steps < args.total_steps:
        for _ in range(args.rollout_steps):
            obs_striker = Variable(
                torch.from_numpy(obs_striker).float()).to(device)
            obs_goalie = Variable(
                torch.from_numpy(obs_goalie).float()).to(device)

            # network forward pass
            policies_striker, values_striker = net_striker(obs_striker)
            policies_goalie, values_goalie = net_goalie(obs_goalie)

            probs_striker = F.softmax(policies_striker, dim=-1)
            probs_goalie = F.softmax(policies_goalie, dim=-1)

            actions_striker = probs_striker.multinomial(1).data
            actions_goalie = probs_goalie.multinomial(1).data

            # gather env data, reset done envs and update their obs
            obs, rewards, dones, _ = env.step(actions_striker, actions_goalie)
            obs_striker, obs_goalie = obs

            # reset the LSTM state for done envs
            masks_striker = (1. - torch.from_numpy(
                np.array(dones[0], dtype=np.float32))).unsqueeze(1).to(device)
            masks_goalie = (1. - torch.from_numpy(
                np.array(dones[1], dtype=np.float32))).unsqueeze(1).to(device)

            total_steps += args.num_workers
            for i, done in enumerate(dones[0]):
                ep_rewards_striker[i] += rewards[0][i]
                if done:
                    ep_rewards_striker[i] = 0

            for i, done in enumerate(dones[1]):
                ep_rewards_goalie[i] += rewards[1][i]
                if done:
                    ep_rewards_goalie[i] = 0

            rewards_striker = torch.from_numpy(
                rewards[0]).float().unsqueeze(1).to(device)
            rewards_goalie = torch.from_numpy(
                rewards[1]).float().unsqueeze(1).to(device)

            steps_striker.append(
                (rewards_striker, masks_striker, actions_striker,
                 policies_striker, values_striker))
            steps_goalie.append((rewards_goalie, masks_goalie, actions_goalie,
                                 policies_goalie, values_goalie))

        final_obs_striker = Variable(
            torch.from_numpy(obs_striker).float()).to(device)
        final_obs_goalie = Variable(
            torch.from_numpy(obs_goalie).float()).to(device)

        _, final_values_striker = net_striker(final_obs_striker)
        _, final_values_goalie = net_goalie(final_obs_goalie)

        steps_striker.append((None, None, None, None, final_values_striker))
        steps_goalie.append((None, None, None, None, final_values_goalie))

        actions_striker, policies_striker, values_striker, returns_striker, advantages_striker = process_rollout(
            args, steps_striker, device)

        actions_goalie, policies_goalie, values_goalie, returns_goalie, advantages_goalie = process_rollout(
            args, steps_goalie, device)

        # calculate action probabilities
        probs_striker = F.softmax(policies_striker, dim=-1)
        probs_goalie = F.softmax(policies_goalie, dim=-1)

        log_probs_striker = F.log_softmax(policies_striker, dim=-1)
        log_probs_goalie = F.log_softmax(policies_goalie, dim=-1)

        log_action_probs_striker = log_probs_striker.gather(
            1, Variable(actions_striker))
        log_action_probs_goalie = log_probs_goalie.gather(
            1, Variable(actions_goalie))

        policy_loss_striker = (-log_action_probs_striker *
                               Variable(advantages_striker)).sum()
        policy_loss_goalie = (-log_action_probs_goalie *
                              Variable(advantages_goalie)).sum()

        value_loss_striker = (
            .5 * (values_striker - Variable(returns_striker))**2.).sum()
        value_loss_goalie = (
            .5 * (values_goalie - Variable(returns_goalie))**2.).sum()

        entropy_loss_striker = (log_probs_striker * probs_striker).sum()
        entropy_loss_goalie = (log_probs_goalie * probs_goalie).sum()

        loss_striker = policy_loss_striker + value_loss_striker * args.value_coeff + entropy_loss_striker * args.entropy_coeff
        loss_goalie = policy_loss_goalie + value_loss_goalie * args.value_coeff + entropy_loss_goalie * args.entropy_coeff

        loss_striker.backward()
        loss_goalie.backward()

        nn.utils.clip_grad_norm_(net_striker.parameters(),
                                 args.grad_norm_limit)
        optim_striker.step()
        optim_striker.zero_grad()

        nn.utils.clip_grad_norm_(net_goalie.parameters(), args.grad_norm_limit)
        optim_goalie.step()
        optim_goalie.zero_grad()

        # cut LSTM state autograd connection to previous rollout
        steps_striker, steps_goalie = [], []

    env.close()
    return


def process_rollout(args, steps, device):
    # bootstrap discounted returns with final value estimates
    _, _, _, _, last_values = steps[-1]
    returns = last_values.data

    advantages = torch.zeros(args.num_workers, 1).to(device)
    # if cuda: advantages = advantages.cuda()

    out = [None] * (len(steps) - 1)

    # run Generalized Advantage Estimation, calculate returns, advantages
    for t in reversed(range(len(steps) - 1)):
        rewards, masks, actions, policies, values = steps[t]
        _, _, _, _, next_values = steps[t + 1]

        returns = rewards + returns * args.gamma * masks

        deltas = rewards + next_values.data * args.gamma * masks - values.data
        advantages = advantages * args.gamma * args.lambd * masks + deltas

        out[t] = actions, policies, values, returns, advantages

    # return data as batched Tensors, Variables
    return map(lambda x: torch.cat(x, 0), zip(*out))


def eval_with_random_agent(net_striker,
                           net_goalie,
                           env,
                           device,
                           eval_epsoid=10):
    obs_striker, obs_goalie = env.reset('team')
    epsoid = 0
    while epsoid < eval_epsoid:
        obs_striker = Variable(
            torch.from_numpy(obs_striker).float()).to(device)
        obs_goalie = Variable(torch.from_numpy(obs_goalie).float()).to(device)

        policies_striker, values_striker = net_striker(obs_striker)
        policies_goalie, values_goalie = net_goalie(obs_goalie)

        probs_striker = F.softmax(policies_striker, dim=-1)
        probs_goalie = F.softmax(policies_goalie, dim=-1)

        actions_striker = probs_striker.multinomial(1).data
        actions_goalie = probs_goalie.multinomial(1).data

        # gather env data, reset done envs and update their obs
        actions_striker = torch.cat(
            [actions_striker[:8],
             torch.zeros(8, dtype=torch.int)])
        actions_goalie = torch.cat(
            [actions_goalie[:8],
             torch.zeros(8, dtype=torch.int)])
        obs, rewards, dones, _ = env.step(actions_striker, actions_goalie,
                                          'team')
        obs_striker, obs_goalie = obs

        rewards_striker = torch.from_numpy(
            rewards[0]).float().unsqueeze(1).to(device)
        rewards_goalie = torch.from_numpy(
            rewards[1]).float().unsqueeze(1).to(device)

        for i in np.argwhere(dones[0]).flatten():
            epsoid += 1
            pass
    return
