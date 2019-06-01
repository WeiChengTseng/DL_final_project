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
    if args.plot_reward:
        total_steps_plt, ep_reward_plt = [], []

    steps_striker, steps_goalie = [], []
    total_steps = 0
    ep_rewards_striker = [0.] * args.num_workers
    ep_rewards_goalie = [0.] * args.num_workers
    # render_timer = 0
    # plot_timer = 0
    writer = SummaryWriter('./a2c/logs/')

    while total_steps < args.total_steps:
        for _ in range(args.rollout_steps):
            obs_striker = Variable(
                torch.from_numpy(obs_striker).float()).to(device)
            obs_goalie = Variable(torch.from_numpy(obs_goalie).float() /
                                  255.).to(device)
            # if cuda:
            #     obs = obs.cuda()

            # network forward pass
            policies_striker, values_striker = net_striker(obs_striker)
            policies_goalie, values_goalie = net_goalie(obs_goalie)

            probs_striker = F.softmax(policies_striker, dim=-1)
            probs_goalie = F.softmax(policies_goalie, dim=-1)

            actions_striker = probs_striker.multinomial(1).data
            actions_goalie = probs_goalie.multinomial(1).data

            # gather env data, reset done envs and update their obs
            obs, rewards, dones, _ = env.step(actions_striker, actions_goalie)

            # reset the LSTM state for done envs
            masks_striker = (1. - torch.from_numpy(
                np.array(dones[0], dtype=np.float32))).unsqueeze(1).to(device)
            masks_goalie = (1. - torch.from_numpy(
                np.array(dones[1], dtype=np.float32))).unsqueeze(1).to(device)
            # if cuda: masks = masks.cuda()

            total_steps += args.num_workers
            for i, done in enumerate(dones[0]):
                ep_rewards_striker[i] += rewards[0][i]
                if done:
                    # if args.plot_reward:
                    #     total_steps_plt.append(total_steps)
                    #     ep_reward_plt.append(ep_rewards[i])
                    ep_rewards_striker[i] = 0
            for i, done in enumerate(dones[1]):
                ep_rewards_goalie[i] += rewards[1][i]
                if done:
                    # if args.plot_reward:
                    #     total_steps_plt.append(total_steps)
                    #     ep_reward_plt.append(ep_rewards[i])
                    ep_rewards_goalie[i] = 0

            # if args.plot_reward:
            #     plot_timer += args.num_workers  # time on total steps
            #     if plot_timer == 100000:
            #         x_means, _, y_means, y_stds = mean_std_groups(
            #             np.array(total_steps_plt), np.array(ep_reward_plt),
            #             args.plot_group_size)
            #         fig = plt.figure()
            #         fig.set_size_inches(8, 6)
            #         plt.ticklabel_format(axis='x',
            #                              style='sci',
            #                              scilimits=(-2, 6))
            #         plt.errorbar(x_means,
            #                      y_means,
            #                      yerr=y_stds,
            #                      ecolor='xkcd:blue',
            #                      fmt='xkcd:black',
            #                      capsize=5,
            #                      elinewidth=1.5,
            #                      mew=1.5,
            #                      linewidth=1.5)
            #         plt.title('Training progress (%s)' % args.env_name)
            #         plt.xlabel('Total steps')
            #         plt.ylabel('Episode reward')
            #         plt.savefig('ep_reward.png', dpi=200)
            #         plt.clf()
            #         plt.close()
            #         plot_timer = 0

            rewards_striker = torch.from_numpy(
                rewards[0]).float().unsqueeze(1).to(device)
            # if cuda: rewards = rewards.cuda()
            steps_striker.append(
                (rewards_striker, masks_striker, actions_striker,
                 policies_striker, values_striker))

            rewards_goalie = torch.from_numpy(
                rewards[1]).float().unsqueeze(1).to(device)
            # if cuda: rewards = rewards.cuda()
            steps_goalie.append((rewards_goalie, masks_goalie, actions_goalie,
                                 policies_goalie, values_goalie))

        # print(np.array([i[0].flatten().sum() for i in steps]).mean())
        # writer.add_scalar(
        #     'average_reward',
        #     np.array([i[0].flatten().sum().item() for i in steps]).mean(),
        #     total_steps)
        final_obs_striker = Variable(
            torch.from_numpy(obs_striker).float()).to(device)
        final_obs_goalie = Variable(
            torch.from_numpy(obs_goalie).float()).to(device)
        # if cuda:
        #     final_obs = final_obs.cuda()
        _, final_values_striker = net_striker(final_obs_striker)
        _, final_values_goalie = net_goalie(final_obs_goalie)

        steps_striker.append((None, None, None, None, final_values_striker))
        steps_goalie.append((None, None, None, None, final_values_goalie))

        actions, policies, values, returns, advantages = process_rollout(
            args, steps, device)
        actions, policies, values, returns, advantages = process_rollout(
            args, steps, device)

        # calculate action probabilities
        probs = F.softmax(policies, dim=-1)
        log_probs = F.log_softmax(policies, dim=-1)
        log_action_probs = log_probs.gather(1, Variable(actions))

        policy_loss = (-log_action_probs * Variable(advantages)).sum()
        value_loss = (.5 * (values - Variable(returns))**2.).sum()
        entropy_loss = (log_probs * probs).sum()

        loss = policy_loss + value_loss * args.value_coeff + entropy_loss * args.entropy_coeff
        loss.backward()

        nn.utils.clip_grad_norm(net.parameters(), args.grad_norm_limit)
        optimizer.step()
        optimizer.zero_grad()

        # cut LSTM state autograd connection to previous rollout
        steps = []

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
