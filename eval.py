import time
import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from a2c.models import AtariCNN, A2C, A2CLarge
from a2c.envs import make_env, RenderSubprocVecEnv
from a2c.train_multi import train

from ppo.PPO import PPO

from maac.attention_sac import AttentionSAC

from env_exp import SocTwoEnv


def eval_with_random_agent(net_striker,
                           net_goalie,
                           env,
                           device,
                           eval_epsoid=40):
    obs_striker, obs_goalie = env.reset('team')
    # time.sleep(5)
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

        actions_striker = torch.cat([
            torch.LongTensor(np.random.randint(0, 7, (8, 1))),
            actions_striker[8:],
        ],
                                    dim=0)
        actions_goalie = torch.cat([
            torch.LongTensor(np.random.randint(0, 5, (8, 1))),
            actions_goalie[8:],
        ],
                                   dim=0)
        # actions_striker = torch.cat([
        #     actions_striker[:8],
        #     torch.LongTensor(np.random.randint(0, 7, (8, 1)))
        # ],
        #                             dim=0)
        # actions_goalie = torch.cat([
        #     actions_goalie[:8],
        #     torch.LongTensor(np.random.randint(0, 5, (8, 1)))
        # ],
        #                            dim=0)

        obs, rewards, dones, _ = env.step(actions_striker, actions_goalie,
                                          'team')
        obs_striker, obs_goalie = obs

        rewards_striker = torch.from_numpy(
            rewards[0]).float().unsqueeze(1).to(device)
        rewards_goalie = torch.from_numpy(
            rewards[1]).float().unsqueeze(1).to(device)

        for i in np.argwhere(dones[0]).flatten():
            epsoid += 1
    return


def eval_self_complete(net_striker,
                       net_goalie,
                       env,
                       device,
                       order='team',
                       eval_epsoid=40):
    obs_striker, obs_goalie = env.reset(order)
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

        obs, rewards, dones, _ = env.step(actions_striker, actions_goalie,
                                          order)
        obs_striker, obs_goalie = obs

        rewards_striker = torch.from_numpy(
            rewards[0]).float().unsqueeze(1).to(device)
        rewards_goalie = torch.from_numpy(
            rewards[1]).float().unsqueeze(1).to(device)

        for i in np.argwhere(dones[0]).flatten():
            epsoid += 1
    return


def eval_self_striker_goalie(net_striker,
                             net_goalie,
                             env,
                             device,
                             order='team',
                             eval_epsoid=40):
    obs_striker, obs_goalie = env.reset(order)
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

        actions_striker = torch.cat([
            actions_striker[:8],
            torch.LongTensor(np.random.randint(0, 7, (8, 1)))
        ],
                                    dim=0)
        actions_goalie = torch.cat([
            torch.LongTensor(np.random.randint(0, 5,
                                               (8, 1))), actions_goalie[8:]
        ],
                                   dim=0)

        obs, rewards, dones, _ = env.step(actions_striker, actions_goalie,
                                          order)
        obs_striker, obs_goalie = obs

        rewards_striker = torch.from_numpy(
            rewards[0]).float().unsqueeze(1).to(device)
        rewards_goalie = torch.from_numpy(
            rewards[1]).float().unsqueeze(1).to(device)

        for i in np.argwhere(dones[0]).flatten():
            epsoid += 1
    return


def eval_agents_compete(strikers,
                        goalies,
                        env,
                        device,
                        order='team',
                        eval_epsoid=40):
    obs_striker, obs_goalie = env.reset(order)
    policies_striker = [None, None]
    policies_goalie = [None, None]
    # time.sleep(5)
    records = [0] * 3

    epsoid = 0
    while epsoid < eval_epsoid:
        obs_striker = Variable(
            torch.from_numpy(obs_striker).float()).to(device)
        obs_goalie = Variable(torch.from_numpy(obs_goalie).float()).to(device)

        policies_striker[0], _ = strikers[0](obs_striker[:8])
        policies_goalie[0], _ = goalies[0](obs_goalie[:8])
        policies_striker[1], _ = strikers[1](obs_striker[8:])
        policies_goalie[1], _ = goalies[1](obs_goalie[8:])

        policy_strikers = torch.cat(policies_striker, dim=0)
        policy_goalies = torch.cat(policies_goalie, dim=0)

        probs_striker = F.softmax(policy_strikers, dim=-1)
        probs_goalie = F.softmax(policy_goalies, dim=-1)

        actions_striker = probs_striker.multinomial(1).data
        actions_goalie = probs_goalie.multinomial(1).data

        obs, rewards, dones, _ = env.step(actions_striker, actions_goalie,
                                          order)
        obs_striker, obs_goalie = obs

        rewards_striker = torch.from_numpy(
            rewards[0]).float().unsqueeze(1).to(device)
        rewards_goalie = torch.from_numpy(
            rewards[1]).float().unsqueeze(1).to(device)

        for i in np.argwhere(dones[0][:8]).flatten():
            epsoid += 1
            if rewards[1][i + 8] < 0:
                records[0] += 1
            elif rewards[0][i] < 0:
                records[1] += 1
            else:
                records[2] += 1

    print(records)
    return


def eval_compete_acppo(strikers,
                       goalies,
                       env,
                       device,
                       order='team',
                       eval_epsoid=40):
    # env.train()
    obs_striker, obs_goalie = env.reset(order)
    policies_striker = [None, None]
    policies_goalie = [None, None]
    # time.sleep(5)
    records = [0] * 3

    epsoid = 0
    while epsoid < eval_epsoid:
        obs_striker = Variable(
            torch.from_numpy(obs_striker).float()).to(device)
        obs_goalie = Variable(torch.from_numpy(obs_goalie).float()).to(device)

        policies_striker[0], _ = strikers[0](obs_striker[:8])
        policies_goalie[0], _ = goalies[0](obs_goalie[:8])
        # policies_striker[1], _ = strikers[1](obs_striker[8:])
        # policies_goalie[1], _ = goalies[1](obs_goalie[8:])
        action_ppo_striker = strikers[1].act(obs_striker[8:])
        action_ppo_goalie = goalies[1].act(obs_goalie[8:])

        policy_strikers = policies_striker[0]
        policy_goalies = policies_goalie[0]

        probs_striker = F.softmax(policy_strikers, dim=-1)
        probs_goalie = F.softmax(policy_goalies, dim=-1)

        actions_striker = probs_striker.multinomial(1).data
        actions_goalie = probs_goalie.multinomial(1).data

        # print(actions_striker)
        actions_striker = torch.cat((actions_striker, action_ppo_striker),
                                    dim=0)
        actions_goalie = torch.cat((actions_goalie, action_ppo_goalie), dim=0)
        # random_act_striker = torch.LongTensor(np.random.randint(7, size=(8,1)))
        # random_act_goalie = torch.LongTensor(np.random.randint(5, size=(8,1)))
        # actions_striker = torch.cat((random_act_striker, action_ppo_striker), dim=0)
        # actions_goalie = torch.cat((random_act_goalie, action_ppo_goalie), dim=0)

        obs, rewards, dones, _ = env.step(actions_striker, actions_goalie,
                                          order)
        obs_striker, obs_goalie = obs

        rewards_striker = torch.from_numpy(
            rewards[0]).float().unsqueeze(1).to(device)
        rewards_goalie = torch.from_numpy(
            rewards[1]).float().unsqueeze(1).to(device)

        for i in np.argwhere(dones[0][:8]).flatten():
            epsoid += 1
            if rewards[1][i + 8] < 0:
                records[0] += 1
            elif rewards[1][i] < 0:
                records[1] += 1
            else:
                records[2] += 1

    print(records)
    return


def eval_agents_compete_(strikers,
                         goalies,
                         env,
                         device,
                         order='team',
                         eval_epsoid=40):
    obs_striker, obs_goalie = env.reset(order)
    actions_strikers = [None, None]
    actions_goalies = [None, None]
    records = [0, 0, 0]

    epsoid = 0
    while epsoid < eval_epsoid:
        obs_striker = Variable(
            torch.from_numpy(obs_striker).float()).to(device)
        obs_goalie = Variable(torch.from_numpy(obs_goalie).float()).to(device)

        actions_strikers[0], _ = strikers[0](obs_striker[:8])
        actions_goalies[0], _ = goalies[0](obs_goalie[:8])
        actions_strikers[1], _ = strikers[1](obs_striker[8:])
        actions_goalies[1], _ = goalies[1](obs_goalie[8:])

        actions_striker = torch.cat(actions_strikers, 0)
        actions_goalie = torch.cat(actions_goalies, 0)

        obs, rewards, dones, _ = env.step(actions_striker, actions_goalie,
                                          order)
        obs_striker, obs_goalie = obs

        for i in np.argwhere(dones[0]).flatten():
            epsoid += 1
            if rewards[1][i] < 0:
                records[0] += 1
            elif rewards[0][i] < 0:
                records[1] += 1
            else:
                records[2] += 1

    return


def eval_maac_with_random(model_path, env, order='team', eval_epsoid=40):
    maac = AttentionSAC.init_from_save(model_path)
    obs_striker, obs_goalie = env.reset(order)

    actions_strikers = [None, None]
    actions_goalies = [None, None]
    records = [0, 0, 0]

    epsoid = 0
    while epsoid < eval_epsoid:
        obs_striker = Variable(
            torch.from_numpy(obs_striker).float()).to(device)
        obs_goalie = Variable(torch.from_numpy(obs_goalie).float()).to(device)

        action_maac = maac.step((obs_striker, obs_goalie), explore=True)

        # print(action_maac)

        actions_strikers[0] = torch.argmax(action_maac[0][:8], dim=-1)
        actions_goalies[0] = torch.argmax(action_maac[1][:8], dim=-1)
        # print(actions_strikers[0])

        actions_strikers[1] = torch.randint(7, size=(8, ))
        actions_goalies[1] = torch.randint(5, size=(8, ))

        # print(actions_strikers)

        actions_striker = torch.cat(actions_strikers, 0)
        actions_goalie = torch.cat(actions_goalies, 0)

        obs, rewards, dones, _ = env.step(actions_striker, actions_goalie,
                                          order)
        obs_striker, obs_goalie = obs

        for i in np.argwhere(dones[0]).flatten():
            epsoid += 1
            if rewards[1][i] < 0:
                records[0] += 1
            elif rewards[0][i] < 0:
                records[1] += 1
            else:
                records[2] += 1

    return


def eval_maac_self_compete(model_path, env, order='team', eval_epsoid=40):
    maac = AttentionSAC.init_from_save(model_path)
    obs_striker, obs_goalie = env.reset(order)

    actions_strikers = [None, None]
    actions_goalies = [None, None]
    records = [0, 0, 0]

    epsoid = 0
    while epsoid < eval_epsoid:
        obs_striker = Variable(
            torch.from_numpy(obs_striker).float()).to(device)
        obs_goalie = Variable(torch.from_numpy(obs_goalie).float()).to(device)

        action_maac = maac.step((obs_striker, obs_goalie), explore=True)

        # print(action_maac)

        actions_strikers[0] = torch.argmax(action_maac[0], dim=-1)
        actions_goalies[0] = torch.argmax(action_maac[1], dim=-1)
        # print(actions_strikers[0])

        # print(actions_strikers)

        # actions_striker = torch.cat(actions_strikers, 0)
        # actions_goalie = torch.cat(actions_goalies, 0)

        obs, rewards, dones, _ = env.step(actions_strikers[0],
                                          actions_goalies[0], order)
        obs_striker, obs_goalie = obs

        for i in np.argwhere(dones[0]).flatten():
            epsoid += 1
            if rewards[1][i] < 0:
                records[0] += 1
            elif rewards[0][i] < 0:
                records[1] += 1
            else:
                records[2] += 1

    return


def eval_maacac_compete(model_path,
                        strikers,
                        goalies,
                        env,
                        order='team',
                        eval_epsoid=200):
    maac = AttentionSAC.init_from_save(model_path)
    obs_striker, obs_goalie = env.reset(order)

    actions_strikers = [None, None]
    actions_goalies = [None, None]
    records = [0, 0, 0]

    epsoid = 0
    while epsoid < eval_epsoid:
        obs_striker = Variable(
            torch.from_numpy(obs_striker).float()).to(device)
        obs_goalie = Variable(torch.from_numpy(obs_goalie).float()).to(device)

        action_maac = maac.step((obs_striker, obs_goalie), explore=True)

        # print(action_maac)

        actions_strikers[0] = torch.argmax(action_maac[0][:8], dim=-1)
        actions_goalies[0] = torch.argmax(action_maac[1][:8], dim=-1)
        # print(actions_strikers[0])

        policy_strikers, _ = strikers(obs_striker[8:])
        policy_goalies, _ = goalies(obs_goalie[8:])

        probs_striker = F.softmax(policy_strikers, dim=-1)
        probs_goalie = F.softmax(policy_goalies, dim=-1)

        actions_strikers[1] = probs_striker.multinomial(1).data.flatten()
        actions_goalies[1] = probs_goalie.multinomial(1).data.flatten()

        # print(actions_strikers)

        actions_striker = torch.cat(actions_strikers, 0)
        actions_goalie = torch.cat(actions_goalies, 0)

        obs, rewards, dones, _ = env.step(actions_striker, actions_goalie,
                                          order)
        obs_striker, obs_goalie = obs

        for i in np.argwhere(dones[0]).flatten():
            epsoid += 1
            if rewards[1][i] < 0:
                records[0] += 1
            elif rewards[0][i] < 0:
                records[1] += 1
            else:
                records[2] += 1

    return


if __name__ == '__main__':
    env_path = './env/macos/SoccerTwosBeta.app'
    env = SocTwoEnv(env_path, worker_id=0, train_mode=False, render=True)
    # net_path = './a2c/ckpt/a2c_step20320000.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_path = './a2c/ckpt_reward_shaping/a2c_step39960000.pth'
    # net_path_large = './a2c/ckpt_rs_large/a2cLarge_step36960000.pth'
    # net_path_large = './a2c/ckpt_rs_large/a2cLarge_step36960000.pth'
    net_path_large = './a2c/ckpt_wors_2e/a2cLarge_step39960000.pth'
    # net_path_large2 = './a2c/ckpt_wors_2e/a2cLarge_step13920000.pth'
    net_path_large2 = './a2c/ckpt_rs_large/a2cLarge_step36960000.pth'

    ppo_striker = './ppo/ckpt/PPO_strikerSoccerTwos_9920.pth'
    ppo_goalie = './ppo/ckpt/PPO_goalieSoccerTwos_9920.pth'

    maac_path = './maac/server/model.pt'
    # maac_path = './maac/models/maac/run10/model.pt'
    # maac_path = './maac/models/maac/run14/model.pt'

    with torch.no_grad():
        # policy_striker, policy_goalie = A2C(7).to(device), A2C(5).to(device)
        policy_striker_large, policy_goalie_large, = A2CLarge(7).to(
            device), A2CLarge(5).to(device)
        # policy_striker_large2, policy_goalie_large2, = A2CLarge(7).to(
        #     device), A2CLarge(5).to(device)

        ckpt_large = torch.load(net_path_large, map_location=device)
        policy_striker_large.load_state_dict(ckpt_large['striker_a2c'])
        policy_goalie_large.load_state_dict(ckpt_large['goalie_a2c'])

        # ckpt_large2 = torch.load(net_path_large2, map_location=device)
        # policy_striker_large2.load_state_dict(ckpt_large2['striker_a2c'])
        # policy_goalie_large2.load_state_dict(ckpt_large2['goalie_a2c'])

        # ckpt = torch.load(net_path, map_location=device)
        # policy_striker.load_state_dict(ckpt['striker_a2c'])
        # policy_goalie.load_state_dict(ckpt['goalie_a2c'])

        ppo_striker = PPO(112, 7, 64, ckpt_path=ppo_striker)
        ppo_goalie = PPO(112, 5, 64, ckpt_path=ppo_goalie)

        policy_striker_large.eval()
        policy_goalie_large.eval()

        # policy_striker.eval()
        # policy_goalie.eval()

        # eval_with_random_agent(policy_striker,
        #                        policy_goalie,
        #                        env,
        #                        device,
        #                        eval_epsoid=100)

        # eval_with_random_agent(policy_striker_large,
        #                        policy_goalie_large,
        #                        env,
        #                        device,
        #                        eval_epsoid=100)

        # eval_self_striker_goalie(policy_striker_large,
        #                          policy_goalie_large,
        #                          env,
        #                          device,
        #                          eval_epsoid=100)

        # eval_self_complete(policy_striker, policy_goalie, env, device, 'team')

        # eval_self_complete(policy_striker_large, policy_striker_large, env,
        #                    device, 'team')
        # eval_self_complete(policy_striker_large2, policy_striker_large2, env,
        #                    device, 'team')

        # eval_agents_compete([policy_striker_large, policy_striker],
        #                     [policy_goalie_large, policy_goalie],
        #                     env,
        #                     device,
        #                     order='team',
        #                     eval_epsoid=100)

        # eval_agents_compete([policy_striker_large, policy_striker_large2],
        #                     [policy_goalie_large, policy_goalie_large2],
        #                     env,
        #                     device,
        #                     order='team',
        #                     eval_epsoid=100)

        # eval_compete_acppo([policy_striker_large, ppo_striker],
        #                     [policy_goalie_large, ppo_goalie],
        #                     env,
        #                     device,
        #                     order='team',
        #                     eval_epsoid=100)
        # eval_maac_with_random(maac_path, env)
        # eval_maac_self_compete(maac_path, env)
        eval_maacac_compete(maac_path, policy_striker_large,policy_goalie_large,env)
    pass