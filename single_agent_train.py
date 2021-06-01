import time
import logging
import torch
from utils import load_config
import torch.multiprocessing as mp
import numpy as np
from ActorCritic import ActorCritic
from rtc_env import GymEnv
import matplotlib.pyplot as plt

def single_agent():
    config = load_config()
    # num_agents = config['num_agents']
    torch.set_num_threads(1)

    env = GymEnv(config=config)
    env.reset()

    net = ActorCritic(True, config)
    net.ActorNetwork.init_params()
    net.CriticNetwork.init_params()

    bwe = config['sending_rate'][config['default_bwe']]

    i = 1
    s_batch = []
    r_batch = []
    a_batch = []

    # experience RTC if not forced to stop
    ax = []
    ay = []
    plt.ion()
    while True:
        # todo: Agent interact with gym
        state, reward, done, _ = env.step(bwe)

        r_batch.append(reward)

        action, entropy = net.predict(state)
        bwe = config['sending_rate'][action]
        a_batch.append(action)
        s_batch.append(state)

        # todo: need to be fixed
        if done:
            action = config['default_bwe']
            bwe = config['sending_rate'][action]
            # update network
            net.getNetworkGradient(s_batch, a_batch, r_batch, done)
            net.updateNetwork()
            print('Network update.')

            i += 1
            ax.append(i)
            # ay.append(entropy)
            ay.append(reward)
            plt.clf()
            plt.plot(ax, ay)
            plt.pause(0.1)
            # s_batch.append(np.zeros(config['state_dim'], config['state_length']))
            # a_batch.append(action)
            env.reset()
            print('Environment has been reset.')
            print('Epoch {}, Reward: {}'.format(i - 1, reward))
        if i % 100 == 0:
            # print('Current BWE: ' + str(bwe))
            torch.save(net.ActorNetwork.state_dict(), config['model_dir'] + '/actor1_{}.pt'.format(str(i)))
            torch.save(net.CriticNetwork.state_dict(), config['model_dir'] + '/critic13m_{}.pt'.format(str(i)))
            print('Model Restored.')
        # else:
        #     s_batch.append(state)
        #     a_batch.append(action)

if __name__ == '__main__':
    single_agent()