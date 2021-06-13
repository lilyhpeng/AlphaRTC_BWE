import datetime
import time
import logging
import torch
from utils import load_config
import torch.multiprocessing as mp
import numpy as np
from ActorCritic_ppo import ActorCritic
from rtc_env_ppo import GymEnv
import matplotlib.pyplot as plt

def single_agent():
    config = load_config()
    # num_agents = config['num_agents']
    torch.set_num_threads(1)

    env = GymEnv(config=config)
    env.reset()

    net = ActorCritic(True, config)
    net.Network.to(config['device'])
    net.oldNetwork.to(config['device'])
    # net.ActorNetwork.init_params()
    # net.CriticNetwork.init_params()
    # net.oldCriticNetwork.load_state_dict(net.CriticNetwork.state_dict())
    # net.oldActorNetwork.load_state_dict(net.ActorNetwork.state_dict())

    bwe = 1.0 * 300000
    i = 1
    s_batch = []
    r_batch = []
    a_batch = []
    v_batch = []
    a_log_batch = []
    is_terminal = []
    episode_reward = 0
    time_step = 0
    bwe_list = []
    time_list = []

    # experience RTC if not forced to stop
    ax = []
    ay = []
    plt.ion()
    while True:
        # todo: Agent interact with gym
        state, reward, done, gcc_estimation = env.step(bwe)  # todo: the shape of state needs to be regulated

        state = torch.Tensor(state)

        r_batch.append(reward)
        episode_reward += reward

        action, action_logprobs, value = net.predict(state)
        bwe = float(pow(2, (2 * action - 1)) * gcc_estimation)
        time_step += 1
        s_batch.append(state)
        a_batch.append(action)
        a_log_batch.append(action_logprobs)
        v_batch.append(value)
        is_terminal.append(done)

        time_list.append(time_step * 60)
        bwe_list.append(bwe)


        # todo: need to be fixed
        if done:
            return_batch = []
            bwe = 1.0 * 300000
            R_batch = np.zeros(len(r_batch) + 1)
            R_batch[-1] = net.getValue(state)
            for t in reversed(range(len(r_batch))):
                R_batch[t] = R_batch[t + 1] * config['discount_factor'] * (1 - is_terminal[t]) + r_batch[t]
                return_batch.append(torch.unsqueeze(torch.tensor(R_batch[t], dtype=torch.float32), 0))
            return_batch.reverse()
            episode_reward /= time_step
            plt.plot(time_list, bwe_list)
            plt.title('{}'.format(time.time()))
            plt.xlabel('Time(ms)')
            plt.ylabel('BWE(bps)')
            plt.savefig('bwe.png')

            # update network
            policy_loss, value_loss = net.updateNetwork(s_batch[1:], a_batch[1:], a_log_batch[1:], v_batch[1:], return_batch[1:])
            # net.oldNetwork.load_state_dict(net.Network.state_dict())
            is_terminal = []
            s_batch = []
            a_batch = []
            a_log_batch = []
            v_batch = []
            r_batch = []
            bwe_list = []
            time_list = []

            i += 1
            ax.append(i)
            # ay.append(entropy)
            ay.append(episode_reward)
            episode_reward = 0
            time_step = 0
            plt.clf()
            plt.plot(ax, ay)
            plt.pause(0.1)
            # s_batch.append(np.zeros(config['state_dim'], config['state_length']))
            # a_batch.append(action)
            env.reset()
            print('Environment has been reset.')
            print('Epoch {}, Reward: {}, Policy loss: {}, Value loss: {}'.format(i - 1, episode_reward, policy_loss, value_loss))
        if i % 100 == 0:
            # print('Current BWE: ' + str(bwe))
            torch.save(net.Network.state_dict(), config['model_dir'] + '/Net_exp.pt')
            print('Model Restored.')
        # else:
        #     s_batch.append(state)
        #     a_batch.append(action)

if __name__ == '__main__':
    single_agent()