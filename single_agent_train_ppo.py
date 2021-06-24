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
    # torch.set_num_threads(1)

    env = GymEnv(config=config)
    env.reset()

    net = ActorCritic(True, config)
    net.Network.to(config['device'])
    net.oldNetwork.to(config['device'])
    # net.ActorNetwork.init_params()
    # net.CriticNetwork.init_params()
    # net.oldCriticNetwork.load_state_dict(net.CriticNetwork.state_dict())
    # net.oldActorNetwork.load_state_dict(net.ActorNetwork.state_dict())
    i = 1
    s_batch = []
    r_batch = []
    a_batch = []
    v_batch = []
    a_log_batch = []
    is_terminal = []
    return_batch = []

    # experience RTC if not forced to stop
    ax = []
    ay = []
    # plt.ion()
    while i < 1000:
        time_step = 0
        episode_reward = 0
        while time_step < config['update_interval']:
            use_action_counter = 0
            done = False
            state = torch.Tensor(env.reset())

            while not done and time_step < config['update_interval']:
                # todo: Agent interact with gym
                if use_action_counter == 5:
                    action, action_logprobs, value = net.predict(state)
                    use_action_counter = 0
                    use_action = True
                else:
                    action = 1.0
                    use_action = False
                    use_action_counter += 1
                state, reward, done = env.step(action, use_action)  # todo: the shape of state needs to be regulated

                if use_action:
                    r_batch.append(reward)
                    episode_reward += reward

                    state = torch.Tensor(state)
                    s_batch.append(state)
                    a_batch.append(action)
                    a_log_batch.append(action_logprobs)
                    v_batch.append(value)
                    is_terminal.append(done)
                    time_step += 1

                # time_list.append(time_step * 60)
                # bwe_list.append(bandwidth_prediction)
        R_batch = np.zeros(len(r_batch) + 1)
        R_batch[-1] = net.getValue(state)
        for t in reversed(range(len(r_batch))):
            R_batch[t] = R_batch[t + 1] * config['discount_factor'] * (1 - is_terminal[t]) + r_batch[t]
            return_batch.append(torch.unsqueeze(torch.tensor(R_batch[t], dtype=torch.float32), 0))
        return_batch.reverse()
        episode_reward /= time_step

        # update network
        policy_loss, value_loss = net.updateNetwork(s_batch, a_batch, a_log_batch, v_batch, return_batch)
        # net.oldNetwork.load_state_dict(net.Network.state_dict())
        is_terminal = []
        s_batch = []
        a_batch = []
        a_log_batch = []
        v_batch = []
        r_batch = []
        return_batch = []

        ax.append(i)
        i += 1
        # ay.append(entropy)
        ay.append(episode_reward)
        print('Epoch {}, Reward: {}, Policy loss: {}, Value loss: {}'.format(i - 1, episode_reward, policy_loss, value_loss))

        if i % 50 == 0:
             # print('Current BWE: ' + str(bwe))
            torch.save(net.Network.state_dict(), config['model_dir'] + '/Net_exp.pt')
            print('Model Restored.')
            plt.xlabel('Epoch')
            plt.ylabel('Avg Reward')
            plt.clf()
            plt.plot(ax, ay)
            plt.savefig('Epoch_50.png')


if __name__ == '__main__':
    single_agent()