import torch
import os
import gym
import datetime
import time
import logging
from utils import load_config
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from ActorCritic_ppo import ActorCritic
from rtc_env_ppo import GymEnv
import matplotlib.pyplot as plt


# todo: log files needs to be created
def central_agent(net_params_queue, exp_queues, config):
    torch.set_num_threads(1)

    # log training info
    logging.basicConfig(filename=config['log_dir'] + '/Central_agent_training.log', filemode='w', level=logging.INFO)

    assert len(net_params_queue) == config['num_agents']
    assert len(exp_queues) == config['num_agents']

    net = ActorCritic(True, config)
    net.Network.to(config['device'])
    net.oldNetwork.to(config['device'])

    # since the original pensieve does not use critic in workers
    # push actor_net_params into net_params_queue only, and save parameters regarding both networks separately
    if config['load_model']:
        net_params = torch.load(config['model_dir'] + '/actor_300k5_40.pt')
        net.Network.load_state_dict(net_params)
        net.oldNetwork.load_state_dict(net_params)

    net_params = list(net.Network.parameters())
    for i in range(config['num_agents']):
        # actor_net_params = net.ActorNetwork.parameters()
        net_params_queue[i].put(net_params)

    epoch = 0
    total_reward = 0.0
    total_batch_len = 0.0
    episode_entropy = 0.0
    ax = []
    ay = []
    plt.ion()

    while True:
        start = time.time()
        net_params = list(net.Network.parameters())
        for i in range(config['num_agents']):
            net_params_queue[i].put(net_params)

        for i in range(config['num_agents']):
            s_batch, a_batch, a_log_batch, v_batch, r_batch, return_batch = exp_queues[i].get()

            policy_loss, value_loss = net.updateNetwork(s_batch, a_batch, a_log_batch, v_batch, return_batch)
            # net.oldNetwork.load_state_dict(net.Network.state_dict())

            total_reward += np.sum(r_batch)
            total_batch_len += len(r_batch)
            # episode_entropy += np.sum(e_batch) / total_batch_len

        # net.updateNetwork()
        epoch += 1
        avg_reward = total_reward / total_batch_len
        # avg_entropy = total_entropy / total_batch_len

        logging.info('Epoch ' + str(epoch) +
                     '\nAverage reward: ' + str(avg_reward))
        ax.append(epoch)
        ay.append(avg_reward)
        plt.clf()
        plt.plot(ax, ay)
        plt.pause(0.1)

        total_reward = 0.0
        total_batch_len = 0
        # episode_entropy = 0.0

        if epoch % config['save_interval'] == 0:
            print('Train Epoch ' + str(epoch) + ', Model restored.')
            print('Epoch costs ' + str(time.time() - start) + ' seconds.')
            torch.save(net.Network.state_dict(), config['model_dir'] + '/actor_300k_' + str(epoch) + '.pt')


def agent(net_params_queue, exp_queues, config, id):
    torch.set_num_threads(1)

    env = GymEnv(env_id=id, config=config)

    net = ActorCritic(False, config)
    net.Network.to(config['device'])
    net.oldNetwork.to(config['device'])

    for param in net.Network.parameters():
        param.requires_grad = False
    net.Network.eval()

    for param in net.oldNetwork.parameters():
        param.requires_grad = False
    net.oldNetwork.eval()

    # experience RTC if not forced to stop
    while True:
        env.reset()
        bwe = 100000 * 1.0
        s_batch = []
        r_batch = []
        a_batch = []
        v_batch = []
        a_log_batch = []
        is_terminal = []

        network_params = net_params_queue.get()
        for target_param, source_param in zip(net.Network.parameters(), network_params):
            target_param.data.copy_(source_param.data)
        net.oldNetwork.load_state_dict(net.Network.state_dict())

        # todo: Agent interact with gym
        while True:
            state, reward, done, gcc_estimation = env.step(bwe)  # todo: the shape of state needs to be regulated

            state = torch.Tensor(state)

            r_batch.append(reward)

            action, action_logprobs, value = net.predict(state)
            bwe = action * gcc_estimation
            s_batch.append(state)
            a_batch.append(action)
            a_log_batch.append(action_logprobs)
            v_batch.append(value)
            is_terminal.append(done)

            if done:
                return_batch = []
                R_batch = np.zeros(len(r_batch) + 1)
                R_batch[-1] = net.getValue(state)
                for t in reversed(range(len(r_batch))):
                    R_batch[t] = R_batch[t + 1] * config['discount_factor'] * (1 - is_terminal[t]) + r_batch[t]
                    return_batch.append(torch.unsqueeze(torch.tensor(R_batch[t]), 0))
                return_batch.reverse()
            # ignore the first bwe and state since we don't have the ability to control it
            # s_batch, a_batch, a_log_batch, v_batch, r_batch, return_batch
                exp_queues.put([s_batch[1:],
                                a_batch[1:],
                                a_log_batch[1:],
                                v_batch[1:],
                                r_batch[1:],
                                return_batch[1:]])
                break


def main():
    start = time.time()
    config = load_config()
    num_agents = config['num_agents']

    net_params_queue = []
    exp_queues = []
    for i in range(num_agents):
        net_params_queue.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # coordinator = Network(config)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queue, exp_queues, config))
    coordinator.start()

    agents = []
    for i in range(num_agents):
        agents.append(mp.Process(target=agent,
                                 args=(net_params_queue[i], exp_queues[i], config, i)))

    for i in range(num_agents):
        agents[i].start()

    # wait until training is done
    coordinator.join()

    for i in range(num_agents):
        agents[i].join()

    print(str(time.time() - start))


if __name__ == '__main__':
    main()
