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
from ActorCritic import ActorCritic
from rtc_env import GymEnv
import matplotlib.pyplot as plt


# todo: log files needs to be created
def central_agent(net_params_queue, exp_queues, config):
    torch.set_num_threads(1)

    # log training info
    logging.basicConfig(filename=config['log_dir'] + '/Central_agent_training.log', filemode='w', level=logging.INFO)

    assert len(net_params_queue) == config['num_agents']
    assert len(exp_queues) == config['num_agents']

    net = ActorCritic(True, config)

    # since the original pensieve does not use critic in workers
    # push actor_net_params into net_params_queue only, and save parameters regarding both networks separately
    if config['load_model']:
        actor_net_params = torch.load(config['model_dir'] + '/actor_300k1_80.pt')
        critic_net_params = torch.load(config['model_dir'] + '/critic_300k1_80.pt')
        net.ActorNetwork.load_state_dict(actor_net_params)
        net.CriticNetwork.load_state_dict(critic_net_params)
    else:
        net.ActorNetwork.init_params()
        net.CriticNetwork.init_params()
    #
    actor_net_params = list(net.ActorNetwork.parameters())
    for i in range(config['num_agents']):
        # actor_net_params = net.ActorNetwork.parameters()
        net_params_queue[i].put(actor_net_params)

    epoch = 0
    total_reward = 0.0
    total_batch_len = 0.0
    episode_entropy = 0.0
    ax = []
    ay = []
    plt.ion()

    while True:
        start = time.time()
        actor_net_params = list(net.ActorNetwork.parameters())
        for i in range(config['num_agents']):
            net_params_queue[i].put(actor_net_params)

        for i in range(config['num_agents']):
            s_batch, a_batch, r_batch, done, e_batch = exp_queues[i].get()

            net.getNetworkGradient(s_batch, a_batch, r_batch, done)

            total_reward += np.sum(r_batch)
            total_batch_len += len(r_batch)
            episode_entropy += np.sum(e_batch)

        net.updateNetwork()
        epoch += 1
        avg_reward = total_reward / total_batch_len
        # avg_entropy = total_entropy / total_batch_len

        logging.info('Epoch ' + str(epoch) +
                     '\nAverage reward: ' + str(avg_reward) +
                     '\nEpisode entropy: ' + str(episode_entropy))
        ax.append(epoch)
        ay.append(episode_entropy)
        plt.clf()
        plt.plot(ax, ay)
        plt.pause(0.1)

        total_reward = 0.0
        total_batch_len = 0
        episode_entropy = 0.0

        if epoch % config['save_interval'] == 0:
            print('Train Epoch ' + str(epoch) + ', Model restored.')
            print('Epoch costs ' + str(time.time() - start) + ' seconds.')
            torch.save(net.ActorNetwork.state_dict(), config['model_dir'] + '/actor_300k_' + str(epoch) + '.pt')
            torch.save(net.CriticNetwork.state_dict(), config['model_dir'] + '/critic_300k_' + str(epoch) + '.pt')


def agent(net_params_queue, exp_queues, config, id):
    torch.set_num_threads(1)

    env = GymEnv(env_id=id, config=config)

    net = ActorCritic(False, config)
    send_rate_list = config['sending_rate']
    default_bwe_idx = config['default_bwe']

    # experience RTC if not forced to stop
    while True:
        env.reset()
        action = default_bwe_idx
        bwe = send_rate_list[action]
        s_batch = []
        a_batch = []
        r_batch = []
        entropy_batch = []

        done = False
        actor_network_params = net_params_queue.get()
        for target_param, source_param in zip(net.ActorNetwork.parameters(), actor_network_params):
            target_param.data.copy_(source_param.data)
        # todo: Agent interact with gym
        while not done:
            state, reward, done, _ = env.step(bwe)  # todo: the shape of state needs to be regulated

            r_batch.append(reward)

            action, entropy = net.predict(state)
            bwe = send_rate_list[action]
            s_batch.append(state)
            a_batch.append(action)
            entropy_batch.append(entropy)
        # ignore the first bwe and state since we don't have the ability to control it
        exp_queues.put([s_batch[1:],
                        a_batch[1:],
                        r_batch[1:],
                        done,
                        entropy_batch[1:]])

        # if len(r_batch) >= config['train_seq_length'] or done:
        #     exp_queues.put([s_batch,
        #                     a_batch,
        #                     r_batch,
        #                     done])
        #     action = default_bwe_idx
        #     bwe = send_rate_list[action]
        #     s_batch = []
        #     a_batch = []
        #     r_batch = []
        #     # s_batch.append(np.zeros(config['state_dim'], config['state_length']))
        #     # a_batch.append(action)
        #     env.reset()
        # else:
        #     s_batch.append(state)
        #     a_batch.append(action)



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
