import torch
import os
import gym
import time
import logging
from utils import load_config
import torch.multiprocessing as mp
import numpy as np
from ActorCritic import ActorCritic
from rtc_env import GymEnv

CONFIG_FILE = ''
TRAIN_TRACES = ''
TEST_TRACES = ''

# todo: log files needs to be created
def central_agent(net_params_queue, exp_queues, config):

    start = time.time()

    # log training info
    logging.basicConfig(filename=config['log_dir'] + '/Central_agent_training.log', filemode='w', level=logging.INFO)

    assert len(net_params_queue) == config['num_agents']
    assert len(exp_queues) == config['num_agents']

    net = ActorCritic(True, config)

    # since the original pensieve does not use critic in workers
    # push actor_net_params into net_params_queue only, and save parameters regarding both networks separately
    if config['load_model']:
        actor_net_params = torch.load(config['saved_actor_model_path'])
        critic_net_params = torch.load(config['saved_critic_model_path'])
        net.ActorNetwork.load_state_dict(actor_net_params)
        net.CriticNetwork.load_state_dict(critic_net_params)
    else:
        net.ActorNetwork.init_params()
        net.CriticNetwork.init_params()
        actor_net_params = net.ActorNetwork.parameters()

    epoch = 0
    total_reward = 0.0
    total_batch_len = 0.0
    total_agents = 0
    # total_entropy = 0.0

    while True:
        for i in range(config['num_agents']):
            net_params_queue[i].put(actor_net_params)

        for i in range(config['num_agents']):
            s_batch, a_batch, r_batch, done = exp_queues[i].get()

            net.getNetworkGradient(s_batch, a_batch, r_batch, done)

            total_reward += np.sum(r_batch)
            total_batch_len += len(r_batch)

        net.updateNetwork()
        epoch += 1
        avg_reward = total_reward / config['num_agents']

        logging.info('Epoch ' + str(epoch) +
                     'Average reward: ' + str(avg_reward))

        if epoch % config['save_interval'] == 0:
            print('Train Epoch ' + str(epoch) + ', Model restored.')
            torch.save(net.ActorNetwork.state_dict(), config['model_dir'] + '/actor_' + str(epoch) + '.pt')
            torch.save(net.CriticNetwork.state_dict(), config['model_dir'] + '/critic_' + str(epoch) + '.pt')


def agent(net_params_queue, exp_queues, config, id):
    env = GymEnv(env_id=id, config=config)
    env.reset()

    net = ActorCritic(False, config)

    actor_network_params = net_params_queue.get()
    net.ActorNetwork.load_state_dict(actor_network_params)

    bwe = config['sending_rate'][config['default_bwe']]

    s_batch = []
    a_batch = []
    r_batch = []

    # experience RTC if not forced to stop
    while True:
        # todo: Agent interact with gym
        state, reward, done, _ = env.step(bwe)  # todo: the shape of state needs to be regulated

        r_batch.append(reward)

        action = net.predict(state)
        bwe = config['sending_rate'][action]

        # ignore the first bwe and state since we don't have the ability to control it
        if len(r_batch) >= config['train_seq_length'] or done:
            exp_queues.put([s_batch,
                            a_batch,
                            r_batch,
                            done])

            # synchronize the network parameters from the coordinator
            actor_network_params = net_params_queue.get()
            net.ActorNetwork.load_state_dict(actor_network_params)

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

        # todo: need to be fixed
        if done:
            action = config['default_bwe']
            bwe = config['sending_rate'][action]
            s_batch.append(np.zeros(config['state_dim'], config['state_length']))
            a_batch.append(action)
            env.reset()
        else:
            s_batch.append(state)
            a_batch.append(action)

    pass


def main():
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
                                 args=(net_params_queue, exp_queues, config, i)))

    for i in range(num_agents):
        agents[i].start()

    # wait until training is done
    coordinator.join()

if __name__ == '__main__':
    main()