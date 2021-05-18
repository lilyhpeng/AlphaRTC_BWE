import torch
import os
import gym
from config import load_config
import torch.multiprocessing as mp
import numpy as np
from network import Network
from rtc_env import GymEnv

CONFIG_FILE = ''
TRAIN_TRACES = ''
TEST_TRACES = ''

# todo: log files needs to be created

def central_agent(net_params_queue, exp_queues, config):

    assert len(net_params_queue) == config['num_agents']
    assert len(exp_queues) == config['num_agents']

    net = Network(config)

    if config['load_model']:
        net_params = torch.load(config['saved_model_path'])
        net.load_state_dict(net_params)
    else:
        net.init_net_params()
        net_params = net.state_dict()

    epoch = 0

    while True:
        #todo: Central_agent update
        for i in range(config['num_agents']):
            net_params_queue[i].put(net_params)
        pass

def agent(net_params_queue, exp_queues, config):
    # todo: gym_id has not considered
    env = GymEnv()
    env.reset()

    net = Network(config)

    network_params = net_params_queue.get()
    net.load_state_dict(network_params)

    bwe = config['default_bwe']

    state = np.zeros(config['state_dim'], config['state_length'])
    s_batch = [np.zeros(config['state_dim'], config['state_length'])]
    a_batch = []
    r_batch = []

    # time_step = 0
    # reward = 0
    # done = False

    # experience RTC if not forced to stop
    while True:
        # todo: Agent interact with gym
        # while not done:
        np.roll(state, -1, axis=1)

        receiving_rate, delay, loss_ratio, done = env.step(bwe)

        # todo: Regularization
        state[0, -1] = receiving_rate
        state[1, -1] = delay
        state[2, -1] = loss_ratio

        reward = receiving_rate - delay - loss_ratio   # todo: redesign linear reward function

        # s_batch.append(state)
        # a_batch.append(bwe)
        r_batch.append(reward)

        bwe, value = net.forward(state)

        if len(r_batch) >= config['train_seq_length'] or done:
            exp_queues.put([s_batch[1:],
                            a_batch[1:],
                            r_batch[1:],
                            done])      # ignore the first bwe and state since we don't have the ability to control it

            # synchronize the network parameters from the coordinator
            network_params = net_params_queue.get()
            net.load_state_dict(network_params)

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

        # todo: need to be fixed
        if done:
            bwe = config['default_bwe']
            s_batch.append(np.zeros(config['state_dim'], config['state_length']))
            a_batch.append(bwe)
            env.reset() # todo: does it necessary?
        else:
            s_batch.append(state)
            a_batch.append(bwe)

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
                                 args=(net_params_queue, exp_queues, config)))

    for i in range(num_agents):
        agents[i].start()

    # wait until training is done
    coordinator.join()

    # if config['load_model']:
    #     saved_params = torch.load(config['saved_model_path'])
    #     coordinator.load_state_dict(saved_params)
    # coordinator.share_memory()
    # optimizer = torch.optim.RMSprop(params=coordinator.parameters(), lr=config['learning_rate'])


    # training loop
    # for episode in range(max_num_episodes):
    #     while time_step < update_interval:
    #         done = False
    #         state = torch.Tensor(env.reset())
    #         while not done and time_step < update_interval:
    #             action = ppo.select_action(state, storage)
    #             state, reward, done, _ = env.step(action)
    #             state = torch.Tensor(state)
    #             # Collect data for update
    #             storage.rewards.append(reward)
    #             storage.is_terminals.append(done)
    #             time_step += 1
    #             episode_reward += reward

if __name__ == '__main__':
    main()