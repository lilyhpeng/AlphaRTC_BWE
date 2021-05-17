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

def central_agent(net_params_queue, exp_queues, config):

    assert len(net_params_queue) == config['num_agents']
    assert len(exp_queues) == config['num_agents']

    net = Network(config)

    if config['load_model']:
        saved_params = torch.load(config['saved_model_path'])
        net.load_state_dict(saved_params)

    epoch = 0

    while True:
        #todo: Central_agent update

        pass

def agent(net_params_queue, exp_queues, config):

    env = GymEnv()

    net = Network(config)

    network_params = net_params_queue.get()
    net.load_state_dict(network_params)

    bwe = 1000

    s_batch = [np.zeros(config['state_dim'],config['state_length'])]
    a_batch = [np.zeros(config['action_dim'])]
    r_batch = []
    entropy_record = []

    # experience RTC
    while True:
        #todo: Agent interact with gym


        pass

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