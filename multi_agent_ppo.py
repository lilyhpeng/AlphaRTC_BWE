import torch
import os
import gym
import datetime
import time
from deep_rl.storage import Storage
from deep_rl.ppo_agent_multiv import PPO
import logging
from utils import load_config, TrafficLight, Counter, shared_batch_buffer
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.optim as optim
# torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from ActorCritic_ppo import ActorCritic
from rtc_env_ppo_gcc import GymEnv
import matplotlib.pyplot as plt

# todo: log files needs to be created
def central_agent(net_params_queue, exp_queues, config):
    torch.set_num_threads(1)

    state_dim = 4
    state_length = 10
    action_dim = 1
    exploration_param = 0.05
    lr = 3e-5
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 37
    ppo_clip = 0.2

    # log training info
    logging.basicConfig(filename=config['log_dir'] + '/Central_agent_training.log', filemode='w', level=logging.INFO)

    assert len(net_params_queue) == config['num_agents']
    assert len(exp_queues) == config['num_agents']

    ppo = PPO(state_dim, state_length, action_dim, exploration_param, lr, betas, gamma, K_epochs, ppo_clip)

    # since the original pensieve does not use critic in workers
    # push actor_net_params into net_params_queue only, and save parameters regarding both networks separately
    if config['load_model']:
        net_params = torch.load(config['model_dir'] + '/actor_300k5_40.pt')
        ppo.policy.load_state_dict(net_params)
        ppo.policy_old.load_state_dict(net_params)

    net_params = list(ppo.policy.parameters())
    for i in range(config['num_agents']):
        # actor_net_params = net.ActorNetwork.parameters()
        net_params_queue[i].put(net_params)

    epoch = 0
    total_reward = 0.0
    total_reward_len = 0.0
    episode_entropy = 0.0
    ax = []
    ay = []
    # plt.ion()

    while True:
        start = time.time()
        net_params = list(ppo.policy.parameters())
        for i in range(config['num_agents']):
            net_params_queue[i].put(net_params)

        for i in range(config['num_agents']):
            states, actions, rewards, logprobs, values, returns = exp_queues[i].get()

            policy_loss, val_loss = ppo.update(states, actions, logprobs, values, returns)
            total_reward += sum(rewards)
            total_reward_len += len(rewards)

            epoch += 1
            avg_reward = total_reward / total_reward_len
            # avg_entropy = total_entropy / total_batch_len

            print('Epoch ' + str(epoch) +
                         '\nAverage reward: ' + str(avg_reward))
            # ax.append(epoch)
            # ay.append(avg_reward)
            # plt.clf()
            # plt.plot(ax, ay)
            # plt.pause(0.1)

            total_reward = 0.0
            total_reward_len = 0
        # episode_entropy = 0.0

        if (epoch + 1) % config['save_interval'] == 0:
            print('Train Epoch ' + str(epoch) + ', Model restored.')
            print('Epoch costs ' + str(time.time() - start) + ' seconds.')
            torch.save(ppo.policy.state_dict(), config['model_dir'] + '/actor_' + str(epoch) + '.pth')

def agent(net_params_queue, exp_queues, config, id):
    torch.set_num_threads(1)

    state_dim = 4
    state_length = 10
    action_dim = 1
    exploration_param = 0.05
    lr = 3e-5
    betas = (0.9, 0.999)
    gamma = 0.99
    K_epochs = 37
    ppo_clip = 0.2

    env = GymEnv(env_id=id, config=config)

    storage = Storage()  # used for storing data
    ppo = PPO(state_dim, state_length, action_dim, exploration_param, lr, betas, gamma, K_epochs, ppo_clip)

    # shared_grad_buffers = shared_grad_buffer(ppo.policy)

    for param in ppo.policy.parameters():
        param.requires_grad = False
    ppo.policy.eval()

    for param in ppo.policy_old.parameters():
        param.requires_grad = False
    ppo.policy_old.eval()

    # experience RTC if not forced to stop
    while True:
        network_params = net_params_queue.get()
        for target_param, source_param in zip(ppo.policy.parameters(), network_params):
            target_param.data.copy_(source_param.data)
        ppo.policy_old.load_state_dict(ppo.policy.state_dict())

        # todo: Agent interact with gym
        while True:
            time_step = 0
            time_to_guide = False
            episode_reward = 0
            while time_step < config['update_interval']:
                done = False
                state = torch.Tensor(env.reset())
                last_estimation = 300000
                action = 0
                if storage.is_terminals != []:
                    storage.is_terminals[-1] = True
                while not done and time_step < config['update_interval']:
                    if time_step % 5 == 4:
                        action = ppo.select_action(state, storage)
                        time_to_guide = True

                    state, reward, done, last_estimation = env.step(action, last_estimation, time_to_guide)
                    time_to_guide = False
                    state = torch.Tensor(state)
                    # Collect data for update
                    if time_step % 5 == 4:
                        storage.rewards.append(reward)
                        storage.is_terminals.append(done)
                    time_step += 1
                    episode_reward += reward

            storage.is_terminals[-1] = True
            next_value = ppo.get_value(state)
            storage.compute_returns(next_value, gamma)
            exp_queues.put([storage.states,
                            storage.actions,
                            storage.rewards,
                            storage.logprobs,
                            storage.values,
                            storage.returns])
            storage.clear_storage()
            break

def chief(config, traffic_light, counter, shared_model, shared_batch_buffer):
    num_agents = config['num_agents']
    update_threshold = num_agents - 1
    gamma = config['discount_factor']

    while True:
        time.sleep(1)
        storage = Storage()

        # worker will wait after last loss computation

        if counter.get() < update_threshold:
            for i in range(num_agents):
                storage.states = shared_batch_buffer.buffer[str(i) + 'states']
                storage.actions = shared_batch_buffer.buffer[str(i) + 'actions']
                storage.rewards = shared_batch_buffer.buffer[str(i) + 'rewards']
                next_value = shared_batch_buffer.buffer[str(i) + 'next_value']
                storage.compute_returns(next_value, gamma)
                policy_loss, val_loss = shared_model.update(storage)
        #     for n, p in shared_model.policy.named_parameters():
        #         p._grad = Variable(shared_gradient_buffer.grads[n+'_grad'])
        #     optimizer.step()
        #     counter.reset()
        #     shared_gradient_buffer.reset()

            counter.reset()
            shared_batch_buffer.clear_buffer()
            traffic_light.switch()  # workers start new collecting
            print('Update Network.')

def train(rank, config, traffic_light, counter, shared_model, shared_batch_buffer):
    torch.manual_seed(123)
    env = GymEnv(env_id=rank, config=config)
    state_dim = config['state_dim']
    state_length = config['state_length']
    action_dim = config['action_dim']
    exploration_param = config['exploration_param']
    lr = config['learning_rate']
    betas = config['betas']
    gamma = config['discount_factor']
    K_epochs = config['ppo_epoch']
    ppo_clip = config['ppo_clip']
    # exploration_size = config['exploration_size']

    ppo = PPO(state_dim, state_length, action_dim, exploration_param, lr, betas, gamma, K_epochs, ppo_clip)
    storage = Storage()
    # memory = ReplayMemory(exploration_size)
    # state = env.reset()
    # state = Variable(torch.Tensor(state).unsqueeze(0))

    # done = False
    episode_len = 0

    while True:
        # sync share_model
        ppo.policy.load_state_dict(shared_model.policy.state_dict())
        ppo.policy_old.load_state_dict(ppo.policy.state_dict())

        time_step = 0
        time_to_guide = False
        episode_reward = 0
        signal_init = traffic_light.get()
        while time_step < config['update_interval']:
            done = False
            state = torch.Tensor(env.reset())
            last_estimation = 300000
            action = 0
            if storage.is_terminals != []:
                storage.is_terminals[-1] = True
            while not done and time_step < config['update_interval']:
                if time_step % 5 == 4:
                    action = ppo.select_action(state, storage)
                    time_to_guide = True

                state, reward, done, last_estimation = env.step(action, last_estimation, time_to_guide)
                time_to_guide = False
                state = torch.Tensor(state)
                # Collect data for update
                if time_step % 5 == 4:
                    storage.rewards.append(reward)
                    storage.is_terminals.append(done)
                time_step += 1
                episode_reward += reward

        # storage.is_terminals[-1] = True
        next_value = ppo.get_value(state)
        # storage.compute_returns(next_value, gamma)
        shared_batch_buffer.push_batch(rank, storage.states, storage.actions, storage.rewards, next_value)
        counter.increment()

        # ppo.get_gradient(storage, shared_gradient_buffer)
        # counter.increment()
        storage.clear_storage()

        while traffic_light.get() == signal_init:
            pass



def main():
    # start = time.time()
    config = load_config()
    num_agents = config['num_agents']
    state_dim = config['state_dim']
    state_length = config['state_length']
    action_dim = config['action_dim']
    exploration_param = config['exploration_param']
    lr = config['learning_rate']
    betas = config['betas']
    gamma = config['discount_factor']
    K_epochs = config['ppo_epoch']
    ppo_clip = config['ppo_clip']

    torch.manual_seed(123)

    traffic_light = TrafficLight()
    counter = Counter()

    shared_model = PPO(state_dim, state_length, action_dim, exploration_param, lr, betas, gamma, K_epochs, ppo_clip)

    shared_model.policy.share_memory()

    batch_buffer = shared_batch_buffer()

    # optimizer = optim.Adam(shared_model.policy.parameters(), lr=lr)

    processes = []
    p = mp.Process(target=chief, args=(config, traffic_light, counter, shared_model, batch_buffer))
    p.start()
    processes.append(p)
    for rank in range(num_agents):
        p = mp.Process(target=train, args=(rank, config, traffic_light, counter, shared_model, batch_buffer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    #

    # agents = []
    # net_params_queue = []
    # exp_queues = []
    # for i in range(num_agents):
    #     net_params_queue.append(mp.Queue(1))
    #     exp_queues.append(mp.Queue(1))
    #
    # # coordinator = Network(config)
    # coordinator = mp.Process(target=central_agent,
    #                          args=(net_params_queue, exp_queues, config))
    # coordinator.start()
    # # agents.append(coordinator)
    #
    # for i in range(num_agents):
    #     p = mp.Process(target=agent, args=(net_params_queue[i], exp_queues[i], config, i))
    #     p.start()
    #     agents.append(p)
    #
    # # wait until training is done
    # coordinator.join()
    #
    # for p in agents:
    #     p.join()


if __name__ == '__main__':
    main()
