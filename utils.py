import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from rtc_env import GymEnv
from deep_rl.storage import Storage
from deep_rl.actor_critic import ActorCritic


def load_config():
    config = {
        #todo: add parameters regarding configuration
        'learning_rate' : 0.001,
        'num_agents': 16,
        'save_interval': 100,
        'default_bwe': 1000,
        'train_seq_length': 1000,
        'state_dim': 3,
        'state_length': 10,
        'action_dim': 1,
        'device': 'cpu',
        'load_model': False,
        'saved_model_path': '',
        'layer1_shape': 256,
        'layer2_shape': 256,

        'trace_dir': './traces',
        'log_dir': './logs',
        'model_dir': './models'
    }

    return config


def draw_state(record_action, record_state, path):
    length = len(record_action)
    plt.subplot(411)
    plt.plot(range(length), record_action)
    plt.xlabel('episode')
    plt.ylabel('action')
    ylabel = ['receiving rate', 'delay', 'packet loss']
    record_state = [t.numpy() for t in record_state]
    record_state = np.array(record_state)
    for i in range(3):
        plt.subplot(411+i+1)
        plt.plot(range(length), record_state[:,i])
        plt.xlabel('episode')
        plt.ylabel(ylabel[i])
    plt.tight_layout()
    plt.savefig("{}test_result.jpg".format(path))


def draw_module(model, data_path, max_num_steps = 1000):
    env = GymEnv()
    record_reward = []
    record_state = []
    record_action = []
    episode_reward  = 0
    time_step = 0
    tmp = model.random_action
    model.random_action = False
    while time_step < max_num_steps:
        done = False
        state = torch.Tensor(env.reset())
        while not done:
            action, _, _ = model.forward(state)
            state, reward, done, _ = env.step(action)
            state = torch.Tensor(state)
            record_state.append(state)
            record_reward.append(reward)
            record_action.append(action)
            time_step += 1
    model.random_action = True
    draw_state(record_action, record_state, data_path)