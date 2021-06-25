import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import random
from rtc_env_ppo import GymEnv
from torch.autograd import Variable
from deep_rl.storage import Storage
from deep_rl.actor_critic import ActorCritic


def load_config():
    config = {
        #todo: add parameters regarding configuration
        # 'actor_learning_rate': 1e-5,
        # 'critic_learning_rate': 1e-5,
        'learning_rate': 3e-5,
        'num_agents': 4,
        'save_interval': 20,

        'default_bwe': 2,
        'train_seq_length': 1000,
        'state_dim': 4,
        'state_length': 10,
        'action_dim': 1,
        'device': 'cpu',
        'discount_factor': 0.99,
        'exploration_param': 0.05,
        'random_action': True,
        'load_model': False,
        'saved_actor_model_path': '',
        'saved_critic_model_path': '',
        'layer1_shape': 128,
        'layer2_shape': 128,

        'entropy_weight': 0.0,
        'ppo_clip': 0.2,
        'ppo_epoch': 37,
        'update_interval': 4000,

        'trace_dir': './traces',
        'log_dir': './logs',
        'model_dir': '/media/ExtHDD02/wb/bwe_models'
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


def draw_module(config, model, data_path, max_num_steps = 1000):
    env = GymEnv(config=config, env_id=1)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    record_reward = []
    record_state = []
    record_action = []
    episode_reward  = 0
    time_step = 0
    model.random_action = False
    while time_step < max_num_steps:
        use_action_counter = 0
        done = False
        state = torch.Tensor(env.reset()).to(device)
        while not done:
            if use_action_counter == 5:
                use_action = True
                use_action_counter = 0
            else:
                use_action = False
                use_action_counter += 1

            action, _, _ = model.predict(state)
            state, reward, done = env.step(action, use_action)
            state = torch.Tensor(state).to(device)
            record_state.append(state)
            record_reward.append(reward)
            record_action.append(action)
            time_step += 1
    model.random_action = True
    draw_state(record_action, record_state, data_path)

class TrafficLight:
    # used by chief to allow workers to run or not

    def __init__(self, val=True):
        self.val = mp.Value("b", False)
        self.lock = mp.Lock()

    def get(self):
        with self.lock:
            return self.val.value

    def switch(self):
        self.val.value = (not self.val.value)

class Counter:
    # enable the chief to access worker's total number of updates
    def __init__(self, val=True):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()

    def get(self):
        # used by chief
        with self.lock:
            return self.val.value

    def increment(self):
        # used by workers
        with self.lock:
            self.val.value += 1

    def reset(self):
        # used by chief
        with self.lock:
            self.val.value = 0


class shared_batch_buffer():
    def __init__(self):
        self.buffer = {}

    def push_batch(self, rank, s_batch, a_batch, r_batch, is_terminal, next_value):
        self.buffer[str(rank) + 'states'] = s_batch
        self.buffer[str(rank) + 'actions'] = a_batch
        self.buffer[str(rank) + 'rewards'] = r_batch
        self.buffer[str(rank) + 'is_terminal'] = is_terminal
        self.buffer[str(rank) + 'next_value'] = next_value

    def clear_buffer(self):
        self.buffer = {}

class shared_grad_buffer():
    def __init__(self, model):
        self.grads = {}
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] = torch.ones(p.size()).share_memory_()

    def add_grad(self, model):
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] = p.grad.data

    def reset(self):
        for name, grad in self.grads.items():
            self.grads[name].fill_(0)

class shared_obs_stats():
    def __init__(self, num_inputs):
        self.n = torch.zeros(num_inputs).share_memory_()
        self.mean = torch.zeros(num_inputs).share_memory_()
        self.mean_diff = torch.zeros(num_inputs).share_memory_()
        self.var = torch.zeros(num_inputs).share_memory_()

    def observes(self, obs):
        x = obs.data.squeeze()
        self.n += 1
        last_mean = self.mean.clone()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_mean = Variable(self.mean.unsqueeze(0).expand_as(inputs))
        obs_std = Variable(torch.sqrt(self.var).unsqueeze(0).expand_as(inputs))
        return torch.clamp((inputs - obs_mean)/obs_std, -5., 5.)


# class ReplayMemory(object):
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#
#     def push(self, events):
#         for event in zip(*events):
#             self.memory.append(event)
#             if len(self.memory) > self.capacity:
#                 del self.memory[0]
#
#     def clear(self):
#         self.memory = []
#
#     def sample(self, batch_size):
#         samples = zip(*random.sample(self.memory, batch_size))
#         return map(lambda x: torch.cat(x, 0), samples)