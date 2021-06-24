# from gym.alphartc_gym.gym import Gym
import sys
import os
import torch
import glob
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym"))
import alphartc_gym
# from gym.alphartc_gym.utils.packet_info import PacketInfo
# from gym.alphartc_gym.utils.packet_record import PacketRecord
from BandwidthEstimator_gcc import Estimator
import matplotlib.pyplot as plt
from utils import draw_module, load_config
from network_ppo import Network

def rl_test():
    config = load_config()
    model = Network(config)
    model.load_state_dict(torch.load(config['model_path']))
    draw_module(config, model, config['data_path'])


if __name__ == '__main__':
    rl_test()