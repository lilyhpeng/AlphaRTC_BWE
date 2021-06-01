#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import random
import numpy as np
import glob
from queue import Queue
import gym
from gym import spaces
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym"))
import alphartc_gym
from alphartc_gym.utils.packet_info import PacketInfo
from alphartc_gym.utils.packet_record import PacketRecord


UNIT_M = 1000000
MAX_BANDWIDTH_MBPS = 8
MIN_BANDWIDTH_MBPS = 0.01
HISTORY_LENGTH = 10
STATE_DIMENSION = 4
LOG_MAX_BANDWIDTH_MBPS = np.log(MAX_BANDWIDTH_MBPS)
LOG_MIN_BANDWIDTH_MBPS = np.log(MIN_BANDWIDTH_MBPS)


def liner_to_log(value):
    # from 10kbps~8Mbps to 0~1
    value = np.clip(value / UNIT_M, MIN_BANDWIDTH_MBPS, MAX_BANDWIDTH_MBPS)
    log_value = np.log(value)
    return (log_value - LOG_MIN_BANDWIDTH_MBPS) / (LOG_MAX_BANDWIDTH_MBPS - LOG_MIN_BANDWIDTH_MBPS)


def log_to_linear(value):
    # from 0~1 to 10kbps to 8Mbps
    value = np.clip(value, 0, 1)
    log_bwe = value * (LOG_MAX_BANDWIDTH_MBPS - LOG_MIN_BANDWIDTH_MBPS) + LOG_MIN_BANDWIDTH_MBPS
    return np.exp(log_bwe) * UNIT_M


class GymEnv:
    def __init__(self, config, step_time=60, env_id=None):
        self.env_id = str(env_id)
        # self.env_id = env_id
        self.step_time = step_time
        self.gym_env = alphartc_gym.Gym(self.env_id)
        self.packet_record = PacketRecord()

        self.config = config

        # initialize state information:
        # self.receiving_rate = np.zeros(HISTORY_LENGTH)
        # self.delay = np.zeros(HISTORY_LENGTH)
        # self.loss_ratio = np.zeros(HISTORY_LENGTH)
        # self.prediction_history = np.zeros(HISTORY_LENGTH)

        # trace_dir = os.path.join(os.path.dirname(__file__), "traces")
        trace_dir = self.config['trace_dir']
        self.trace_set = glob.glob('{}/*/*.json'.format(trace_dir), recursive=True)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=np.zeros((STATE_DIMENSION, HISTORY_LENGTH)),
            high=np.ones((STATE_DIMENSION, HISTORY_LENGTH)),
            dtype=np.float64)

        self.state = torch.zeros((1, self.config['state_dim'], self.config['state_length']))
        self.receiving_rate_list = []
        self.delay_list = []
        self.loss_ratio_list = []

    def reset(self):
        # self.gym_env.reset(trace_path=random.choice(self.trace_set), report_interval_ms=self.step_time,
                           # duration_time_ms=0)
        self.gym_env.reset(trace_path='{}/trace_300k.json'.format(self.config['trace_dir']), report_interval_ms=self.step_time,
                           duration_time_ms=0)
        self.packet_record.reset()
        # self.receiving_rate = np.zeros(HISTORY_LENGTH)
        # self.delay = np.zeros(HISTORY_LENGTH)
        # self.loss_ratio = np.zeros(HISTORY_LENGTH)
        # self.prediction_history = np.zeros(HISTORY_LENGTH)

        # states = np.vstack((self.receiving_rate, self.delay, self.loss_ratio, self.prediction_history))
        self.state = torch.zeros((1, self.config['state_dim'], self.config['state_length']))
        # return states

    def get_reward(self):
        # reward = self.receiving_rate[HISTORY_LENGTH-1] - self.delay[HISTORY_LENGTH-1] - self.loss_ratio[HISTORY_LENGTH-1]
        reward = self.receiving_rate - self.delay - self.loss_ratio
        # reward = self.receiving_rate
        return reward

    def step(self, action):
        # action: log to linear
        # bandwidth_prediction = log_to_linear(action)
        bandwidth_prediction = action

        # run the action, get related packet list:
        packet_list, done = self.gym_env.step(bandwidth_prediction)
        for pkt in packet_list:
            packet_info = PacketInfo()
            packet_info.payload_type = pkt["payload_type"]
            packet_info.ssrc = pkt["ssrc"]
            packet_info.sequence_number = pkt["sequence_number"]
            packet_info.send_timestamp = pkt["send_time_ms"]
            packet_info.receive_timestamp = pkt["arrival_time_ms"]
            packet_info.padding_length = pkt["padding_length"]
            packet_info.header_length = pkt["header_length"]
            packet_info.payload_size = pkt["payload_size"]
            packet_info.bandwidth_prediction = bandwidth_prediction
            self.packet_record.on_receive(packet_info)

        # calculate state:
        self.receiving_rate = self.packet_record.calculate_receiving_rate(interval=self.step_time)
        self.receiving_rate_list.append(self.receiving_rate)
        # states.append(liner_to_log(receiving_rate))
        # self.receiving_rate.append(receiving_rate)
        # np.delete(self.receiving_rate, 0, axis=0)
        self.delay = self.packet_record.calculate_average_delay(interval=self.step_time)
        self.delay_list.append(self.delay)
        # states.append(min(delay/1000, 1))
        # self.delay.append(delay)
        # np.delete(self.delay, 0, axis=0)
        self.loss_ratio = self.packet_record.calculate_loss_ratio(interval=self.step_time)
        self.loss_ratio_list.append(self.loss_ratio)

        self.state = self.state.clone().detach()
        self.state = torch.roll(self.state, -1, dims=-1)
        # states.append(loss_ratio)
        # self.loss_ratio.append(loss_ratio)
        # np.delete(self.loss_ratio, 0, axis=0)
        # latest_prediction = self.packet_record.calculate_latest_prediction()
        # states.append(liner_to_log(latest_prediction))
        # self.prediction_history.append(latest_prediction)
        # np.delete(self.prediction_history, 0, axis=0)
        # states = np.vstack((self.receiving_rate, self.delay, self.loss_ratio, self.prediction_history))
        # todo: regularization needs to be fixed
        self.state[0, 0, -1] = self.receiving_rate / 300000.0
        self.state[0, 1, -1] = self.delay / 200.0
        self.state[0, 2, -1] = self.loss_ratio

        # maintain list length
        if len(self.receiving_rate_list) == self.config['state_length']:
            self.receiving_rate_list.pop(0)
            self.delay_list.pop(0)
            self.loss_ratio_list.pop(0)

        # calculate reward:
        reward = self.get_reward()

        return self.state, reward, done, {}

