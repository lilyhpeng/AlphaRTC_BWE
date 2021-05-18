#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import random
import numpy as np
import glob
from Queue import Queue
import gym
from gym import spaces

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
    def __init__(self, step_time=60):
        self.gym_env = None     
        self.step_time = step_time
        trace_dir = os.path.join(os.path.dirname(__file__), "traces")
        self.trace_set = glob.glob('{trace_dir}/**/*.json', recursive=True)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=np.zeros((STATE_DIMENSION,HISTORY_LENGTH)),
            high=np.ones((STATE_DIMENSION,HISTORY_LENGTH)),
            dtype=np.float64)


    def reset(self):#reset history of state and return np.array
        self.gym_env = alphartc_gym.Gym()
        self.gym_env.reset(trace_path=random.choice(self.trace_set),
            report_interval_ms=self.step_time,
            duration_time_ms=0)
        self.packet_record = PacketRecord()
        self.packet_record.reset()
        self.receiving_rate=np.zeros(HISTORY_LENGTH)
        self.delay=np.zeros(HISTORY_LENGTH)
        self.loss_ratio=np.zeros(HISTORY_LENGTH)
        self.prediction_history=np.zeros(HISTORY_LENGTH)

        states=np.vstack((self.receiving_rate,self.delay,self.loss_ratio,self.prediction_history))

        return states

    def get_reward(self):
        reward = self.receiving_rate[HISTORY_LENGTH-1] - self.delay[HISTORY_LENGTH-1] - self.loss_ratio[HISTORY_LENGTH-1]

        return reward

    def step(self, action):
        # action: log to linear
        bandwidth_prediction = log_to_linear(action)

        # run the action
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

        # calculate state

        receiving_rate = self.packet_record.calculate_receiving_rate(interval=self.step_time)
        #states.append(liner_to_log(receiving_rate))
        self.receiving_rate.append(receiving_rate)
        np.delete(self.receiving_rate, 0, axis=0)
        delay = self.packet_record.calculate_average_delay(interval=self.step_time)
        #states.append(min(delay/1000, 1))
        self.delay.append(delay )
        np.delete(self.delay, 0, axis=0)
        loss_ratio = self.packet_record.calculate_loss_ratio(interval=self.step_time)
        #states.append(loss_ratio)
        self.loss_ratio.append(loss_ratio)
        np.delete(self.loss_ratio, 0, axis=0)
        latest_prediction = self.packet_record.calculate_latest_prediction()
        #states.append(liner_to_log(latest_prediction))
        self.prediction_history.append(latest_prediction)
        np.delete(self.prediction_history, 0, axis=0)
        states = np.vstack((self.receiving_rate, self.delay, self.loss_ratio, self.prediction_history))
        # calculate reward

        reward = self.get_reward()

        return states, reward, done, {}

