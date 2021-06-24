#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import time
from torch import nn
from .actor_critic import ActorCritic


class PPO:
    def __init__(self, state_dim, state_length, action_dim, exploration_param, lr, betas, gamma, ppo_epoch, ppo_clip, use_gae=False):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.ppo_clip = ppo_clip
        self.ppo_epoch = ppo_epoch
        self.use_gae = use_gae

        self.device = torch.device("cpu")
        self.policy = ActorCritic(state_dim, state_length, action_dim, exploration_param, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, state_length, action_dim, exploration_param, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, state, storage):
        state = torch.FloatTensor(state).to(self.device)
        action, action_logprobs, value, action_mean = self.policy_old.forward(state)

        storage.logprobs.append(action_logprobs)
        storage.values.append(value)
        storage.states.append(state)
        storage.actions.append(action)
        return action

    def get_value(self, state):
        action, action_logprobs, value, action_mean = self.policy_old.forward(state)
        return value

    def update(self, states, actions, logprobs, values, returns):
        episode_policy_loss = 0
        episode_value_loss = 0
        if self.use_gae:
            raise NotImplementedError
        advantages = (torch.tensor(returns) - torch.tensor(values)).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(actions).to(self.device), 1).detach()
        old_action_logprobs = torch.squeeze(torch.stack(logprobs), 1).to(self.device).detach()
        old_returns = torch.squeeze(torch.stack(returns), 1).to(self.device).detach()

        for t in range(self.ppo_epoch):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_action_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.ppo_clip, 1+self.ppo_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (state_values - old_returns).pow(2).mean()
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            episode_policy_loss += policy_loss.detach()
            episode_value_loss += value_loss.detach()

        self.policy_old.load_state_dict(self.policy.state_dict())

        return episode_policy_loss / self.ppo_epoch, episode_value_loss / self.ppo_epoch

    def save_model(self, data_path):
        torch.save(self.policy.state_dict(), '{}ppo_{}.pth'.format(data_path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))
