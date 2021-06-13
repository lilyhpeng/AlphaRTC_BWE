import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from network_ppo import Network

class ActorCritic():
    def __init__(self, is_central=False, config=None):
        self.Network = self.getNetwork(config)
        self.device = config['device']
        if is_central:
            self.optimizer = optim.Adam(self.Network.parameters(), lr=config['learning_rate'], betas=(0.9, 0.999))
            self.optimizer.zero_grad()

        self.oldNetwork = self.getNetwork(config)
        self.oldNetwork.load_state_dict(self.Network.state_dict())

        self.ppo_clip = config['ppo_clip']
        self.ppo_epoch = config['ppo_epoch']

    def getNetwork(self, config):
        # self.ActorNetwork = ActorNetwork(self.config)
        return Network(config)

    def predict(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action, action_logprobs, value, action_mean = self.oldNetwork.forward(state)

        return action, action_logprobs, value

    def getValue(self, state):
        state = torch.FloatTensor(state).to(self.device)
        _, _, value, _ = self.oldNetwork.forward(state)
        return value

    def updateNetwork(self, s_batch, a_batch, a_log_batch, v_batch, return_batch):
        episode_policy_loss = 0
        episode_value_loss = 0

        advantages = (torch.tensor(return_batch).to(self.device) - torch.tensor(v_batch).to(self.device)).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        old_states = torch.squeeze(torch.stack(s_batch).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(a_batch).to(self.device), 1).detach()
        old_action_logprobs = torch.squeeze(torch.stack(a_log_batch), 1).to(self.device).detach()
        old_returns = torch.squeeze(torch.stack(return_batch), 1).to(self.device).detach()

        for t in range(self.ppo_epoch):
            logprobs, state_values, dist_entropy = self.Network.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_action_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (state_values - old_returns).pow(2).mean()
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            episode_policy_loss += policy_loss.detach()
            episode_value_loss += value_loss.detach()

        self.oldNetwork.load_state_dict(self.Network.state_dict())
        return episode_policy_loss / self.ppo_epoch, episode_value_loss / self.ppo_epoch

