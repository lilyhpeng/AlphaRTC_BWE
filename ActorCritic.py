import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from network import ActorNetwork, CriticNetwork

class ActorCritic():
    def __init__(self, is_central, config):
        self.config = config
        self.ActorNetwork = self.getActorNetwork().to(self.config["device"])
        self.CriticNetwork = self.getCriticNetwork().to(self.config["device"])
        if is_central:
            self.actorOptimizer = optim.RMSprop(self.ActorNetwork.parameters(),lr=self.config['actor_learning_rate'], alpha=0.9, eps=1e-10)
            self.actorOptimizer.zero_grad()
            self.criticOptimizer = optim.RMSprop(self.CriticNetwork.parameters(),lr=self.config['critic_learning_rate'], alpha=0.9, eps=1e-10)
            self.criticOptimizer.zero_grad()
        self.loss_function = nn.MSELoss()
        pass

    def getActorNetwork(self):
        # self.ActorNetwork = ActorNetwork(self.config)
        return ActorNetwork(self.config)

    def getCriticNetwork(self):
        # self.CriticNetwork = CriticNetwork(self.config)
        return CriticNetwork(self.config)

    def getNetworkGradient(self, s_batch, a_batch, r_batch, done):
        s_batch = torch.cat(s_batch).to(self.config['device'])
        a_batch = torch.LongTensor(a_batch).to(self.config['device'])
        r_batch = torch.tensor(r_batch).to(self.config['device'])
        R_batch = torch.zeros(r_batch.shape).to(self.config['device'])

        R_batch[-1] = r_batch[-1]
        for t in reversed(range(r_batch.shape[0] - 1)):
            R_batch[t] = r_batch[t] + self.config['discount_factor'] * R_batch[t + 1]

        with torch.no_grad():
            v_batch = self.CriticNetwork.forward(s_batch).sqeeze().to(self.config['device'])
        td_batch = R_batch - v_batch

        prob = self.ActorNetwork.forward(s_batch)
        m_probs = Categorical(prob)
        log_probs = m_probs.log_prob(a_batch)
        actor_loss = torch.sum(log_probs * (-td_batch))
        entropy_loss = -self.config['entropy_weight'] * torch.sum(m_probs.entropy())
        actor_loss += entropy_loss
        actor_loss.backward()

        value = self.CriticNetwork.forward(s_batch).squeeze()
        critic_loss = self.loss_function(R_batch, value)
        critic_loss.backward()

    def predict(self, inputs):
        with torch.no_grad():
            prob = self.ActorNetwork.forward(inputs)
            m = Categorical(prob)
            action = m.sample().item()
            return action

    def updateNetwork(self):
        self.actorOptimizer.step()
        self.actorOptimizer.zero_grad()

        self.criticOptimizer.step()
        self.criticOptimizer.zero_grad()

