# import tflearn
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.distributions import MultivariateNormal

# todo: network structure needs to be reconsidered

class ActorNetwork(nn.Module):
    def __init__(self, config):
        super(ActorNetwork, self).__init__()
        self.device = config['device']
        self.lr_rate = config['actor_learning_rate']
        self.s_dim = config['state_dim']
        self.s_info = config['state_length']
        self.a_dim = config['action_dim']

        self.layer1_shape = config['layer1_shape']
        self.layer2_shape = config['layer2_shape']

        self.discount = config['discount_factor']

        self.rConv1d = nn.Conv1d(1, self.layer1_shape, (3, 3))
        self.dConv1d = nn.Conv1d(1, self.layer1_shape, (3, 3))
        self.lConv1d = nn.Conv1d(1, self.layer1_shape, (3, 3))
        self.fc = nn.Linear(self.layer1_shape * self.s_dim, self.layer2_shape)
        # self.h1 = nn.Linear(self.s_dim * self.s_info, self.layer1_shape)
        # self.h2 = nn.Linear(self.layer1_shape, self.layer2_shape)
        self.actor_output = nn.Linear(self.layer2_shape, self.a_dim)
        # self.action_var = torch.full((self.config["action_dim"],), self.config['exploration_param']).to(self.config["device"])

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data, 0.1)
            elif isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data, 0.1)

    def forward(self, inputs):
        # cov_mat = torch.diag(self.action_var).to(self.config['device'])
        # x = F.leaky_relu(self.h1(inputs))
        # x = F.leaky_relu(self.h2(x))
        receivingConv = F.leaky_relu(self.rConv1d(inputs[:, 0:1, -1]), inplace=True)
        delayConv = F.leaky_relu(self.dConv1d(inputs[:, 1:2, -1]), inplace=True)
        lossConv = F.leaky_relu(self.lConv1d(inputs[:, 2:3, -1]), inplace=True)
        merge = torch.cat([receivingConv, delayConv, lossConv], 1)
        fcOut = F.leaky_relu(self.fc(merge), inplace=True)
        action = F.softmax(self.actor_output(fcOut))
        # dist = MultivariateNormal(action, cov_mat)
        # dist_entropy = dist.entropy()
        # action_logprobs = dist.log_prob(action)
        return action

class CriticNetwork(nn.Module):
    def __init__(self, config):
        super(CriticNetwork, self).__init__()
        self.device = config['device']
        self.lr_rate = config['critic_learning_rate']
        self.s_dim = config['state_dim']
        self.s_info = config['state_length']
        self.a_dim = config['action_dim']

        self.layer1_shape = config['layer1_shape']
        self.layer2_shape = config['layer2_shape']

        self.discount = config['discount_factor']

        self.rConv1d = nn.Conv1d(1, self.layer1_shape, (3, 3))
        self.dConv1d = nn.Conv1d(1, self.layer1_shape, (3, 3))
        self.lConv1d = nn.Conv1d(1, self.layer1_shape, (3, 3))
        self.fc = nn.Linear(self.layer1_shape * self.s_dim, self.layer2_shape)
        # self.h1 = nn.Linear(self.s_dim * self.s_info, self.layer1_shape)
        # self.h1 = nn.Conv2d([self.s_dim, self.s_info], self.layer1_shape, [3, 3])
        # self.a1 = nn.LeakyReLU()
        # self.h2 = nn.Linear(self.layer1_shape, self.layer2_shape)
        self.critic_output = nn.Linear(self.layer2_shape, 1)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data, 0.1)
            elif isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data, 0.1)

    def forward(self, inputs):
        receivingConv = F.leaky_relu(self.rConv1d(inputs[:, 0:1, -1]), inplace=True)
        delayConv = F.leaky_relu(self.dConv1d(inputs[:, 1:2, -1]), inplace=True)
        lossConv = F.leaky_relu(self.lConv1d(inputs[:, 2:3, -1]), inplace=True)
        merge = torch.cat([receivingConv, delayConv, lossConv], 1)
        fcOut = F.leaky_relu(self.fc(merge), inplace=True)
        value = self.critic_output(fcOut)
        return value