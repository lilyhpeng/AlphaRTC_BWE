# import tflearn
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.nn import init
# from torch.distributions import MultivariateNormal

# todo: network structure needs to be reconsidered

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.device = config['device']
        self.lr_rate = config['learning_rate']
        self.s_dim = config['state_dim']
        self.s_info = config['state_length']
        self.a_dim = config['action_dim']
        self.exploration_param = config['exploration_param']
        self.random_action = config['random_action']

        self.layer1_shape = config['layer1_shape']
        self.layer2_shape = config['layer2_shape']

        self.numFcInput = 3072

        self.discount = config['discount_factor']

        self.rConv1d = nn.Conv1d(1, self.layer1_shape, 3)
        self.dConv1d = nn.Conv1d(1, self.layer1_shape, 3)
        self.lConv1d = nn.Conv1d(1, self.layer1_shape, 3)
        self.fc = nn.Linear(self.numFcInput, self.layer2_shape)
        # self.h1 = nn.Linear(self.s_dim * self.s_info, self.layer1_shape)
        # self.h2 = nn.Linear(self.layer1_shape, self.layer2_shape)
        self.actor_output = nn.Linear(self.layer2_shape, self.a_dim)
        self.critic_output = nn.Linear(self.layer2_shape, 1)
        self.action_var = torch.full((self.a_dim,), self.exploration_param**2).to(self.device)
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
        receivingConv = F.relu(self.rConv1d(inputs[:, 0:1, :]), inplace=True)
        delayConv = F.relu(self.dConv1d(inputs[:, 1:2, :]), inplace=True)
        lossConv = F.relu(self.lConv1d(inputs[:, 2:3, :]), inplace=True)
        receiving_flatten = receivingConv.view(receivingConv.shape[0], -1)
        delay_flatten = delayConv.view(delayConv.shape[0], -1)
        loss_flatten = lossConv.view(lossConv.shape[0], -1)
        merge = torch.cat([receiving_flatten, delay_flatten, loss_flatten], 1)
        fcOut = F.relu(self.fc(merge), inplace=True)
        action_mean = torch.sigmoid(self.actor_output(fcOut))
        cov_mat = torch.diag(self.action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        if not self.random_action:
            action = action_mean
        else:
            action = dist.sample()
        action_logprobs = dist.log_prob(action)
        # critic
        receivingConv_critic = F.relu(self.rConv1d(inputs[:, 0:1, :]), inplace=True)
        delayConv_critic = F.relu(self.dConv1d(inputs[:, 1:2, :]), inplace=True)
        lossConv_critic = F.relu(self.lConv1d(inputs[:, 2:3, :]), inplace=True)
        receiving_flatten_critic = receivingConv.view(receivingConv_critic.shape[0], -1)
        delay_flatten_critic = delayConv.view(delayConv_critic.shape[0], -1)
        loss_flatten_critic = lossConv.view(lossConv_critic.shape[0], -1)
        merge_critic = torch.cat([receiving_flatten_critic, delay_flatten_critic, loss_flatten_critic], 1)
        fcOut_critic = F.relu(self.fc(merge_critic), inplace=True)
        value = self.critic_output(fcOut_critic)

        return action.detach(), action_logprobs, value, action_mean

    def evaluate(self, state, action):
        _, _, value, action_mean = self.forward(state)
        cov_mat = torch.diag(self.action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(value), dist_entropy
