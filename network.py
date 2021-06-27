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

        self.numFcInput = 3072

        self.discount = config['discount_factor']

        self.rConv1d = nn.Conv1d(1, self.layer1_shape, 3)
        self.dConv1d = nn.Conv1d(1, self.layer1_shape, 3)
        self.lConv1d = nn.Conv1d(1, self.layer1_shape, 3)
        self.fc = nn.Linear(self.numFcInput, self.layer2_shape)
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
        receivingConv = F.relu(self.rConv1d(inputs[:, 0:1, :]), inplace=True)
        delayConv = F.relu(self.dConv1d(inputs[:, 1:2, :]), inplace=True)
        lossConv = F.relu(self.lConv1d(inputs[:, 2:3, :]), inplace=True)
        receiving_flatten = receivingConv.view(receivingConv.shape[0], -1)
        delay_flatten = delayConv.view(delayConv.shape[0], -1)
        loss_flatten = lossConv.view(lossConv.shape[0], -1)
        merge = torch.cat([receiving_flatten, delay_flatten, loss_flatten], 1)
        fcOut = F.relu(self.fc(merge), inplace=True)
        output = F.softmax(self.actor_output(fcOut), dim=-1)
        # dist = MultivariateNormal(action, cov_mat)
        # dist_entropy = dist.entropy()
        # action_logprobs = dist.log_prob(action)
        return output

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

        self.numFcInput = 3072

        self.discount = config['discount_factor']
    # def create_network(self):
    #     # Structure of Network
    #     # self.net = tf.keras.models.Sequential()
    #     # self.net.add(tf.keras.Input(shape=(self.s_dim, self.s_info)))
    #     # self.net.add(tf.keras.layers.Dense(self.layer1_shape, activation='relu'))
    #     # self.net.add(tf.keras.layers.Dense(self.layer2_shape, activation='relu'))
    #     # self.net.add(tf.keras.layers.Dense(self.a_dim))
    #     # inputs = tf.random.normal((self.s_dim, self.s_info))
    #     #
    #     #
    #     # h1 = tf.keras.layers.Dense(units=self.layer1_shape, name='fc1')
    #     # h1 = h1(inputs)
    #     # h1 = tf.keras.layers.BatchNormalization(h1, training=is_training, scale=False)
    #     # h1 = tf.nn.relu(h1)fs
    #     #
    #     # h2 = tf.keras.layers.Dense(units=self.layer2_shape, name='fc2')
    #     # h2 = h2(h1)
    #     # h2 = tf.keras.layers.BatchNormalization(h2, training=is_training, scale=False)
    #     # h2 = tf.nn.relu(h2)
    #
    #     # output = tf.keras.layers.Dense(units=self.a_dim, activation=tf.nn.tanh)
    #     # inputs = tflearn.input_data(shape=[self.s_dim, self.s_info])
    #     #
    #     # # Network Structure (Fully Connected)
    #     # h1 = tflearn.fully_connected(inputs, self.layer1_shape, activation='leaky_relu')
    #     # h2 = tflearn.fully_connected(h1, self.layer2_shape, activation='leaky_relu')
    #     # output = tflearn.fully_connected(h2, self.a_dim, activation='linear')
    #
    #     # (Pytorch) Network Structure
    #     self.net = nn.Sequential(
    #         nn.Linear(self.s_dim * self.s_info, self.layer1_shape),
    #         nn.LeakyReLU(),
    #         nn.Linear(self.layer1_shape, self.layer2_shape),
    #         nn.LeakyReLU(),
    #         nn.Linear(self.layer2_shape, self.a_dim),
    #         nn.Tanh()
    #     )
    #
    #     # Network Structure (Convolutional)
    #     # h1 = tflearn.conv_1d(inputs, self.layer1_shape, 4, activation='relu')
    #     # h2 = tflearn.fully_connected(h1, self.layer2_shape, activation='relu')
    #     # output = tflearn.fully_connected(h2, self.a_dim, activation='linear')
    #
    #     # inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])
    #     # split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], 128, activation='relu')
    #     # split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], 128, activation='relu')
    #     # split_2 = tflearn.conv_1d(inputs[:, 2:3, :], 128, 4, activation='relu')
    #     # split_3 = tflearn.conv_1d(inputs[:, 3:4, :], 128, 4, activation='relu')
    #     # split_4 = tflearn.conv_1d(inputs[:, 4:5, :A_DIM], 128, 4, activation='relu')
    #     # split_5 = tflearn.fully_connected(inputs[:, 4:5, -1], 128, activation='relu')
    #     #
    #     # split_2_flat = tflearn.flatten(split_2)
    #     # split_3_flat = tflearn.flatten(split_3)
    #     # split_4_flat = tflearn.flatten(split_4)
    #     #
    #     # merge_net = tflearn.merge([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')
    #     #
    #     # dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
    #     # out = tflearn.fully_connected(dense_net_0, self.a_dim, activation='softmax')
    #     # return inputs, output

        self.rConv1d = nn.Conv1d(1, self.layer1_shape, 3)
        self.dConv1d = nn.Conv1d(1, self.layer1_shape, 3)
        self.lConv1d = nn.Conv1d(1, self.layer1_shape, 3)
        self.fc = nn.Linear(self.numFcInput, self.layer2_shape)
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
        # receivingConv = F.leaky_relu(self.rConv1d(inputs[:, 0:1, -1]), inplace=True)
        # delayConv = F.leaky_relu(self.dConv1d(inputs[:, 1:2, -1]), inplace=True)
        # lossConv = F.leaky_relu(self.lConv1d(inputs[:, 2:3, -1]), inplace=True)
        # merge = torch.cat([receivingConv, delayConv, lossConv], 1)
        # fcOut = F.leaky_relu(self.fc(merge), inplace=True)
        # value = self.critic_output(fcOut)
        receivingConv = F.relu(self.rConv1d(inputs[:, 0:1, :]), inplace=True)
        delayConv = F.relu(self.dConv1d(inputs[:, 1:2, :]), inplace=True)
        lossConv = F.relu(self.lConv1d(inputs[:, 2:3, :]), inplace=True)
        receiving_flatten = receivingConv.view(receivingConv.shape[0], -1)
        delay_flatten = delayConv.view(delayConv.shape[0], -1)
        loss_flatten = lossConv.view(lossConv.shape[0], -1)
        merge = torch.cat([receiving_flatten, delay_flatten, loss_flatten], 1)
        fcOut = F.relu(self.fc(merge), inplace=True)
        value = self.critic_output(fcOut)
        return value
