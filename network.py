import tflearn


class Network:
    def __init__(self):
        pass

    def create_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

        split_0 = tflearn.fully_connected(inputs[:, 0:1, -1], 128, activation='relu')
        split_1 = tflearn.fully_connected(inputs[:, 1:2, -1], 128, activation='relu')
        split_2 = tflearn.conv_1d(inputs[:, 2:3, :], 128, 4, activation='relu')
        split_3 = tflearn.conv_1d(inputs[:, 3:4, :], 128, 4, activation='relu')
        split_4 = tflearn.conv_1d(inputs[:, 4:5, :A_DIM], 128, 4, activation='relu')
        split_5 = tflearn.fully_connected(inputs[:, 4:5, -1], 128, activation='relu')

        split_2_flat = tflearn.flatten(split_2)
        split_3_flat = tflearn.flatten(split_3)
        split_4_flat = tflearn.flatten(split_4)

        merge_net = tflearn.merge([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')

        dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
        out = tflearn.fully_connected(dense_net_0, self.a_dim, activation='softmax')

        return inputs, out

    def create_lstm_network(self):
        pass