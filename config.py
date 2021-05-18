import json

def load_config():
    config = {
        #todo: add parameters regarding configuration
        'learning_rate' : 0.001,
        'num_agents': 16,
        'save_interval': 100,
        'default_bwe': 1000,
        'train_seq_length': 1000,
        'state_dim': 3,
        'state_length': 10,
        'action_dim': 1,
        'device': 'cpu',
        'load_model': False,
        'saved_model_path': '',
        'layer1_shape': 256,
        'layer2_shape': 256
    }

    return config