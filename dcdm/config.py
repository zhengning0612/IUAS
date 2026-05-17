class Config:
    T = 30
    node_types = 2
    edge_types = 2
    hidden_dim = 128
    num_heads = 4
    num_layers = 3
    Z = 256
    gamma = 0.99
    lr = 3e-4
    batch_size = 32
    alpha1 = 1.0
    alpha2 = 0.1
    alpha3 = 0.1

    num_devices = 10
    user_pos_range = [0, 10]
    device_pos_range = [0, 10]


config = Config()
