import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .config import config


class ISACGraphData:
    def __init__(self, num_devices, user_pos, device_positions, wavelength=0.1):
        self.num_nodes = num_devices
        self.user_pos = torch.tensor(user_pos, dtype=torch.float32)
        self.device_pos = torch.tensor(device_positions, dtype=torch.float32)
        self.wavelength = wavelength

        self.node_type = torch.randint(0, config.node_types, (self.num_nodes,))
        if torch.all(self.node_type == 0):
            self.node_type[random.randint(0, self.num_nodes - 1)] = 1
        if torch.all(self.node_type == 1):
            self.node_type[random.randint(0, self.num_nodes - 1)] = 0

        self.edge_index = self._generate_edges()
        self.edge_type = self._generate_edges_with_fresnel_constraint()

    def _generate_edges(self):
        txs = torch.where(self.node_type == 0)[0]
        rxs = torch.where(self.node_type == 1)[0]
        edges = []
        for tx in txs:
            for rx in rxs:
                edges.append([tx.item(), rx.item()])
        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).T

    def _is_in_fresnel(self, tx_pos, rx_pos, user_pos):
        d_tx_user = torch.norm(user_pos - tx_pos)
        d_rx_user = torch.norm(user_pos - rx_pos)
        d_tx_rx = torch.norm(tx_pos - rx_pos)
        if d_tx_rx < 1e-6:
            return False
        fresnel_radius = torch.sqrt(self.wavelength * d_tx_user * d_rx_user / d_tx_rx)
        area = torch.abs(
            (rx_pos[0] - tx_pos[0]) * (tx_pos[1] - user_pos[1])
            - (tx_pos[0] - user_pos[0]) * (rx_pos[1] - tx_pos[1])
        )
        dist = area / d_tx_rx
        return dist < 0.6 * fresnel_radius

    def _generate_edges_with_fresnel_constraint(self):
        edge_count = self.edge_index.shape[1]
        if edge_count == 0:
            return torch.empty((0,), dtype=torch.long)

        edge_type = torch.randint(0, config.edge_types, (edge_count,))
        valid_edges = []
        for i in range(edge_count):
            tx = self.edge_index[0, i]
            rx = self.edge_index[1, i]
            if self._is_in_fresnel(self.device_pos[tx], self.device_pos[rx], self.user_pos):
                valid_edges.append(i)

        if len(valid_edges) == 0:
            new_positions = np.random.uniform(*config.device_pos_range, size=(self.num_nodes, 2))
            new_graph = ISACGraphData(self.num_nodes, self.user_pos.numpy(), new_positions)
            self.edge_index = new_graph.edge_index
            return new_graph.edge_type

        if edge_type[valid_edges].sum() == 0:
            edge_type[random.choice(valid_edges)] = 1
        return edge_type

    def to_data(self):
        return Data(
            x=F.one_hot(self.node_type, config.node_types).float(),
            edge_index=self.edge_index,
            edge_attr=F.one_hot(self.edge_type, config.edge_types).float(),
            user_pos=self.user_pos.unsqueeze(0),
            device_pos=self.device_pos,
        )
