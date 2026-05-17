import numpy as np
import torch
from tqdm import tqdm
import random

DCDM_NODES = 9

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, num_nodes=9):
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.data = self.generate_graph_dataset()

    def generate_graph_dataset(self):
        """生成图数据集：(设备分布, 用户位置) → (最优图邻接矩阵, 效用值)"""
        dataset = []
        for _ in tqdm(range(self.num_samples), desc="生成图数据集"):
            # 1. 随机生成设备位置（室内场景10m×10m）
            node_pos = np.random.uniform(0, 10, (DCDM_NODES, 2))
            # 2. 随机生成用户位置
            user_pos = np.random.uniform(0, 10, 2)
            # 3. 生成最优图（模拟论文D-CDM目标：高SSNR+低成本）
            adj = np.zeros((DCDM_NODES, DCDM_NODES))  # 邻接矩阵（Tx→Rx为1）

            # 随机生成TX、RX个数
            tx_idx = np.random.choice(DCDM_NODES, random.randint(1, DCDM_NODES - 1), replace=False)
            tx_cnt = len(tx_idx)
            rx_idx = np.random.choice([i for i in range(DCDM_NODES) if i not in tx_idx],
                                      random.randint(1, DCDM_NODES-tx_cnt), replace=False)
            # 节点类型（0=未激活，1=Tx，2=Rx）
            node_types = np.zeros(self.num_nodes, dtype=int)
            node_types[tx_idx] = 1
            node_types[rx_idx] = 2

            # 激活Tx→Rx链路
            for tx in tx_idx:
                for rx in rx_idx:
                    adj[tx, rx] = 1

            # 4. 封装样本（设备分布=节点位置，用户位置，图=邻接矩阵+Tx/Rx掩码，效用值）
            sample = {
                "node_types": torch.tensor(node_types),
                "edge_types": torch.tensor(adj),
                "tx_idx": torch.tensor(tx_idx),
                "rx_idx": torch.tensor(rx_idx),
                "tx_pos": torch.tensor(node_pos[tx_idx], dtype=torch.float32),
                "rx_pos": torch.tensor(node_pos[rx_idx], dtype=torch.float32),
                "user_pos": torch.tensor(user_pos, dtype=torch.float32),
            }
            dataset.append(sample)
        return dataset

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]