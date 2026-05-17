from tmp6 import *
import torch
import numpy as np
from tqdm import tqdm
import random


def generate_valid_G0(num_samples=5000, filename="G0_dataset.pt"):
    """
    生成严格符合 ISAC 物理约束的 G0 数据集。
    约束条件：
    1. 图中至少包含一个 Tx 和一个 Rx。
    2. [关键] 必须存在至少一条被激活的链路 (edge_type=1)。
    3. [关键] 该激活链路必须满足：用户位于其第一菲涅尔区内 (User in Fresnel Zone)。
    """
    dataset = []

    pbar = tqdm(total=num_samples, desc=f"Generating {filename}")

    while len(dataset) < num_samples:
        # 随机生成位置
        user_pos = np.random.uniform(*config.user_pos_range, size=2)
        device_positions = np.random.uniform(*config.device_pos_range, size=(config.num_devices, 2))

        # 构造图 (启用 strict_fresnel=True)
        # 自动计算菲涅尔区，并尝试激活一条合法的边
        g0 = ISACGraphData(config.num_devices, user_pos, device_positions, strict_fresnel=True)

        # 检查基本拓扑约束
        if (g0.node_type == 0).sum() == 0: continue  # 没有发射机
        if (g0.node_type == 1).sum() == 0: continue  # 没有接收机

        # 检查菲涅尔区约束 (User Constraints)
        # g0.edge_type 中如果全为 0，说明 _generate_edges_with_fresnel_constraint 未找到任何有效边
        # 这意味着用户不在任何 Tx-Rx 对的菲涅尔区内 -> 样本无效，丢弃
        if g0.edge_type.sum() == 0:
            continue

        # 双重验证 (Sanity Check)
        # 确保被激活的那条边，确实覆盖了用户
        active_edge_idx = torch.argmax(g0.edge_type).item()
        tx_idx = g0.edge_index[0, active_edge_idx]
        rx_idx = g0.edge_index[1, active_edge_idx]

        is_valid = g0._is_in_fresnel(g0.device_pos[tx_idx], g0.device_pos[rx_idx], g0.user_pos)

        if is_valid:
            dataset.append(g0.to_data())
            pbar.update(1)
        else: continue #(理论上不会进入这里，因为 g0 初始化逻辑已保证)

    pbar.close()
    print(f"Successfully generated {len(dataset)} valid samples.")
    print(f"Condition 'User in Fresnel Zone' enforced for all samples.")
    torch.save(dataset, filename)
    print(f"Saved to {filename}")

    return dataset


if __name__ == "__main__":
    print("=== Generating Training Set ===")
    generate_valid_G0(num_samples=10000, filename="./G0_dataset.pt")

    print("\n=== Generating Test Set ===")
    generate_valid_G0(num_samples=5000, filename="./G0_dataset_test.pt")