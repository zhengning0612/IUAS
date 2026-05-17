import random

import numpy as np
import torch
import torch.nn.functional as F

from .data import ISACGraphData
from .diffusion import DiscreteDiffusion, reverse_step_digress
from .model import DenoisingNetwork


class DCDM_Trainer:
    def __init__(self, config, device=None):
        self.config = config
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.diffusion = DiscreteDiffusion(config, self.device)
        self.denoising_net = DenoisingNetwork(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.denoising_net.parameters(), lr=config.lr)
        self.reward_baseline = 0.0
        self.baseline_momentum = 0.9

    def compute_reward(self, graph_data):
        try:
            num_activated_links = int((graph_data.edge_type == 1).sum().item()) if graph_data.edge_type.numel() > 0 else 0
        except Exception:
            try:
                num_activated_links = int((torch.argmax(graph_data.edge_attr, dim=-1) == 1).sum().item())
            except Exception:
                num_activated_links = 0

        node_type_cpu = torch.argmax(graph_data.x, dim=-1).cpu() if graph_data.x.dim() > 1 else graph_data.node_type.cpu()
        num_tx = int((node_type_cpu == 0).sum().item())
        num_rx = int((node_type_cpu == 1).sum().item())

        if num_activated_links == 0 or num_tx == 0 or num_rx == 0:
            return 0.0

        ssnr_sum = 0.0
        num_activated_links /= 2

        device_pos = graph_data.device_pos.cpu()
        user_pos = graph_data.user_pos.cpu() if graph_data.user_pos.dim() > 1 else graph_data.user_pos.unsqueeze(0).cpu()
        edge_index = graph_data.edge_index.cpu()

        tx_indices = torch.where(node_type_cpu == 0)[0]
        rx_indices = torch.where(node_type_cpu == 1)[0]

        for tx in tx_indices:
            for rx in rx_indices:
                edge_mask = (edge_index[0] == tx) & (edge_index[1] == rx)
                if edge_mask.any():
                    idxs = torch.where(edge_mask)[0]
                    for eidx in idxs:
                        if hasattr(graph_data, "edge_type"):
                            edge_type_val = graph_data.edge_type[eidx].item()
                        else:
                            edge_type_val = torch.argmax(graph_data.edge_attr[eidx]).item()
                        if edge_type_val == 1:
                            tx_pos = device_pos[tx]
                            rx_pos = device_pos[rx]
                            dist_tx = torch.norm(user_pos - tx_pos)
                            dist_rx = torch.norm(user_pos - rx_pos)
                            dist_D = torch.norm(tx_pos - rx_pos)
                            denom = 4 * np.pi * (dist_tx * dist_rx) ** 2 * (1.0 / (dist_D ** 2 + 0.1))
                            ssnr = 100.0 / (denom.item() + 1e-9)
                            ssnr_sum += ssnr

        reward = self.config.alpha1 * ssnr_sum - self.config.alpha2 * num_tx - self.config.alpha3 * num_activated_links
        return float(reward)

    def sample_trajectory_from_G0(self, dataset, batch_size):
        trajectories = []
        rewards = []
        self.denoising_net.eval()

        indices = np.random.choice(len(dataset), batch_size)
        with torch.no_grad():
            for idx in indices:
                g0 = dataset[idx]
                noisy = self.diffusion.forward(g0)
                trajectory = []
                current = noisy

                for t in reversed(range(self.config.T)):
                    trajectory.append((current.clone(), t))
                    t_tensor = torch.tensor([t], dtype=torch.long).to(self.device)

                    node_logits, edge_logits = self.denoising_net(current, t_tensor)
                    p0_node = F.softmax(node_logits, dim=-1)
                    node_sample = reverse_step_digress(
                        current.x.clone(), t, p0_node, self.diffusion.node_Qt, self.diffusion.node_Q1
                    )
                    current.x = node_sample

                    if edge_logits.shape[0] > 0:
                        p0_edge = F.softmax(edge_logits, dim=-1)
                        edge_sample = reverse_step_digress(
                            current.edge_attr.clone(), t, p0_edge, self.diffusion.edge_Qt, self.diffusion.edge_Q1
                        )
                        current.edge_attr = edge_sample

                final_graph = ISACGraphData(
                    self.config.num_devices,
                    g0.user_pos[0].cpu().numpy(),
                    g0.device_pos.cpu().numpy(),
                )

                final_graph.node_type = torch.argmax(current.x, dim=-1).cpu()
                if current.edge_attr.shape[0] > 0:
                    final_graph.edge_type = torch.argmax(current.edge_attr, dim=-1).cpu()
                else:
                    final_graph.edge_type = torch.zeros((final_graph.edge_index.shape[1],), dtype=torch.long)

                if (final_graph.node_type == 0).sum().item() == 0:
                    final_graph.node_type[random.randint(0, final_graph.num_nodes - 1)] = 0
                if (final_graph.node_type == 1).sum().item() == 0:
                    final_graph.node_type[random.randint(0, final_graph.num_nodes - 1)] = 1

                final_graph.edge_index = final_graph._generate_edges()
                if final_graph.edge_index.shape[1] > 0:
                    if final_graph.edge_type.shape[0] != final_graph.edge_index.shape[1]:
                        final_graph.edge_type = torch.zeros((final_graph.edge_index.shape[1],), dtype=torch.long)

                    if final_graph.edge_type.sum().item() == 0:
                        valid_edges = []
                        for ei in range(final_graph.edge_index.shape[1]):
                            tx_idx = final_graph.edge_index[0, ei].item()
                            rx_idx = final_graph.edge_index[1, ei].item()
                            tx_pos = final_graph.device_pos[tx_idx]
                            rx_pos = final_graph.device_pos[rx_idx]
                            if final_graph._is_in_fresnel(tx_pos, rx_pos, final_graph.user_pos):
                                valid_edges.append(ei)
                        if len(valid_edges) > 0:
                            final_graph.edge_type[random.choice(valid_edges)] = 1
                        else:
                            final_graph.edge_type[random.randint(0, final_graph.edge_index.shape[1] - 1)] = 1

                rewards.append(self.compute_reward(final_graph.to_data()))
                trajectories.append(trajectory)

        self.denoising_net.train()
        return trajectories, rewards

    def train_step(self, dataset):
        self.denoising_net.train()
        total_loss_val = 0.0
        trajectories, rewards = self.sample_trajectory_from_G0(dataset, self.config.batch_size)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        mean_reward = rewards.mean().item() if rewards.numel() > 0 else 0.0
        self.reward_baseline = (
            self.baseline_momentum * self.reward_baseline + (1 - self.baseline_momentum) * mean_reward
        )
        advantages = (rewards - self.reward_baseline).to(self.device)

        self.optimizer.zero_grad()
        for i, traj in enumerate(trajectories):
            adv = advantages[i].item() if advantages.numel() > 0 else 0.0
            k = min(random.randint(1, self.config.T), len(traj))
            sampled = random.sample(traj, k)
            for noisy_data_t, t in sampled:
                t_tensor = torch.tensor([t], dtype=torch.long).to(self.device)
                node_logits, edge_logits = self.denoising_net(noisy_data_t, t_tensor)
                node_target = noisy_data_t.x
                log_prob_node = F.log_softmax(node_logits, dim=-1) * node_target
                log_prob_node = log_prob_node.sum(dim=-1).mean()

                log_prob_edge = torch.tensor(0.0).to(self.device)
                if edge_logits.shape[0] > 0:
                    edge_target = noisy_data_t.edge_attr
                    log_prob_edge = F.log_softmax(edge_logits, dim=-1) * edge_target
                    log_prob_edge = log_prob_edge.sum(dim=-1).mean()

                total_log_prob = log_prob_node + log_prob_edge
                loss = -adv * total_log_prob
                total_loss_val += float(loss.detach().cpu().item())
                loss.backward()

        torch.nn.utils.clip_grad_norm_(self.denoising_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        return total_loss_val / max(1, self.config.batch_size)


DCDMTrainer = DCDM_Trainer
