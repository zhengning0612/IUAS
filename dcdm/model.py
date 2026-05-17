import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import LayerNorm, MessagePassing
from torch_geometric.utils import softmax as pyg_softmax


class GraphTransformerLayer(MessagePassing):
    def __init__(self, hidden_dim, num_heads):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(hidden_dim, num_heads)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ln1 = LayerNorm(hidden_dim)
        self.ln2 = LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = self.propagate(edge_index, q=q, k=k, v=v, edge_attr=edge_attr, size=None)
        x = self.ln1(x + self.out_proj(out))
        x = self.ln2(x + self.ffn(x))
        return x

    def message(self, q_i, k_j, v_j, edge_attr, index):
        edge_count = q_i.size(0)
        head_count = self.num_heads
        q_i = q_i.view(edge_count, head_count, self.head_dim)
        k_j = k_j.view(edge_count, head_count, self.head_dim)
        v_j = v_j.view(edge_count, head_count, self.head_dim)

        attn = torch.sum(q_i * k_j, dim=-1) / np.sqrt(self.head_dim)
        if edge_attr is not None and edge_attr.shape[0] > 0:
            attn = attn + self.edge_proj(edge_attr)

        attn_flat = attn.view(-1)
        index_rep = index.repeat_interleave(head_count)
        attn_flat_norm = pyg_softmax(attn_flat, index_rep)
        attn_weight = attn_flat_norm.view(edge_count, head_count)

        out = v_j * attn_weight.unsqueeze(-1)
        return out.view(edge_count, self.hidden_dim)


class DenoisingNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.node_emb = nn.Linear(config.node_types, config.hidden_dim)
        self.edge_emb = nn.Linear(config.edge_types, config.hidden_dim)
        self.time_emb = nn.Embedding(config.T, config.hidden_dim)

        self.flm_scale = nn.Linear(4, config.hidden_dim)
        self.flm_shift = nn.Linear(4, config.hidden_dim)

        self.layers = nn.ModuleList(
            [GraphTransformerLayer(config.hidden_dim, config.num_heads) for _ in range(config.num_layers)]
        )

        self.node_out = nn.Linear(config.hidden_dim, config.node_types)
        self.edge_out = nn.Linear(config.hidden_dim, config.edge_types)

    def forward(self, noisy_data, t):
        x = noisy_data.x
        edge_index = noisy_data.edge_index
        edge_attr = noisy_data.edge_attr

        user_pos = noisy_data.user_pos
        device_pos = noisy_data.device_pos
        device_avg = device_pos.mean(dim=0, keepdim=True)
        condition = torch.cat([user_pos, device_avg], dim=-1)

        x = self.node_emb(x)
        if edge_attr is not None and edge_attr.shape[0] > 0:
            edge_attr = self.edge_emb(edge_attr)
        else:
            edge_attr = torch.zeros((0, self.config.hidden_dim), device=x.device)

        time_emb = self.time_emb(t).squeeze(0)
        x = x + time_emb

        scale = self.flm_scale(condition).sigmoid()
        shift = self.flm_shift(condition)
        x = x * scale + shift

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        node_logits = self.node_out(x)
        if edge_index.shape[1] > 0:
            src, dst = edge_index
            edge_feat = x[src] + x[dst]
            edge_logits = self.edge_out(edge_feat)
        else:
            edge_logits = torch.empty((0, self.config.edge_types), device=x.device)
        return node_logits, edge_logits
