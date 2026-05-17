import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def compute_transition_powers(Q1, T):
    class_count = Q1.size(0)
    device = Q1.device
    Qt_list = []
    Qt = torch.eye(class_count, device=device)
    Qt_list.append(Qt.clone())
    for _ in range(1, T + 1):
        Qt = Qt @ Q1
        Qt_list.append(Qt.clone())
    return Qt_list


def reverse_step_digress(x_t, t, p0, Qt_list, Q1):
    device = x_t.device
    _, class_count = x_t.size()
    Q1 = Q1.to(device)

    if t == 0:
        idx = torch.argmax(p0.to(device), dim=-1)
        return F.one_hot(idx, class_count).float().to(device)

    Qt_prev = Qt_list[t - 1].to(device)
    p0 = p0.to(device)

    temp1 = p0 @ Qt_prev
    temp2 = x_t @ Q1.t()
    posterior = temp1 * temp2
    posterior = posterior + 1e-9
    posterior = posterior / posterior.sum(dim=-1, keepdim=True)

    idx = torch.multinomial(posterior, 1).squeeze(-1)
    return F.one_hot(idx, class_count).float().to(device)


class DiscreteDiffusion:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.node_Q1 = self._make_Q1(config.node_types).to(device)
        self.edge_Q1 = self._make_Q1(config.edge_types).to(device)
        self.node_Qt = compute_transition_powers(self.node_Q1, config.T)
        self.edge_Qt = compute_transition_powers(self.edge_Q1, config.T)

    def _make_Q1(self, class_count):
        noise = 1.0 / self.config.T
        Q = (1 - noise) * torch.eye(class_count)
        Q = Q + noise * (torch.ones(class_count, class_count) - torch.eye(class_count)) / (class_count - 1)
        return Q

    def forward(self, data):
        x = data.x.clone().to(self.device)
        edge_attr = data.edge_attr.clone().to(self.device)

        for _ in range(self.config.T):
            x = x @ self.node_Q1
            if edge_attr.shape[0] > 0:
                edge_attr = edge_attr @ self.edge_Q1

        return Data(
            x=x,
            edge_index=data.edge_index.to(self.device),
            edge_attr=edge_attr,
            user_pos=data.user_pos.to(self.device),
            device_pos=data.device_pos.to(self.device),
        )
