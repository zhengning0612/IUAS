import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .diffusion import reverse_step_digress


def visualize_g0_comparison(trainer, dataset, num_samples=6, save_path="g0_comparison.png", show=True):
    trainer.denoising_net.eval()

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for idx in range(num_samples):
            g0_original = dataset[np.random.randint(len(dataset))]

            user_pos_orig = g0_original.user_pos[0].cpu().numpy()
            device_pos_orig = g0_original.device_pos.cpu().numpy()
            node_type_orig = torch.argmax(g0_original.x, dim=-1).cpu().numpy()
            edge_index_orig = g0_original.edge_index.cpu().numpy()
            edge_type_orig = torch.argmax(g0_original.edge_attr, dim=-1).cpu().numpy()

            current = trainer.diffusion.forward(g0_original)
            for t in reversed(range(trainer.config.T)):
                t_tensor = torch.tensor([t], dtype=torch.long).to(trainer.device)
                node_logits, edge_logits = trainer.denoising_net(current, t_tensor)

                p0_node = F.softmax(node_logits, dim=-1)
                current.x = reverse_step_digress(
                    current.x.clone(), t, p0_node, trainer.diffusion.node_Qt, trainer.diffusion.node_Q1
                )

                if edge_logits.shape[0] > 0:
                    p0_edge = F.softmax(edge_logits, dim=-1)
                    current.edge_attr = reverse_step_digress(
                        current.edge_attr.clone(), t, p0_edge, trainer.diffusion.edge_Qt, trainer.diffusion.edge_Q1
                    )

            node_type_denoised = torch.argmax(current.x, dim=-1).cpu().numpy()
            edge_type_denoised = (
                torch.argmax(current.edge_attr, dim=-1).cpu().numpy() if current.edge_attr.shape[0] > 0 else np.array([])
            )

            _plot_single_graph(
                axes[idx, 0],
                user_pos_orig,
                device_pos_orig,
                node_type_orig,
                edge_index_orig,
                edge_type_orig,
                title=f"Original G0 (Sample {idx + 1})",
            )
            _plot_single_graph(
                axes[idx, 1],
                user_pos_orig,
                device_pos_orig,
                node_type_denoised,
                edge_index_orig,
                edge_type_denoised,
                title=f"Denoised G0 (Sample {idx + 1})",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"G0 comparison visualization saved as '{save_path}'")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_single_graph(ax, user_pos, device_pos, node_type, edge_index, edge_type, title=""):
    ax.scatter(
        user_pos[0],
        user_pos[1],
        c="red",
        s=200,
        marker="*",
        label="User",
        zorder=5,
        edgecolors="black",
        linewidths=1.5,
    )

    tx_mask = node_type == 0
    rx_mask = node_type == 1

    if tx_mask.any():
        ax.scatter(
            device_pos[tx_mask, 0],
            device_pos[tx_mask, 1],
            c="blue",
            s=150,
            marker="^",
            label="Tx",
            zorder=4,
            edgecolors="black",
            linewidths=1,
        )
    if rx_mask.any():
        ax.scatter(
            device_pos[rx_mask, 0],
            device_pos[rx_mask, 1],
            c="green",
            s=150,
            marker="o",
            label="Rx",
            zorder=4,
            edgecolors="black",
            linewidths=1,
        )

    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i]
        dst_idx = edge_index[1, i]
        src_pos = device_pos[src_idx]
        dst_pos = device_pos[dst_idx]

        if i < len(edge_type) and edge_type[i] == 1:
            ax.plot([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]], "r-", linewidth=2, alpha=0.7, zorder=2)
        else:
            ax.plot(
                [src_pos[0], dst_pos[0]],
                [src_pos[1], dst_pos[1]],
                "gray",
                linestyle="--",
                linewidth=1,
                alpha=0.3,
                zorder=1,
            )

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_xlabel("X Position", fontsize=11)
    ax.set_ylabel("Y Position", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_aspect("equal")
