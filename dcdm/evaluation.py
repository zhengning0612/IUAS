import torch
import matplotlib.pyplot as plt


def evaluate_model(trainer, dataset, batch_size=50, num_batches=200, save_path="fig9_reward_curve.png", show=True):
    trainer.denoising_net.eval()
    rewards_all = []
    test_rewards_per_epoch = []

    print(f"\n=== Start Testing {batch_size} Samples ===\n")

    with torch.no_grad():
        for batch_idx in range(num_batches):
            _, rewards = trainer.sample_trajectory_from_G0(dataset, batch_size=batch_size)
            valid_rewards = [reward for reward in rewards if reward != 0]
            rewards_all.extend(valid_rewards)
            avg_reward = sum(valid_rewards) / len(valid_rewards) if len(valid_rewards) > 0 else 0
            test_rewards_per_epoch.append(avg_reward)
            print(f"Test Batch {batch_idx + 1}/{num_batches} | Valid Avg Reward: {avg_reward:.4f}")

    final_avg = sum(rewards_all) / len(rewards_all) if len(rewards_all) > 0 else 0

    print("\n=== Testing Completed ===")
    print(f"Final Average Reward over {batch_size} samples: {final_avg:.4f}")

    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_batches + 1), test_rewards_per_epoch, linewidth=2)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel(f"Average Reward ({batch_size} samples)", fontsize=14)
    plt.title("Reward vs Epoch (Fig. 9 Reproduction)", fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return final_avg
