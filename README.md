# D-CDM ISAC Graph Diffusion

项目基于图 Transformer + 离散扩散模型，在固定 ISAC 设备分布、动态用户位置条件下，自动生成高 SSNR、低资源占用的通信拓扑图，为室内用户安全感知提供最优通信链路部署方案。
The project is based on a graph Transformer combined with a discrete diffusion model. Under the conditions of fixed ISAC device distribution and dynamic user locations, it automatically generates a communication topology graph with high SSNR and low resource consumption, providing the optimal communication link deployment solution for indoor user security perception.
## Structure

- `main.py`: command-line entry point for training, evaluation, and plotting.
- `dcdm/config.py`: experiment configuration.
- `dcdm/data.py`: ISAC graph construction and PyG `Data` conversion.
- `dcdm/diffusion.py`: discrete diffusion transition and reverse sampling utilities.
- `dcdm/model.py`: graph transformer denoising network.
- `dcdm/trainer.py`: D-CDM training and reward logic.
- `dcdm/evaluation.py`: model evaluation and reward curve plotting.
- `dcdm/visualization.py`: G0 comparison visualization.
- `dcdm/runtime.py`: runtime setup, reproducibility, paths, and run reports.
- `dcdm/datasets.py`: dataset loading, validation, and summary reports.
- `dcdm/metrics.py`: reward statistics, epoch records, and CSV history.
- `dcdm/checkpointing.py`: checkpoint save/load, resume, and retention helpers.

## Run

Put `G0_dataset.pt` and `G0_dataset_test.pt` under `graph/`, then run:

```bash
python main.py
```

For a quicker smoke run:

```bash
python main.py --epochs 1 --eval-batches 1 --comparison-samples 1 --no-show
```

Useful project-style options:

```bash
python main.py --seed 42 --save-every 10 --keep-checkpoints 3
python main.py --resume-latest
python main.py --output-dir runs/experiment_001 --no-show
```

Training now writes:

- `dcdm_denoising_net.pth`: final model weights.
- `logs/training_history.csv`: per-epoch loss and reward metrics.
- `checkpoints/checkpoint_epoch_XXXX.pt`: resumable training checkpoints.
- `checkpoints/best.pt`: best checkpoint by valid average reward.
