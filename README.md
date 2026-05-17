# IUAS

《Generative AI Based Secure Wireless Sensing for ISAC Networks》的仿真代码。项目基于图 Transformer + 离散扩散模型，在固定 ISAC 设备分布、动态用户位置条件下，自动生成高 SSNR、低资源占用的通信拓扑图，为室内用户安全感知提供最优通信链路部署方案。

## 文件结构

- `main.py`: 主函数入口.
- `dcdm/config.py`: 参数配置.
- `dcdm/data.py`: 自定义图结构转换成 PyTorch Geometric 的 Data 对象.
- `dcdm/diffusion.py`: 离散扩散转换和反向采样.
- `dcdm/model.py`: 网络架构.
- `dcdm/trainer.py`: 模型训练.
- `dcdm/evaluation.py`: 模型评估.
- `dcdm/visualization.py`: 可视化.
- `dcdm/runtime.py`: 运行时设置和运行报告.
- `dcdm/datasets.py`: 数据加载.
- `dcdm/metrics.py`: 奖励统计数据.
- `dcdm/checkpointing.py`: 检查点保存/加载.

## 运行

Put `G0_dataset.pt` and `G0_dataset_test.pt` under `graph/`, then run:

```bash
python main.py
```

For a quicker smoke run:

```bash
python main.py --epochs 1 --eval-batches 1 --comparison-samples 1 --no-show
```

## 项目声明

本项目的作者及单位：
The author and affiliation of this project:

```text
项目名称 (Project Name): IUAS
项目作者 (Author): Ning Zheng, Zhen Chen
作者单位 (Affiliation): 暨南大学网络空间安全学院 (College of Cyber Security, Jinan University)
```
