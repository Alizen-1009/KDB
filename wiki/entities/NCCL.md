---
type: entity
entity_type: 框架
topic: 并行与分布式
sources: 2
updated: 2026-04-23
---

# NCCL

## 一句话说明

NVIDIA 提供的集合通信库，用于把多 GPU / 多节点训练中的 collective operation 高效映射到实际硬件通信路径上。

## 类型

- 通信库 / 基础设施

## 核心信息

- 全称通常指 `NVIDIA Collective Communication Library`。
- 在 PyTorch 分布式 GPU 训练中，经常作为 backend 承担 collective communication 的实际执行。
- 它的性能高度依赖 GPU 间互联和节点拓扑，例如 NVLink、NVSwitch、PCIe 以及跨节点网络。
- `vLLM AFD Plugin` 的 `P2pNcclAFDConnector` 使用同步 P2P 路径在 Attention 与 FFN 服务间交换激活，面向 decode，并支持来源所述的 `FULL_DECODE_ONLY` CUDA Graph 路径。

## 相关概念

- [[集合通信]]
- [[Torch Distributed]]
- [[数据并行]]
- [[Attention-FFN 分离]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 8 - Distributed communication and training code]]
- [[../sources/vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署]]

## 冲突与备注

- 归档时要区分 `torch.distributed` 这样的接口层和 `NCCL` 这样的 backend 执行层，它们不是一回事
