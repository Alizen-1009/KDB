# Torch Distributed

## 定义

PyTorch 提供的分布式训练与通信接口层，用于组织多进程、多 GPU 环境下的 collective operation 和更高层并行训练功能。

## 它解决什么问题

- 给训练代码提供统一的分布式通信 API
- 让开发者在不直接操纵底层通信细节的情况下实现数据并行、张量并行或更高层封装

## 核心机制

- 暴露 `init_process_group`、`all_reduce`、`all_gather`、`reduce_scatter`、`send`、`recv` 等接口
- 通过 backend 把高层调用映射到底层执行库，如 GPU 上常见的 NCCL
- 在这些原语之上还能构建更高层算法，例如 DDP 和 FSDP

## 关键权衡

- 接口统一、工程可用性高
- 但真正性能仍然取决于 backend、网络拓扑、张量大小和调用时机

## 相关实体

- [[../entities/NCCL]]
- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 8 - Distributed communication and training code]]

## 相关概念

- [[集合通信]]
- [[数据并行]]
- [[FSDP]]

## 研究备注

- 后续可补 `gloo`、`nccl` 等 backend 的适用边界，以及 PyTorch 分布式 API 向 DeviceMesh / DTensor 的演化
