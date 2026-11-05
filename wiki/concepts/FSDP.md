---
type: concept
topic: 并行与分布式
sources: 1
updated: 2026-04-23
---

# FSDP

## 定义

一种以模块或 block 为粒度，在前向和反向过程中按需 all-gather 参数、按需 reduce-scatter 梯度的全分片训练方式，常被视作 ZeRO Stage 3 的工程化实现路径。

## 它解决什么问题

- 让超大模型在更多设备上以接近线性的参数内存缩放方式训练
- 避免所有设备始终完整持有全部参数

## 核心机制

- 参数默认分片存放
- 前向到某个 block 时临时 all-gather 所需参数
- 反向后再把梯度 reduce-scatter，并尽快释放不再需要的完整参数/梯度

## 前反向与更新流程

1. 前向前按 block AllGather 参数，获得当前 block 的完整权重。全分片策略可在前向后释放完整权重缓冲，保留本地分片。
2. 反向前重新 AllGather 已释放的参数，计算本地 batch 的梯度。反向后 ReduceScatter 梯度，让每个 rank 获得归约后的对应梯度分片，并释放临时完整参数。
3. optimizer step 用本地梯度分片更新本地参数与 Adam 状态。梯度归约与优化器状态更新发生在不同阶段，Adam 状态留在其 owner 上。全局梯度范数等辅助统计可另外触发归约。

FSDP1 的 `FULL_SHARD` 对应 ZeRO-3 风格。`SHARD_GRAD_OP` 前向后保留完整权重，反向完成后重新分片，因此省去反向前的 AllGather，通信量接近 ZeRO-2，但完整参数的存放时机有差异。FSDP2 使用 `reshard_after_forward` 控制该权衡。每卡字节数见 [[ZeRO#每步通信与单卡状态显存|ZeRO 通信账本]]。

按 block 划分通信组，可以预取下一 block 的参数并与当前 block 计算重叠。组太细会增加 collective 次数，预取也会增加同时驻留的临时完整权重。

- [PyTorch FSDP1 ShardingStrategy](https://docs.pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy)
- [PyTorch FSDP2 fully_shard](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)

## 关键权衡

- 参数内存缩放很好
- 但通信时机更细粒度，对 overlap、调度和网络质量更敏感

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 7 - Parallelism basics]]

## 相关概念

- [[ZeRO]]
- [[数据并行]]
- [[重计算]]

## 研究备注

- 后续可补 FSDP block 划分粒度、prefetch 策略和 activation checkpointing 的协同关系
