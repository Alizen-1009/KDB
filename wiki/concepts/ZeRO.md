# ZeRO

## 定义

一类通过分片参数相关状态来降低数据并行内存开销的训练优化方法，核心思想是“不再让每张卡完整持有所有昂贵状态”。

## 它解决什么问题

- 缓解数据并行下参数、梯度和优化器状态全量复制带来的巨大内存压力
- 在不完全转向模型并行的前提下提升可训练模型规模

## 核心机制

- Stage 1：分片 optimizer state
- Stage 2：进一步分片 gradients
- Stage 3：连参数也一起分片，并按需通信和释放

## 关键权衡

- Stage 1 和 Stage 2 通常能以较低额外通信换到明显内存收益
- Stage 3 内存收益最大，但通信与调度复杂度也最高

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 7 - Parallelism basics]]

## 相关概念

- [[数据并行]]
- [[FSDP]]
- [[集合通信]]

## 研究备注

- 后续可补 ZeRO 在 DeepSpeed 与 PyTorch FSDP 中的实现差异，以及 BF16 / optimizer state 精度对内存账本的影响
