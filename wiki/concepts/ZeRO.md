---
type: concept
topic: 并行与分布式
sources: 1
updated: 2026-04-23
---

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

## 每步通信与单卡状态显存

设数据并行度为 $N$，$W$、$G$、$O$ 分别为完整参数、梯度、优化器状态的字节数。显存列按状态存储精度计算；通信列的 $W$、$G$ 按传输精度计算。混合精度下，若 FP32 master weights 独立维护，将其计入 $O$。

以一次前反向对应一次更新、均匀分片、ring collective 为例。设 $f=(N-1)/N$。对大小为 $S$ 的完整张量，每卡单向发送量为 AllGather 或 ReduceScatter 的 $fS$，以及 AllReduce 的 $2fS$。收发合计乘二。忽略 padding、梯度范数等额外通信。

| 方案 | 分片内容 | 每卡状态显存 | 主要通信 | 每卡发送量 |
| --- | --- | --- | --- | --- |
| DDP | 全量复制 | $W+G+O$ | 梯度 AllReduce | $2fG$ |
| ZeRO-1 | 优化器状态 | $W+G+O/N$ | 梯度归约 → 本地更新 → 收集更新后的参数 | 梯度全量 AllReduce 时为 $f(2G+W)$；按 owner 归约且参数用 AllGather 时为 $f(G+W)$ |
| ZeRO-2 | 优化器状态、梯度 | $W+(G+O)/N$ | 梯度 ReduceScatter → 本地更新 → 更新后的参数 AllGather | $f(G+W)$ |
| ZeRO-3 | 优化器状态、梯度、参数 | $(W+G+O)/N$ | 前向与反向各一次参数 AllGather，反向后梯度 ReduceScatter | $f(2W+G)$ |

ZeRO-1 的公式以更新后参数按等量 ring AllGather 分发估算。实际也可用 Broadcast，例如 PyTorch 的 DDP + ZeroRedundancyOptimizer。ZeRO-2/3 列为论文通信模型，实际 collective 受框架、版本和配置影响。ZeRO-3 一行假设前向后释放完整参数。状态显存公式不含 activation、临时完整权重或梯度、bucket 和碎片。

当参数和梯度的传输字节数相等时，ZeRO-2 与 DDP 的主通信量相同，ZeRO-3 为 DDP 的 1.5 倍。ZeRO-2 将 DDP 中可视为梯度 AllGather 的部分换成了更新后参数的 AllGather。通信量是机制估算，step 延迟还受 bucket 粒度、网络拓扑和通信计算重叠影响。

- [ZeRO 论文 §7.2](https://arxiv.org/abs/1910.02054)
- [DeepSpeed ZeRO 配置](https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training)
- [PyTorch ZeroRedundancyOptimizer](https://docs.pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer)
- [NCCL Tests 通信量口径](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)

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
