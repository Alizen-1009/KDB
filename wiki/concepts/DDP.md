---
type: concept
topic: 并行与分布式
sources: 2
updated: 2026-05-17
---

# DDP

## 定义

`DDP`（DistributedDataParallel）是 PyTorch 中最常用的数据并行训练封装：每个 rank 持有完整模型副本，处理不同 mini-batch shard，在 backward 过程中通过梯度 `all-reduce` 保持参数更新一致。

## 它解决什么问题

- 在模型能放进单卡时，用多卡并行提升训练吞吐。
- 避免单进程多卡 `DataParallel` 的 Python/GIL 与单进程调度瓶颈。
- 用 bucket 化和通信计算重叠降低梯度同步开销。

## 核心机制

- `torchrun` 或 launcher 启动多个进程，通常一进程绑定一张 GPU。
- `init_process_group` 建立通信组，GPU 训练通常使用 `NCCL` backend。
- 每个 rank 初始化同构模型，并由 DDP 在初始化时广播参数，确保副本一致。
- forward 各 rank 独立处理自己的 batch shard。
- backward 中 autograd hook 在梯度 ready 后触发 bucket all-reduce。
- all-reduce 后各 rank 得到相同平均梯度，再各自执行 optimizer step。

## 面试高频考点

- `rank / local_rank / world_size`：rank 是全局进程序号，local_rank 是节点内 GPU 序号，world_size 是总进程数。
- `torchrun` 启动链路：设置 rendezvous、`MASTER_ADDR`、`MASTER_PORT`、rank/world size，再初始化 process group。
- `DistributedSampler`：保证不同 rank 读不同数据 shard；每个 epoch 需要 `sampler.set_epoch(epoch)` 保证 shuffle 一致且可复现。
- 全局 batch：`global_batch = per_device_batch * world_size * grad_accum_steps`。
- 梯度同步时机：DDP 默认在 backward 期间同步，不是 optimizer step 之后才整体同步。
- Bucket：多个参数梯度被打包成 bucket，bucket ready 就 all-reduce，从而与后续 backward 计算 overlap。
- `no_sync()`：梯度累积时跳过中间 step 的 all-reduce，只在累积边界同步。
- `find_unused_parameters`：动态图或条件分支中有未参与 loss 的参数时需要关注；开启会增加 autograd graph traversal 开销。
- `static_graph=True`：图结构稳定时可减少额外开销，但要求每轮用到的参数集合稳定。
- `SyncBatchNorm`：普通 BN 在每个 rank 内统计，跨卡同步统计需转换为 `SyncBatchNorm`。
- 随机性：需要按 rank 设置 seed，并注意 dropout、sampler、数据增强的可复现边界。

## 通信量与性能

- DDP 每个 step 主要通信对象是梯度，逻辑通信量约等于模型参数量对应的 gradient bytes。
- Ring all-reduce 下，每个 rank 每个 bucket 的发送量约 `2 * (P - 1) / P * bucket_bytes`。
- 性能关键不是只看总通信量，还要看 bucket 大小、bucket 数量、通信是否与 backward overlap、网络拓扑和是否出现 rank straggler。

## 常见故障与排查

- 启动失败：检查 `MASTER_ADDR / MASTER_PORT`、端口冲突、防火墙、节点间 hostname/IP、rank/world size 是否一致。
- NCCL timeout：不一定是 NCCL 坏，常见根因是某个 rank 数据异常、shape 不一致、提前 OOM、collective 调用顺序不一致或网络拓扑问题。
- Hang 在 backward：检查是否有 rank 跳过 loss/backward、条件分支导致某些参数未产生梯度、不同 rank batch shape 或 control flow 不一致。
- 数据重复或漏样本：检查 `DistributedSampler`、`drop_last`、`set_epoch`、dataset 长度和恢复训练时的 sampler state。
- 性能差：用 `Nsight Systems` 或 profiler 看 all-reduce 是否在关键路径、bucket 是否太碎、GPU 是否在等 dataloader、rank 间 step time 是否不均。

## 面试回答模板

> DDP 的核心是每个进程一张 GPU、每个 rank 持有完整模型副本，数据按 batch 维切分。forward 各 rank 独立算，backward 时 DDP 给参数梯度注册 autograd hook，梯度 ready 后按 bucket 触发 all-reduce，所以通信可以和后续反向计算重叠。optimizer step 前各 rank 已经拿到一致的平均梯度，因此参数保持同步。面试里我会重点关注启动链路、rank/local_rank/world_size、DistributedSampler、bucket overlap、no_sync 梯度累积、find_unused_parameters、SyncBatchNorm，以及 NCCL hang 的排查。排查 hang 时不能只盯 NCCL，要看是否某个 rank 数据 shape、控制流、OOM 或 collective 顺序先出了问题。

## 相关实体

- [[../entities/NCCL]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 7 - Parallelism basics]]
- [[../sources/斯坦福CS336 Lecture 8 - Distributed communication and training code]]

## 相关概念

- [[数据并行]]
- [[Torch Distributed]]
- [[集合通信]]
- [[FSDP]]
- [[ZeRO]]

## 研究备注

- 后续可补 PyTorch DDP reducer/bucket 源码路径、`gradient_as_bucket_view`、communication hook、mixed precision DDP 与 ZeRO/FSDP 的差异。
