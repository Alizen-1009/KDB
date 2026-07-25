---
type: concept
topic: 并行与分布式
sources: 1
updated: 2026-07-25
---

# 跨 Mesh 权重重分片

## 定义

跨 Mesh 权重重分片（Mesh-to-Mesh Resharding）是把同一个逻辑张量从一组 GPU 的分片/复制布局，直接转换为另一组 GPU 所需布局的数据移动过程。源侧和目标侧可以采用不同的 TP、EP、PP 或复制策略，因此不能简单按相同 rank 编号复制。

## 什么时候需要训练与推理交互

传统离线流程通常是：

```text
训练完成 -> 保存 checkpoint -> 推理加载 -> 两侧不再通信
```

这种静态部署不需要持续的 Mesh-to-Mesh 通信，最多在部署时做一次权重转换。

在线强化学习不同。PPO、GRPO 等系统通常把 Rollout 推理作为训练循环的一部分：

```text
Rollout Mesh 使用 policy version N 生成轨迹
    -> response / reward / old logprob 等训练数据返回 Trainer
Training Mesh 更新策略，得到 version N+1
    -> 新策略权重重分片并同步到 Rollout Mesh
下一轮 Rollout 使用 version N+1
```

因此存在两条不同的数据流：

- `Rollout -> Training`：轨迹、response、reward、logprob、advantage 等训练样本；这不是 M2N 的主要职责。
- `Training -> Rollout`：更新后的 policy 权重；这是 NCCL M2N 的主要场景。

权重通常在训练 step、Rollout iteration 或模型版本边界切换，不会在同一条 response 生成到一半时随意替换。异步 RL 可以容忍不同 worker 使用稍旧版本，但策略陈旧度、版本一致性和切换时机属于上层 runtime 问题，不由重分片原语决定。

## 为什么两个 Mesh 布局会不同

训练侧需要参数、梯度、Optimizer state、反向激活，常按大规模训练效率选择 `DP + TP + PP + FSDP/ZeRO + EP`。Rollout 推理侧没有 backward，但受 KV Cache、decode latency、continuous batching 和专家吞吐影响，可能采用不同的 `DP/TP/PP/EP` 配比。

例如同一个 `W[8, 8]`：

```text
训练侧沿行切 4 份：
T0 = W[0:2, :]
T1 = W[2:4, :]
T2 = W[4:6, :]
T3 = W[6:8, :]

推理侧沿列切 2 份：
I0 需要 W[:, 0:4]
I1 需要 W[:, 4:8]
```

此时每个推理 rank 都需要从多个训练 rank 取得局部区域，不能使用 `T0 -> I0、T1 -> I1` 的一一映射。

## 核心机制

NCCL M2N 的抽象包含：

- `ncclMesh_t`：描述二维 Mesh、起始 rank，以及每个 Mesh 轴是复制还是切某个 tensor dimension。
- `ncclDistTensor_t`：描述本 rank 的数据指针、local shape、dtype 和所属 Mesh。
- `ncclReshardWithWindow`：使用 communicator 和 NCCL Window 执行 reshard。

重分片可以归约为全局坐标求交：

1. 把每个源 rank 的 local shard 映射到全局张量区间。
2. 把每个目标 rank 所需 local shard 映射到全局区间。
3. 计算源区间和目标区间的交集。
4. 根据交集生成源/目标 offset、stride、连续 chunk 和外层循环组成的 TransferPlan。
5. Device kernel 按计划把数据写入目标 Window。

```text
source rank 持有范围 ∩ destination rank 需要范围
= 这对 rank 之间需要传输的数据
```

复杂拓扑分析和 TransferPlan 主要在 Host 侧完成，Device 侧消费紧凑计划执行热路径，因此更准确的说法是“Host 规划、Device 执行”。

## RING 与 DIRECT

- `DIRECT`：每个源 rank 直接向目标 ranks 发起 GIN put；小传输可能延迟较低，但目标扇出大时会增加 NIC 和准备阶段压力。
- `RING`：使用层次化 ring 与域内 fan-out，减少大规模跨 NVLink 域传输的直接扇出，更适合特定大规模拓扑。

来源文章称 GB200 NVL72 的模型级 benchmark 中，RING 相比 Direct P2P 的端到端最大延迟有 `2.20x–2.65x` 优势；这是特定来源 benchmark，不是所有 tensor shape 和 Mesh 的通用结论。

## “零拷贝”的边界

这里的零拷贝主要指：

- 避免先把所有 shard 聚合为完整权重；
- 避免 GPU 到 CPU、文件系统再回到 GPU 的 staging；
- 避免不必要的中间 Buffer；
- 直接从源 GPU 的局部 shard 写入目标 GPU Window 中的目标位置。

数据仍然要经过 NVLink、NVSwitch、RDMA 等互联移动，因此“零拷贝”不等于零通信或零网络流量。

## 与 checkpoint 同步的区别

传统方式：

```text
训练 GPU -> 聚合/保存 checkpoint -> 共享存储
-> 推理侧读取 -> 按目标布局重新切分 -> 加载
```

M2N 方向：

```text
训练 GPU 上已有 shard
-> GPU-to-GPU overlap-aware reshard
-> Rollout GPU 目标 shard
```

如果只是训练结束后部署一次，且转换延迟不重要，checkpoint 往往已经足够。M2N 更适合在线 RL、持续训练到服务、独立评估集群和频繁策略刷新。

## 关键权衡

- 优点：减少 Host staging、完整权重聚合和文件系统 I/O，适合频繁同步超大模型。
- 代价：需要描述两套 Mesh、维护 Window/communicator、生成传输计划，并处理策略版本和切换一致性。
- 性能强依赖源/目标 shard 形状、网络拓扑、NVLink 域、chunk size、CTA 数和 RING/DIRECT 选择。
- M2N 只负责权重数据通路；Rollout 调度、轨迹回传、策略陈旧度和原子版本切换仍由上层 RL 系统负责。

## 面试口径

一句话：在线 RL 中 Rollout 推理不是训练完成后的独立服务，而是训练循环的数据生成阶段；Trainer 更新策略后，需要把新权重从训练布局直接重分片成 Rollout 布局，M2N 就负责这条 `Training -> Rollout` 权重同步路径。

## 相关实体

- [[../entities/NCCL Extensions]]
- [[../entities/NCCL]]

## 相关来源

- [[../sources/NVIDIA 开源 NCCL Extensions：把 MoE 专家路由与跨 Mesh 权重重分片推进到 GPU 设备侧]]

## 相关概念

- [[Tensor Parallelism]]
- [[Expert Parallelism]]
- [[流水线并行]]
- [[集合通信]]
- [[通信-计算重叠]]

## 研究备注

- 当前依据为仓库解读文章；M2N 支持的 tensor 维数、Mesh 约束、非均匀 shard、源/目标组重叠规则和一致性语义应按对应源码 commit 核实。
- 后续可结合具体 PPO/GRPO runtime 研究权重同步 barrier、异步 rollout policy staleness 和不中断 CUDA Graph 的权重切换方式。
