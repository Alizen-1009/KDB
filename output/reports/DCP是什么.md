# DCP 是什么

## 背景

`DCP` 指 **Decode Context Parallel（解码上下文并行）**，是一种面向长上下文 LLM 推理 decode 阶段的多 GPU 并行策略。

相关页面：[[../../wiki/concepts/Decode Context Parallel|Decode Context Parallel]]、[[../../wiki/concepts/KV Cache|KV Cache]]、[[../../wiki/concepts/Tensor Parallelism|Tensor Parallelism]]。

## 核心观点

DCP 把**同一个请求的历史 KV Cache 沿 token/context 维分片到多张 GPU**：每张卡只保存并读取一部分历史 KV，分别计算局部 attention，最后跨卡合并成精确的全局 attention 输出。

它主要解决：长上下文 decode 时 KV Cache 占用大，以及 GQA/MQA/MLA 模型的 KV heads 太少、普通 TP 无法继续有效切分 KV Cache，进而导致多卡重复存储的问题。

## 机制拆解

假设 `DCP = 4`，一个请求的历史 token KV 可以交错分布为：

```text
rank 0: token 0, 4, 8, ...
rank 1: token 1, 5, 9, ...
rank 2: token 2, 6, 10, ...
rank 3: token 3, 7, 11, ...
```

每个 rank 用相同的当前 Query 与本地 KV shard 计算：

```text
local_lse_i = logsumexp(Q @ K_i^T)
local_out_i = softmax(Q @ K_i^T) @ V_i
```

由于不同 shard 的 softmax 分母不同，`local_out` 不能直接相加。系统需要利用 local LSE 对各局部输出重新缩放，再通过 collective communication 合并：

```text
global_lse = logsumexp(local_lse_0, ..., local_lse_n)
global_out = Σ exp(local_lse_i - global_lse) * local_out_i
```

## 对比分析

| 技术 | 切分粒度 | 主要目标 |
| --- | --- | --- |
| TP | 单层权重、hidden/head 等张量维度 | 分摊模型权重与层内计算 |
| DCP | 同一请求的历史 context/KV token | 减少 KV Cache 重复和单卡读写量 |
| DP Attention | 请求/batch | 用多个 attention replica 提高多请求吞吐 |
| Flash Decoding | kernel 内的 KV splits | 在短 Query、长 KV 时增加 kernel 并行度 |

DCP **不是 TP 的替代品**。在 vLLM 的现有知识库口径中，它复用既有 TP group，把 attention 的并行布局从单纯 head 维切分扩展为 `head × context` 二维切分。

## DCP 是否需要 DP？有多少张卡？

DCP **不要求**启用 DP。DP 把不同请求分给不同模型副本；DCP 把同一个请求的历史 KV 分给同一 TP group 内的多个 rank。二者可以独立使用，也可以组合。

在 vLLM 复用 TP group 的口径中，DCP 不额外增加进程或 GPU：

```text
粗略总卡数 = DP × PP × TP
DCP 不再额外相乘
```

因此只说 `DCP = 4` 不能唯一确定卡数：

| 配置 | 总卡数 | DCP rank 组织 |
| --- | ---: | --- |
| `TP=4, DCP=4` | 4 | 一个四成员 DCP group |
| `TP=8, DCP=4` | 8 | 逻辑上是 `TP/DCP × DCP = 2 × 4`，即两个 head 分片方向，每个方向有四个 context shards |
| `TP=4, DCP=4, PP=2` | 8 | 两个 pipeline stage 各有一个四成员 TP/DCP group |
| `TP=4, DCP=4, DP=2` | 8 | 两个 DP 副本各有一个四成员 TP/DCP group |

在常见一进程一卡的部署中，同一个物理进程可以同时拥有 global rank、TP rank 和 DCP group 内的 local rank。`DCP=4` 表示组内 local DCP rank 为 `0..3`，并不是另加四张卡。具体 global rank 编号如何映射到这个二维布局依赖实现版本。

### 单 KV head 的 Wide-EP 例子

对于 `DP Attention=8, TP=1, EP=8` 的 8 卡部署：

```text
GPU 0: Attention DP rank 0 + Expert shard 0
GPU 1: Attention DP rank 1 + Expert shard 1
...
GPU 7: Attention DP rank 7 + Expert shard 7
```

不同请求被分配到不同 Attention DP ranks；每个请求的完整单-head/latent KV Cache 留在它所属的一个 rank。进入 MoE 层后，token 才通过 EP group 在 8 张卡之间 dispatch/combine。

此时 `TP=1`，DCP 没有可复用的多成员 TP group，因此只能是 `DCP=1`，即没有 context 分片。单 KV head 并不等于自动启用 DCP；它只是意味着在 TP 拉大后，普通 TP 更容易复制 KV，从而更值得用 DCP。

如果固定 8 张卡并希望 `DCP=4`，一种概念重排是：

```text
Attention DP=2, TP=4, DCP=4, EP=8
```

这时有两个 Attention 请求副本组，每组四张卡；一个请求的 KV 在组内切成四份，而 experts 仍跨全部八张卡分布。它把请求级并行度从 8 降为 2，换来单请求 KV context 的四路分片。具体框架版本是否支持这种 DP/TP/DCP/EP 组合需要实测核对。

### 8 KV heads、TP=8 时能否开 DCP=4？

若这里的 8 heads 指 MHA 的 8 个 **KV heads**：

```text
TP=8, DCP=1:
每 rank = 1 个 KV head × 完整长度 S

假设按二维布局使用 TP=8, DCP=4:
head 并行度 = TP/DCP = 2
每 rank = 4 个 KV heads × 长度 S/4
```

因此每个 rank 确实会计算 4 个 heads，但只持有这些 heads 各自的四分之一 context，并不是 4 个完整 head cache。两种布局的单 rank KV 元素数相同：

```text
1 × S = 4 × S/4
```

所以纯 TP 已经把 8 个 KV heads 完全切开的情况下，DCP=4 不节省 KV Cache，只增加通信。按知识库记录的 vLLM 配置约束，这种情况下实用 DCP 上限为 1，`DCP=4` 通常不是有效或有意义的配置。

如果“8 heads”指 8 个 Q heads，而模型使用 GQA/MQA、KV heads 少于 8，则应按 `num_kv_heads` 而非 Q-head 数重新判断。

### 1 KV head、TP=8、DCP=8、EP=8

同一批 8 张卡可以同时是 TP8、DCP8 和 EP8；这些并行 size 描述不同 process-group 视角，不相乘。每个 rank 的职责是：

```text
TP:  dense 权重/计算的 1/8 shard
EP:  一部分 experts
DCP: 唯一 KV head 的 1/8 context shard
```

每个 rank 的张量形状里仍有一个 KV head，但保存的 token 不同：

```text
rank 0: head 0 的 token 0, 8, 16, ...
rank 1: head 0 的 token 1, 9, 17, ...
...
rank 7: head 0 的 token 7, 15, 23, ...
```

因此不是 8 份完整 head cache，而是一个逻辑 head 被沿 context 切为 8 份：

```text
每 rank: 1 head × S/8
全局合计: 1 head × S
```

相反，`TP=8, DCP=1` 才可能让 8 个 TP ranks 各保存 `1 head × S`，形成完整 KV Cache 的八路复制。DCP8 会让相同 Query 在各 rank 对本地 KV shard 计算 partial attention，再跨 rank 合并 LSE/output。

如果总共只有 8 张卡，这种拓扑通常对应 Attention DP=1；若要求 8 个 Attention DP replicas、且每个 replica 内都是 TP8/DCP8，则需要 64 个 ranks（不考虑 PP），EP group 的具体划分再由框架配置决定。

## 性能权衡

**适合：**

- 长上下文 decode；
- KV Cache 成为显存容量或 HBM 带宽瓶颈；
- GQA、MQA、MLA 等 `num_kv_heads` 较少的模型；
- `TP size` 已超过 KV head 维可有效切分的范围。

**代价：**

- 每层 attention 增加跨 GPU 的 LSE、partial output 或 Query 重排通信；
- 短上下文或 KV Cache 不紧张时，通信开销可能大于收益；
- Prefill 写 cache、Prefix Caching、Chunked Prefill 和 attention backend 都需要兼容分布式 KV 布局。

## 一句话理解

> **TP 是把模型计算切到多卡；DCP 是把同一个长请求的历史 KV Cache 切到多卡。**

## 待核实

vLLM 的具体参数约束、支持的 attention backend，以及 `AllGather/ReduceScatter` 与 `All-to-All` 的精确数据流会随版本变化，部署时应以目标版本的官方文档和源码为准。
