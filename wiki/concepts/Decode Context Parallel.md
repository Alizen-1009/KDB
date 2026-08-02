---
type: concept
topic: 并行与分布式
sources: 2
updated: 2026-06-12
---

# Decode Context Parallel

## 定义

`Decode Context Parallel`，简称 `DCP`，是长上下文 serving 中专门面向 decode 阶段的上下文并行策略：在已经使用 [[Tensor Parallelism]] 的 GPU 组内，进一步把历史 [[KV Cache]] 沿 context/token 维切分，减少普通 TP 下 KV cache 重复保存，并提升长上下文 decode 的可承载 batch size。

## 它解决什么问题

- Decode 每步只有少量新 `Q` token，却要读取大量历史 `K/V`。
- 普通 TP 可以先沿 `KV head` 维切 KV cache，但 `num_kv_heads` 由模型结构决定，常常很小。
- 当 `TP size` 继续大于可切的 `KV head` 数时，多出来的 TP rank 往往会重复保存同一段 KV cache。
- 长上下文场景里，KV cache 重复会直接吞掉显存，导致 batch size 上不去，吞吐也上不去。

## 核心机制

- 第一层：普通 [[Tensor Parallelism]] 仍负责层内权重/计算切分，也可以沿 `KV head` 维切一部分 KV cache。
- 第二层：`DCP` 在同一批 GPU 内复用 TP group，把每个请求的历史 context 再沿 token 维切成多个 shard。
- 每个 DCP rank 只保存和读取自己负责的 `KV token` shard。
- `vllm并行策略之DCP` 这篇来源补充了 vLLM 的具体存储方式：KV cache 采用 interleaving，单个 request 中 `token_idx = n` 的 token 存到 `n % cp_world_size` 对应的 DCP rank，而不是简单按连续 token range 切块。
- Attention 输出需要像 [[Flash Decoding]] 一样合并各 shard 的局部 softmax 统计，而不是直接相加局部结果。
- 在 vLLM 口径里，`dcp_size` 不增加启动 GPU 数，而是在既有 `tp_size` 内减少 KV cache 重复；官方文档给出的约束口径是 `dcp_size` 落在 `[1, tp_size / H]`，其中 `H` 是模型的 `KV heads` 数量。
- 该来源还给出使用口径：通过 `--decode-context-parallel-size x` 开启，`TP` 需要能被 `DCP` 整除，`TP world_size` 不会因为 DCP 增加；attention 内部的 head 维并行度从 `TP` 变为 `TP / DCP`，同时增加 `seq_len` 维 `DCP` 并行。

## 分布式 Softmax 合并

DCP rank 用本地 KV shard 计算：

```text
scores_i = Q @ K_i^T
local_lse_i = logsumexp(scores_i)
local_out_i = softmax(scores_i) @ V_i
```

`local_out_i` 不能直接相加，因为每个 shard 的 Softmax 分母不同。精确全局输出需要：

```text
global_lse = logsumexp(local_lse_0, ..., local_lse_n)
global_out = Σ exp(local_lse_i - global_lse) * local_out_i
```

具体 backend 可用 AllGather/ReduceScatter 或 All-to-All 交换 Query、LSE、partial output 等张量，但稳定机制是“KV 保持 context 分片、本地 Partial Attention、跨 rank 合并 Softmax 统计”，不应笼统写成每层 AllGather 完整 KV。

## 为什么 TP 不一定够

TP 解决的是“单层权重和矩阵计算怎么分摊”，它很适合降低单卡权重压力和单请求时延；但 decode 长上下文时，瓶颈常常变成 `KV Cache` 容量和带宽。

一个直觉例子：

- 假设模型只有 `H = 2` 个 KV heads。
- 你开 `TP = 8`。
- 沿 KV head 最多只能有效切成 2 份。
- 剩下的 `8 / 2 = 4` 组 rank 可能各自重复保存同样的 KV cache。
- 这时加 `DCP = 4`，就可以把原本重复的 KV cache 改成沿 context 维切 4 份。

所以 DCP 不是比 TP “更高一级所以替代 TP”，而是当 TP 已经拉大、KV head 维切不动时，给 decode 阶段补上的 KV/context 维切分。

## 是否需要 DP，以及 DCP rank 如何组织

- DCP **不要求**启用 Data Parallelism。DP 是请求/副本维并行，DCP 是同一请求的 context/KV 维并行，两者正交，可以单独使用，也可以组合。
- 在 vLLM 的复用 TP group 口径中，`dcp_size` **不乘到物理 GPU 数上**。它只是在已有 `tp_size` 个进程/GPU 中建立 DCP 子组，因此仅知道 `DCP = 4` 不能唯一确定总卡数，还要看 `TP`、`PP` 和可选的 `DP`。
- 常见的一进程一卡语境下，每个物理进程同时有 global rank、TP rank，以及所在 DCP group 内的 local DCP rank；`DCP = 4` 表示每个 DCP group 有 `dcp_rank = 0..3` 四个成员，并不额外创建四个进程。
- 例如 `TP = 4, DCP = 4, PP = 1, DP = 1` 使用 4 张卡，四个 TP rank 同时组成一个四成员 DCP group。
- `TP = 8, DCP = 4, PP = 1, DP = 1` 仍使用 8 张卡；attention 逻辑上形成 `TP / DCP = 2` 个 head/KV-head 分片方向，每个方向内有 4 个 context shards，也可看成 `2 × 4` 的二维布局。具体 global rank 到二维坐标的编号顺序属于实现细节，不应脱离版本假定。
- `TP = 4, DCP = 4, PP = 2` 使用 8 张卡；每个 pipeline stage 各自有一个 4-rank TP/DCP group。
- 若再启用普通 `DP = 2`，则会复制两套上述模型并行拓扑；在不考虑其他并行维度时，物理卡数可粗略看作 `DP × PP × TP`，DCP 不再额外相乘。

配置上通常要求 `TP` 能被 `DCP` 整除，并受模型 KV-head 数量和 backend 支持约束；因此 `DCP = 4` 至少需要一个可容纳 4 个成员的有效 TP group，但“DCP=4 就一定是 4 张卡”只有在 `TP=4、PP=1、DP=1` 时才成立。

### 单 KV head、`DP=8, TP=1, EP=8` 的例子

这是 [[Wide Expert Parallelism]] / DP Attention 的典型逻辑拓扑，而不是 DCP 拓扑：8 张 GPU 同时充当 8 个 Attention DP ranks 和一个 8-rank EP group。每个请求被分配给某一个 Attention DP rank，该 rank 单独保存该请求完整的单-head/latent KV Cache；到 MoE 层时，token 再通过 EP dispatch 去 8 张卡上寻找对应 expert。

因为每个 Attention replica 的 `TP=1`，可复用的 TP group 只有一个成员，所以按 vLLM 的 TP-group 复用口径只能有 `DCP=1`，也就是实际上没有 context 分片。`num_kv_heads=1` 说明“如果把 TP 拉大，DCP 很有价值”，但它不会让 `TP=1` 自动产生多个 DCP ranks。

若固定仍是 8 张卡，并希望 `DCP=4`，一种概念上的重排是 `Attention DP=2, TP=4, DCP=4, EP=8`：两组 Attention replicas 各占 4 个 TP/DCP ranks，每个请求在组内把 KV context 切成 4 份，而 MoE experts 仍可跨全部 8 ranks 做 EP。相比原来的 `DP=8, TP=1, EP=8`，这是用请求级并行宽度从 8 降到 2，换单请求 context 分片、KV 容量与带宽分摊；同时增加 DCP/TP 通信。该组合是否被目标 vLLM 版本、模型与 backend 完整支持，必须用实际配置和源码确认。

### `8 KV heads, TP=8, DCP=4` 的二维布局直觉

这里必须区分 `Q heads` 和 `KV heads`；DCP 是否有价值主要看 `num_kv_heads`。若假设是 MHA，确实有 `8` 个 KV heads：

- 只开 `TP=8` 时，head 维并行度为 8，每个 rank 持有 `1` 个 KV head 的完整 context。
- 若纯按二维公式把它重排为 `TP=8, DCP=4`，head 维并行度变为 `TP / DCP = 2`；每个 rank 的局部 attention 视图会覆盖 `8 / 2 = 4` 个 KV heads，但每个 head 只持有 `1/4` context。
- 因而“每 rank 有 4 个 heads”只说对了一半：准确说法是“每 rank 有 4 个 heads 的局部 context shards”，不是 4 个完整 head cache。
- 单 rank KV 元素量并未下降：`1 × S = 4 × (S/4)`。因此当 `num_kv_heads = TP = 8`、纯 TP 已无 KV 复制时，这种 DCP 重排不会节省 KV Cache，反而增加跨 rank softmax/output 合并通信。

按照本页已有的 vLLM 配置口径，`num_kv_heads=8, TP=8` 时实用 DCP 上限就是 1，因此 `DCP=4` 通常不是有效或有意义的配置。DCP 主要面向 `num_kv_heads < TP` 的 GQA/MQA/MLA；若用户所说的“8 heads”只是 8 个 Q heads，而 KV heads 更少，则必须改用真实 `num_kv_heads` 重新计算。

### `1 KV head, TP=8, DCP=8, EP=8` 是否复制 head

这个配置可逻辑理解为同一批 8 个 ranks 同时组成 TP、DCP 和 EP group；三个 size 不相乘。若没有额外 DP/PP，物理上仍是 8 张卡：

- TP：每个 rank 持有 dense 权重/计算的一个张量分片。
- EP：每个 rank 持有一部分 experts，并参与 token dispatch/combine。
- DCP：8 个 ranks 共同处理同一个请求，把唯一 KV head 的 context 切成 8 份。

每个 rank 的局部张量在 head 维上确实仍标记为 `1 KV head`，但这不等于保存了 8 份完整 cache。它们保存的是同一逻辑 KV head 的不同 token shards：

```text
rank 0: KV head 0 的 token 0, 8, 16, ...
rank 1: KV head 0 的 token 1, 9, 17, ...
...
rank 7: KV head 0 的 token 7, 15, 23, ...
```

所以每 rank KV 量约为 `1 head × S/8`，8 ranks 合起来才是 `1 head × S`。真正发生多 rank 复制的是纯 `TP=8, DCP=1`：因为单 KV head 无法沿 head 维继续切，每个 TP rank 可能保存 `1 head × S` 的完整 cache。启用 `DCP=8` 后，复制的是很小的当前 Query/必要元数据，历史 KV 主体则沿 context 分片；各 rank 对同一 head 算 partial attention，最后合并 LSE/output。

该拓扑通常意味着 Attention 请求级 DP 宽度为 1：一个请求跨 8 ranks 执行。若还要 `DP Attention=8` 且每个副本内部都是 `TP/DCP=8`，则在不考虑 PP 时需要 64 个 ranks；`EP=8` 如何分组还需由具体框架拓扑决定。

## 和相邻概念的关系

- 与 [[Tensor Parallelism]]：TP 主要切权重/hidden/head 相关维度；DCP 主要切 decode 阶段的历史 context/KV token 维度。实际部署中常先增大 TP 到性能满意，再加 DCP 减少 KV 重复。
- 与 [[Flash Decoding]]：Flash Decoding 可理解为单机/单 kernel 内的 Split-KV decode 思路；DCP 是把这种 `KV/context` 维切分扩展到多 GPU/KV cache 分片口径。`vllm并行策略之DCP` 明确把 DCP 类比为“分布式 Flash Decoding”，但 vLLM 实现还要处理 TP group 复用、head 维并行度变化和跨 GPU 通信。
- 与 [[DP Attention]]：DP Attention 更像请求/batch 级 attention replica，目标也是减少某些模型下 TP attention 的 KV 重复；DCP 则是在一个 TP group 内继续沿 context 维切同一请求的 KV。DPA 提升的是多请求吞吐和每个 replica 的 KV 独立性，DCP 补的是单个长上下文请求在 TP group 内的 KV/context 分片。
- 与 [[Sequence Parallelism]]：SP 常用于 TP 配套的激活序列维切分；DCP 专注 serving decode 的历史 KV cache 分片。

## 和 Flash Decoding 的区别

二者共同点是都把 decode attention 的历史 `KV` 沿 context/token 维拆开，各自计算局部 attention，并用 [[Online Softmax]] / log-sum-exp 统计合并精确输出。

关键区别在层级：

| 维度 | Flash Decoding | DCP |
| --- | --- | --- |
| 抽象层级 | attention kernel / 算法思路 | serving 并行与 KV cache 分布策略 |
| 主要目标 | `Q` 很短时增加单次 decode kernel 的 `KV` 维并行度，提高 GPU 利用率 | 在 TP group 内减少 `MLA/MQA/GQA` 等小 `num_kv_heads` 模型的 KV cache 复制，扩大长上下文容量或 batch size |
| KV 所在位置 | 通常是同卡或同一 kernel 视角下的多个 `KV split`，也可扩展到多设备 | KV cache 物理分布在不同 DCP rank；vLLM 来源中是 interleaved 存储 |
| 额外系统约束 | 主要关心 split 合并 kernel、cache layout、batch/context 形态 | 还要关心 `TP / DCP` 关系、process group、prefill 写 cache、prefix cache、跨 rank 通信 |
| 通信/合并 | split 间合并局部 max、sum、output | DCP rank 间交换 `lse` / partial output，并与 TP head 维重排配合 |

一句话：Flash Decoding 更像“decode attention 怎么沿 KV 维多切几刀来跑得满”；DCP 更像“vLLM 在多 GPU serving 里把同一请求的 KV cache 放到不同卡上，顺便借用 Flash Decoding 那套局部 softmax 合并思想”。

## Prefill 阶段影响

DCP 主要优化 decode 阶段，但 prefill 仍需要配合：

- KV cache 写入必须遵守 DCP 的 interleaved 布局，否则 decode 阶段无法按 DCP rank 读取本地 shard。
- 对 [[MLA]]，来源称 prefill 阶段通常是非矩阵吸收的 MHA 形态，主体 attention 可继续用 TP；但在处理已有 context cache 时，需要对 interleaved KV cache 做 all-gather / reorg 后再参与合并。
- 对 `GQA`，来源称 context 部分也可以使用 DCP 式二维切分并在 DCP group 间通信；较短的 query 部分仍可按纯 TP 计算。
- 这些细节和 [[Chunked Prefill]]、[[Prefix Caching]] 的兼容关系均属于实现/版本相关内容，引用时应落到具体 vLLM 版本。

## 面试回答模板

> 只用 TP 不一定够，因为 TP 主要解决权重和层内计算切分。Decode 长上下文时，真正占显存和带宽的是 KV cache。普通 TP 可以先按 KV head 切 cache，但 GQA/MQA/MLA 模型的 KV heads 很少；当 TP size 大于 KV heads 后，部分 rank 会重复保存同样的 KV cache。DCP 就是在既有 TP group 内，再把历史 context/token 维切开，让不同 GPU 保存不同 KV shard，attention 时各自算局部结果，再用 online softmax/log-sum-exp 合并。它不是替代 TP，而是补 TP 在长上下文 decode KV cache 上的短板。

## 相关概念

- [[Tensor Parallelism]]
- [[KV Cache]]
- [[Flash Decoding]]
- [[DP Attention]]
- [[Sequence Parallelism]]
- [[PagedAttention]]
- [[Chunked Prefill]]
- [[Prefix Caching]]
- [[Prefill Context Parallel]]
- [[Online Softmax]]

## 相关来源

- [[../sources/vllm并行策略之DCP(Decode Context Parallel)]]
- [[../sources/vllm PCP 与 DCP 深度解析]]

## 研究备注

- DCP 的收益依赖长上下文、`num_kv_heads`、`tp_size`、batch size、attention backend 和跨 rank 合并开销；短上下文或 KV cache 不紧张时不一定值得开。
- vLLM 官方文档将 DCP 描述为适用于 MLA 和 GQA 模型的 decode context parallel；具体支持矩阵和参数名应随 vLLM 版本核实。
- `vllm并行策略之DCP` 中关于 CUDA backend、PCP 开发状态、`dcp_all2all` 通信和 Chunked Prefill / Prefix Cache 兼容性的描述，应视为来源时点的实现观察，后续需要按官方文档、PR 或 commit 复核。
- 新来源同时出现“DCP 复用 TP group、不增加 world size”和“总 GPU=TP×DCP”的冲突拓扑。本页暂保留已有 vLLM 来源的复用 TP group 口径；group construction、参数约束和后端通信必须绑定版本。
- 新来源对 `ag_rs` 先后给出“完整 KV AllGather”与“local attention + LSE/output merge”两种不同数据流。本页采用后者作为稳定数学机制，精确 collective 张量待源码核实。
