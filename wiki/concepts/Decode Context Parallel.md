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
