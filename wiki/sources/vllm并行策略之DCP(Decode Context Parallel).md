# vllm并行策略之DCP(Decode Context Parallel)

## 来源信息

- 标题：vllm并行策略之DCP(Decode Context Parallel)
- 作者：[[../entities/梦初AI Infra]]
- 日期：2026-03-25
- 类型：文章 / vLLM 并行策略解析
- 原始文件：[[../../raw/articles/vllm并行策略之DCP(Decode Context Parallel)|vllm并行策略之DCP(Decode Context Parallel)]]
- 原始链接：[知乎专栏](https://zhuanlan.zhihu.com/p/2020086868914499979)

## 2-3 条核心摘要

- 文章将 `DCP` 定义为面向 decode 阶段的 context parallel：通过把 [[KV Cache]] 沿 `seq_len` 维分片，让单卡 KV cache 显存与读取量近似下降到 `1 / DCP`，但会引入跨 DCP rank 的 softmax state 合并通信。
- 在 vLLM 实现口径中，`--decode-context-parallel-size x` 不增加 TP world size，而是复用已有 [[Tensor Parallelism]] group；`TP` 需要能被 `DCP` 整除，attention 内部形成 `head` 维 `TP / DCP` 与 `seq_len` 维 `DCP` 的二维 KVCache 布局。
- 实现重点是 interleaved KV cache 存储：对单个 request，`token_idx = n` 的 token KV 严格存到 `n % cp_world_size` 对应的 DCP rank。Decode 时各 rank 先基于本地 KV shard 计算 partial output 与 `lse`，再通过跨 rank 通信合并为全局 attention output。

## 值得关注的论断

- 原文把 DCP 理解为“分布式 [[Flash Decoding]]”的近似类比：二者都沿 context/KV 维切分并依赖 online softmax / log-sum-exp 合并，但 vLLM DCP 还要处理 TP group 复用、head 维重排和跨 GPU 通信。
- 原文给出一条经验配置口径：`MHA` 通常直接用 TP，`GQA` 可让 `DCP = TP / num_kv_heads`，`MLA/MQA` 常接近 `DCP = TP`。这应视为该来源的部署经验，而不是无条件规则。
- 原文声称截至其写作时 DCP 在 CUDA backend 已实现，可通过 `--decode-context-parallel-size` 使用，且兼容 [[Chunked Prefill]] 与 [[Prefix Caching]]；其他 backend、PCP 状态、`dcp_all2all` 通信支持均需按具体 vLLM 版本核实。

## 关键概念

- [[Decode Context Parallel]]
- [[KV Cache]]
- [[Tensor Parallelism]]
- [[Flash Decoding]]
- [[Chunked Prefill]]
- [[Prefix Caching]]
- [[MLA]]

## 相关实体

- [[../entities/vLLM]]
- [[../entities/梦初AI Infra]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`Decode Context Parallel`、`KV Cache`、`Tensor Parallelism`、`Flash Decoding`、`Chunked Prefill`、`Prefix Caching`、`MLA`
- 会更新哪些实体页：`vLLM`、`梦初AI Infra`
- 是否存在冲突：无直接冲突；本来源把既有 `DCP` 概念页从“context/token 维分片”进一步补到 vLLM 的 interleaved KV 存储、TP group 复用、decode/prefill 处理和通信路径。

## 待确认

- 原文中的 `25年Q4` 引入时间、CUDA backend 支持范围、PCP 开发状态和 `dcp_all2all` PR 状态需要按 vLLM 官方文档或具体 commit 核实。
- Helix Parallelism 论文中的命名和 group 设计与 vLLM 实现不完全相同；引用图示时应保留“作者改图/类比”的语境。
- DCP 与 Chunked Prefill、Prefix Cache 的兼容性依赖具体 backend、attention kernel、cache manager 和版本，不宜脱离版本写成稳定承诺。
