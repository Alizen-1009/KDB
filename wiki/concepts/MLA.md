---
type: concept
topic: 注意力机制
sources: 5
updated: 2026-06-12
---

# MLA

## 定义

`MLA`（Multi-head Latent Attention）是一种 attention 结构，通过低秩 KV 联合压缩把历史 `K/V Cache` 缓存在低维 latent 表示中，并在推理时用矩阵吸收避免显式还原完整 per-head `K/V`。

## 它解决什么问题

- 标准 `MHA` decode attention 需要为每个生成 token 读取完整历史 `K/V cache`，长上下文下容易 memory-bound。
- `GQA/MQA` 能减少 KV heads，但可能牺牲部分表达能力或性能。
- `MLA` 试图在保持多头表达能力的同时，显著降低 KV cache 显存占用和 HBM 读流量。

## 核心机制

- 低秩 KV 联合压缩：`c_t^KV = W_DKV h_t`，推理时缓存低维 `c_t^KV`，而不是完整 `k_t / v_t`。
- KV 还原矩阵：理论上可由 `k_t^C = W_UK c_t^KV`、`v_t^C = W_UV c_t^KV` 得到完整 content key/value。
- decoupled RoPE：把 content 部分和 RoPE 位置部分拆开，缓存 `c_t^KV` 与较小的 `k_t^R`，避免位置相关 RoPE 阻碍矩阵吸收。
- 这也避免了 `MQA/KV 共享` 下直接旋转共享 KV 表示导致 V 被位置信息污染；RoPE 只由额外的小型 positional key 承载。
- `W_UK` 吸收：`(q_i^C)^T W_UK_i c_j^KV = (W_UK_i^T q_i^C)^T c_j^KV`，因此可先把 query 投影到 latent score 空间，再直接和 latent cache 做 dot product。
- `W_UV` 吸收：`sum_j p_j W_UV_i c_j^KV = W_UV_i (sum_j p_j c_j^KV)`，因此可先在 latent cache 上聚合，再把 value up-projection 与 output projection 放到后面或融合实现。

## Decode 路径

```text
1. c_t^Q = W_DQ h_t
2. q_t = W_UQ c_t^Q，并拆成 q_nope / q_rope
3. c_t^KV = W_DKV h_t
4. k_t^R = RoPE(W_KR h_t)
5. cache 写入 c_t^KV 与 k_t^R
6. q_nope_abs = W_UK^T q_nope
7. score = q_nope_abs @ kv_cache + q_rope @ pe_cache
8. p = softmax(score)
9. z = p @ kv_cache
10. output 侧应用 W_UV / W_O
```

## Kernel/backend 视角

- [[FlashMLA]] 是 DeepSeek 开源的 MLA decode kernel/backend 线索，目标是把 latent KV cache、paged KV cache、变长序列和 Hopper GPU 优化结合起来，补足通用 [[FlashAttention]] 不能直接覆盖 MLA 数据流的问题。
- 这说明 `MLA` 的实际收益不仅取决于模型结构是否压缩 KV cache，还取决于 backend 能否高效执行 latent cache 读取、Split-KV、softmax 与 value 聚合。

## 算术强度与 latency

- `MLA` 的核心收益来自减少 decode attention 每步读取的历史 cache：从完整 per-head `K/V` 降为 latent `c^KV` 加小的 RoPE key。
- 它通常不是单纯减少 FLOPs，而是把 HBM bytes 降得更多，使 `FLOPs / byte` 上升。
- 在长上下文、大 batch、KV cache 压力大的 serving 场景中，MLA 可能降低 latency 或提高吞吐；当 HBM 瓶颈被缓解后，瓶颈可能迁移到 compute-bound。

## 和系统并行策略的关系

- `MLA` 是模型结构，目标是减少单 token 的 KV cache 成本。
- [[DP Attention]] 是系统并行策略，目标是在多 GPU serving 中避免 latent KV cache 被普通 TP attention 重复保存。
- [[Decode Context Parallel]] 也是系统并行策略；`vllm并行策略之DCP` 将 MLA decode 视为 DCP 的典型适用场景之一，因为 MLA decode 可呈现接近 MQA 的单 KV head 形态，纯 TP 容易复制 KV cache。
- 这些策略经常在 DeepSeek/MLA serving 中配合出现，但作用层级不同。

## 常见 serving 拓扑

对 DeepSeek 这类 `MLA + MoE` 模型，vLLM 官方单机示例采用 `TP=1, DP=8, EP=8`：Attention 用8个 DP replicas 分别承载请求和独立 latent KV Cache，experts 跨8 ranks 分片。这是高并发吞吐优先的 [[Wide Expert Parallelism]] 路线。

若 dense/attention 权重单卡放不下，可改为 `DP×TP` 混合拓扑，例如固定8卡的 `DP=4, TP=2, EP=8`；Attention 在每个 DP engine 内走 TP2，experts 仍跨 `DP×TP=8` ranks 做 EP。若 MLA 单 KV head 在 TP ranks 间产生 cache 重复，可在支持的 backend 上加入 `DCP=TP`，例如 `DP4/TP2/DCP2/EP8`。

长上下文单请求容量优先时，可进一步减少 DP、增大 TP/DCP；vLLM 官方以 DeepSeek-R1 的 `TP8+DCP8` 为消除8路 MLA KV duplication 的案例。固定8卡可形成从 `DP8/TP1/DCP1` 到 `DP1/TP8/DCP8` 的连续权衡：前者偏请求吞吐，后者偏单请求 KV context 分片。

官方依据：[Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)、[Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)、[Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)。

## 与 PCP/DCP 的边界

MLA 压缩历史 KV 后，纯 TP 沿 KV-head 维的切分空间更有限，因此 [[Decode Context Parallel]] 可沿 token/context 维减少重复 cache。对 Prefill，[[Prefill Context Parallel]] 切的是长 Prompt 的 sequence computation。vLLM `main@1ad5182` 当前 PCP 是 MLA-only 的 partial-Q/full-KV AllGather 路径，Ring Attention 仍是 active-development 方向，未实现 Ulysses；不能用传统 MHA head-shard 公式直接推断 latent layout。

## 关键权衡

- 优点：显著降低 KV cache 占用与读流量，长上下文 decode 更友好。
- 代价：实现复杂度高，需要处理 decoupled RoPE、矩阵吸收、特殊 attention backend 和 kernel 优化。
- 真实性能依赖模型配置、dtype、batch、上下文长度、硬件 ridge point 和具体 backend。

## 相关实体

- [[../entities/DeepSeek-AI]]
- [[../entities/SGLang]]

## 相关来源

- [[../sources/MLA与DP Attention面试整理]]
- [[../sources/DeepSeekV4中RoPE设计解析]]
- [[../sources/陈巍：DeepSeek 开源Day（1）-FlashMLA 深入分析（收录于：DeepSeek技术详解系列）]]
- [[../sources/vllm并行策略之DCP(Decode Context Parallel)]]
- [[../sources/vllm PCP 与 DCP 深度解析]]

## 相关概念

- [[CSA-HCA|CSA/HCA]]
- [[KV Cache]]
- [[Roofline 模型]]
- [[DP Attention]]
- [[Tensor Parallelism]]
- [[FlashAttention]]
- [[FlashMLA]]
- [[Decode Context Parallel]]
- [[Prefill Context Parallel]]

## 研究备注

- DeepSeek-V2 论文是 MLA 的核心来源之一，DeepSeek-V3 代码展示了 naive 与吸收路径的实现差异。
- 后续可以继续补 `FlashMLA`、`CutlassMLA`、`TRTLLM MLA` 等 backend 的 kernel 视角，区分 naive MLA、absorbed MLA 和不同硬件上的 latency 表现。
- DeepSeek V4 RoPE 解析把 MLA 的 decoupled RoPE 作为 CSA/HCA 处理共享 KV 与位置编码冲突的前置背景。
- 陈巍 FlashMLA 解析补充了 MLA 的工程落地视角；其中性能数字、函数签名和具体 kernel 文件名需按官方 repo 版本核实。
