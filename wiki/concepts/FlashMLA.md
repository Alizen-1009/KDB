---
type: concept
topic: 注意力机制
sources: 1
updated: 2026-05-17
---

# FlashMLA

## 定义

`FlashMLA` 是 DeepSeek 开源的面向 [[MLA]] decode 推理的高性能 attention kernel / backend，目标是在缓存 latent KV 的前提下，用 IO-aware tiling、paged KV cache 和 Hopper GPU 特性提高长上下文推理吞吐。

## 它解决什么问题

- [[MLA]] 降低了单 token 的 KV cache 存储成本，但 decode 阶段仍要高效读取历史 latent cache 并完成 attention。
- 通用 [[FlashAttention]] 主要服务标准 attention 数据流，不能直接覆盖 MLA 的 latent KV、矩阵吸收和特殊 cache layout。
- 长上下文 serving 中，decode 往往受 [[KV Cache]] HBM 读写、ragged batch 和负载不均影响，需要专门 kernel 配合模型结构落地。

## 核心机制

- 面向 DeepSeek 系列模型的 MLA inference decode，而不是训练反向传播。
- 支持 paged KV cache 与变长序列，通过 metadata、block table、`cache_seqlens`、`num_splits` 等信息调度不规则请求。
- 使用 Split-KV 思路把长 KV 序列拆给多个 SM 或多轮迭代，再用 combine kernel 合并 partial 结果。
- 在 row-wise / block-wise 粒度把中间状态尽量留在 shared memory 或寄存器中，减少额外 HBM 往返。
- 面向 Hopper / SM90 的实现会利用 Tensor Core、GMMA、named barrier、`cp.async` 等硬件能力；具体使用范围需按官方实现核实。

## 和相关技术的关系

- 与 [[MLA]]：`MLA` 是模型 attention 结构，`FlashMLA` 是让 MLA decode 在 GPU 上跑得更好的 kernel/backend。
- 与 [[FlashAttention]]：两者都强调 IO-aware dataflow、tiling 和减少中间矩阵物化；FlashMLA 是 MLA 语境下的特化实现。
- 与 [[PagedAttention]]：二者都关心 paged KV cache 和动态序列管理，但 PagedAttention 更常指 vLLM 的 KV cache 分页抽象与 decode kernel，FlashMLA 则聚焦 DeepSeek MLA kernel。
- 与 [[CUDA Kernel]]：FlashMLA 属于 attention 类高性能自定义 kernel，性能依赖 shape、dtype、layout、硬件架构和调度策略。

## 关键权衡

- 优点：把 MLA 的 KV cache 压缩收益转化为实际 decode throughput，尤其适合长上下文和 DeepSeek 系列模型 serving。
- 代价：实现高度硬件相关，Hopper/SM90 优化、paged cache layout、变长调度和 Split-KV 合并都会增加维护和移植成本。
- 风险：公开文章中的性能数字需要结合官方 benchmark、硬件配置、batch/context 分布和具体 commit 复核。

## 相关实体

- [[../entities/DeepSeek-AI]]

## 相关来源

- [[../sources/陈巍：DeepSeek 开源Day（1）-FlashMLA 深入分析（收录于：DeepSeek技术详解系列）]]

## 相关概念

- [[MLA]]
- [[KV Cache]]
- [[FlashAttention]]
- [[PagedAttention]]
- [[CUDA Kernel]]
- [[Tiling]]
- [[Online Softmax]]

## 研究备注

- 后续应补 DeepSeek 官方 FlashMLA repo 的 README、benchmark 与核心 CUDA 文件，区分来源解读、官方接口和具体版本实现。
- 需要进一步核实 BF16/FP16 支持、CUDA 版本要求、FP8 路线、H800/H100 上的真实性能边界，以及 vLLM / SGLang / TensorRT-LLM 集成方式。
