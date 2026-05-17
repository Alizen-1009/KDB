# 知识库操作日志

按时间记录 ingest、query、lint 等操作，帮助 LLM 与人类共同追踪知识库的演化过程。

## [2026-05-17] query | Engram O(1) 查表与静态记忆表理解

- 读取实体页：`wiki/entities/Engram.md`
- 读取概念页：`wiki/concepts/Conditional Memory.md`
- 更新实体页：`wiki/entities/Engram.md`
- 更新概念页：`wiki/concepts/Conditional Memory.md`
- 本次 query 将 Engram 的大规模静态记忆表补充为“以压缩后局部 `N-gram` 为 key 的超大 embedding table”理解：表项是可训练参数，训练时只更新命中的 rows；推理时通过固定 hash 计算 memory slot 并数组索引访问，因此 lookup 是 `O(1)`。同时补充它与普通 token embedding、RAG 和 KV cache 的边界。

## [2026-05-17] ingest | 陈巍：DeepSeek 开源Day（1）-FlashMLA 深入分析（收录于：DeepSeek技术详解系列）

- 读取原始资料：`raw/articles/陈巍：DeepSeek 开源Day（1）-FlashMLA 深入分析（收录于：DeepSeek技术详解系列）.md`
- 创建来源页：`wiki/sources/陈巍：DeepSeek 开源Day（1）-FlashMLA 深入分析（收录于：DeepSeek技术详解系列）.md`
- 创建概念页：`wiki/concepts/FlashMLA.md`
- 创建实体页：`wiki/entities/陈巍.md`
- 更新概念页：`wiki/concepts/MLA.md`、`wiki/concepts/KV Cache.md`、`wiki/concepts/FlashAttention.md`、`wiki/concepts/CUDA Kernel.md`
- 更新实体页：`wiki/entities/DeepSeek-AI.md`
- 本次 ingest 将 FlashMLA 整理为 DeepSeek MLA decode backend / kernel 线索：它把 latent KV cache、paged KV cache、变长序列、Split-KV 和 Hopper/SM90 kernel 优化结合起来，使 MLA 的 KV 压缩收益能在长上下文 serving 中落地。原文中的 `3000 GB/s`、`580 TFLOPS`、具体文件名和函数签名均标记为待按官方 repo 与 benchmark 核实。

## [2026-05-15] query | DP Attention 具体执行逻辑

- 读取/参考：`wiki/concepts/MLA.md`、`wiki/concepts/DP Attention.md`、`wiki/concepts/Tensor Parallelism.md`
- 参考官方文档：SGLang `DP, DPA and SGLang DP Router`
- 本次 query 将 `DP Attention` 解释为 attention component 级别的 data parallel：普通 TP 在 MLA 模型上容易让每个 TP rank 都保存同一请求的 latent KV cache；DPA 则把请求/batch 分给不同 DP attention replica，每个 replica 独立执行 prefill/decode attention 并维护自己的 KV cache，避免跨 TP rank 复制同一份 KV。对于 DeepSeek MoE 类模型，常见组合是 attention 用 DPA，MoE experts 用 EP，服务层再用 cache-aware router 做请求分发。

## [2026-05-15] query | MLA 是否等同 MQA 与 TP/DPA 关系

- 读取概念页：`wiki/concepts/MLA.md`
- 读取概念页：`wiki/concepts/Tensor Parallelism.md`
- 读取概念页：`wiki/concepts/DP Attention.md`
- 读取来源页：`wiki/sources/MLA与DP Attention面试整理.md`
- 本次 query 澄清：`MLA` 的 latent KV cache 看起来像共享 KV，但不等同于 `MQA`。`MQA` 是多个 query heads 共享同一个或少数几个实际 K/V head；`MLA` 是缓存共享的低维 `c^KV`，每个 query head 仍通过各自的 `W_UK_i / W_UV_i` 或吸收后的投影获得不同的 attention score/value 路径。多 GPU serving 中，普通 TP 可以切 Q/O/MLP 等权重，但 attention 侧如果每个 TP rank 都复制 latent KV cache 会浪费显存；因此 DeepSeek/MLA 类模型常结合 `DP Attention`，让 attention/KV cache 按 DP replica 管理，同时权重侧再与 TP/EP 等策略组合。

## [2026-05-15] query | MLA 公式与推理加速机制

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/MLA.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取概念页：`wiki/concepts/Roofline 模型.md`
- 本次 query 将 `MLA` 解释为“低维 latent KV cache + decoupled RoPE + 矩阵吸收”的组合：训练/数学上可由 latent `c^KV` 还原 content K/V，但推理中不显式缓存完整 per-head K/V，而是把 `W_UK` 吸收到 query 侧、把 `W_UV` 延后到 value 聚合之后，从而把 decode attention 的 HBM 读取从完整 K/V cache 降到 latent cache 加小型 RoPE key。该优化主要减少 memory traffic，提高 arithmetic intensity；真实性能取决于 kernel 是否高效复用 latent cache、batch/context/hardware/backend 等因素。

## [2026-05-14] ingest | vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现

- 读取原始资料：`raw/articles/vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现.md`
- 创建来源页：`wiki/sources/vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现.md`
- 创建实体页：`wiki/entities/方佳瑞.md`
- 更新概念页：`wiki/concepts/PagedAttention.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 更新概念页：`wiki/concepts/FlashAttention.md`
- 更新实体页：`wiki/entities/vLLM.md`
- 本次 ingest 将该文中的 `PAv1` CUDA 并行算法视角整理入库：一个 thread block 负责一个 `sequence/head` 输出行，warp/thread group 遍历 paged KV cache 并完成 `QK -> softmax -> PV`；同时补充 K/V cache layout 差异、与 FlashAttention/FlashDecoding 的任务划分差异，以及 PAv1 长序列并行度、MQA/GQA KV 复用等版本相关待核实点。

## [2026-05-14] query | PagedAttention CUDA 计算流程

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/PagedAttention.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取来源页：`wiki/sources/PageAttention代码走读.md`
- 读取原始资料：`raw/articles/PageAttention代码走读.md`
- 读取原始资料：`raw/articles/vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现.md`
- 本次 query 将两篇文章合并为一条 decode kernel 流程：一个 CUDA thread block 负责一个 sequence/head 的输出行，先通过 `block_tables` 把逻辑 KV block 映射到 physical block，遍历历史 token 计算 `q · k` 并把 logits 放入 shared memory；随后在 block 内做 max/sum reduction 和 softmax；最后按 V cache layout 读取 value 并完成 `softmax(scores) @ V`，再跨 warp 合并写回 `out`。关键区分是：`CUDA thread block` 不是 `KV cache block`，`K cache` 与 `V cache` layout 不同是为了适配 QK 与 PV 两段不同的访存/并行模式。

## [2026-05-12] query | PagedAttention K/V cache layout

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/PagedAttention.md`
- 读取原始资料：`raw/articles/PageAttention代码走读.md`
- 本次 query 聚焦 vLLM/PagedAttention 中 K cache 与 V cache 的物理布局差异：K cache 常见形状为 `[num_blocks, num_kv_heads, head_size / x, block_size, x]`，把 head_dim 按向量化宽度 `x` 分组；V cache 常见形状为 `[num_blocks, num_kv_heads, head_size, block_size]`，相当于在 block 内按 head_dim 行、token 列转置，以适配 `softmax(scores) @ V` 阶段沿 token 维的连续向量化读取。

## [2026-05-12] query | DDP 分布式训练面试考点

- 读取概念页：`wiki/concepts/数据并行.md`
- 读取概念页：`wiki/concepts/Torch Distributed.md`
- 读取概念页：`wiki/concepts/集合通信.md`
- 创建概念页：`wiki/concepts/DDP.md`
- 更新概念页：`wiki/concepts/数据并行.md`
- 更新概念页：`wiki/concepts/Torch Distributed.md`
- 更新索引页：`wiki/index.md`
- 本次 query 将 DDP 面试点整理为：`torchrun/init_process_group` 启动链路、`rank/local_rank/world_size`、`DistributedSampler`、全局 batch、autograd hook、bucket all-reduce 与通信计算重叠、`no_sync` 梯度累积、`find_unused_parameters/static_graph`、SyncBatchNorm、NCCL hang 排查和性能 profiling。

## [2026-05-12] query | 单卡 MoE 瓶颈与 grouped matmul

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/MoE.md`
- 更新概念页：`wiki/concepts/MoE.md`
- 本次 query 补充单卡 MoE 视角：去掉跨卡 all-to-all 后，主要瓶颈转向 token dispatch/permutation、不规则 expert batch、小 GEMM launch overhead、Tensor Core 利用率和 expert 间 tail effect；`grouped matmul` 与 `batched matmul` 都位于 expert FFN 阶段，高性能 MoE 通常更适合用 grouped GEMM 处理不同 expert 的变长 `n_e`，batched matmul 更适合 padding 到固定 capacity 的实现。

## [2026-05-12] query | MoE 算子计算流程

- 读取索引页：`wiki/index.md`
- 读取既有报告：`output/reports/字节二面高压题拆解.md`
- 读取既有报告：`output/reports/面试经验.md`
- 读取既有报告：`output/reports/多卡与推理系统面试梳理.md`
- 读取原始资料：`raw/articles/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构.md`
- 创建概念页：`wiki/concepts/MoE.md`
- 更新概念页：`wiki/concepts/CUDA Kernel.md`
- 本次 query 将 MoE 的执行路径整理为 `router projection -> softmax -> top-k routing -> capacity/load balancing -> token dispatch -> expert FFN -> weighted gather -> residual`，并补充了算子视角下的性能瓶颈：dispatch/gather、跨卡 all-to-all、expert batch 碎片化、负载不均和显存峰值。

## [2026-05-11] ingest | DeepSeekV4中RoPE设计解析

- 读取原始资料：`raw/articles/DeepSeekV4中RoPE设计解析.md`
- 创建来源页：`wiki/sources/DeepSeekV4中RoPE设计解析.md`
- 创建概念页：`wiki/concepts/CSA-HCA.md`
- 创建实体页：`wiki/entities/DeepSeek V4.md`
- 更新概念页：`wiki/concepts/RoPE.md`
- 更新概念页：`wiki/concepts/MLA.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 更新实体页：`wiki/entities/DeepSeek-AI.md`
- 更新实体页：`wiki/entities/kaiyuan.md`
- 本次 ingest 将 DeepSeek V4 解析中的压缩 attention RoPE 设计整理为可复用条目：MLA 的 decoupled RoPE 用于解释共享 KV 下避免 V 被位置污染；CSA/HCA 中压缩 KV 倾向压缩后按块标定位置再旋转，HCA 示例采用 `128 * t` 起始位置；若 V 路径被旋转，输出 O 需要逆旋转以减少绝对位置项。`DeepSeek V4`、`CSA/HCA` 与 `C128A` 具体命名和实现细节已标记为待按源码核实。

## [2026-05-11] ingest | 彻底搞懂RoPE计算原理：从1D到3D

- 读取原始资料：`raw/articles/彻底搞懂RoPE计算原理：从1D到3D.md`
- 创建来源页：`wiki/sources/彻底搞懂RoPE计算原理：从1D到3D.md`
- 创建概念页：`wiki/concepts/M-RoPE.md`
- 创建实体页：`wiki/entities/kaiyuan.md`
- 创建实体页：`wiki/entities/Qwen VL.md`
- 更新概念页：`wiki/concepts/RoPE.md`
- 本次 ingest 将 RoPE 的二维旋转直觉、多维旋转平面拆分、`rotate_half` 工程配对、视觉 2D RoPE、Qwen VL 系列 M-RoPE 与 Interleaved-MRoPE 整理为可链接知识条目。未发现与现有 `RoPE` / `Dual RoPE` 页面直接冲突；原文代码块存在少量排版损坏，已标记为后续需按源码核实。

## [2026-05-11] query | CUDA 算子分类与设计最佳实践

- 读取概念页：`wiki/concepts/CUDA Kernel.md`
- 读取来源页：`wiki/sources/CUDA优化维度框架.md`
- 读取来源页：`wiki/sources/你一定要知道：CUDA优化六要.md`
- 更新概念页：`wiki/concepts/CUDA Kernel.md`
- 本次 query 将常见手写算子按数据依赖和性能形态整理为 `pointwise / reduction / scan / GEMM / attention / gather-scatter / stencil / sort-topk / communication-adjacent`，并补入通用设计流程：先做 shape 与数据流账本，再用 roofline 判断瓶颈，随后决定并行粒度、数据驻留、tiling、归约模板、资源占用、融合边界、profiling 与 autotune。

## [2026-05-11] query | TP forward 通信量估算

- 读取概念页：`wiki/concepts/Tensor Parallelism.md`
- 读取概念页：`wiki/concepts/集合通信.md`
- 更新概念页：`wiki/concepts/Tensor Parallelism.md`
- 本次 query 将 Megatron-style TP 推理 forward 的通信量整理为面试公式：常见 Transformer layer 每层约两次 activation all-reduce，单次逻辑 payload 为 `M * H * dtype_bytes`；ring all-reduce 每 rank 单次发送通信量约 `2 * (P - 1) / P * M * H * dtype_bytes`，每层约 `4 * (P - 1) / P * M * H * dtype_bytes`，整模型再乘层数。Decode payload 小但同步高频，prefill payload 大但更容易被计算覆盖。

## [2026-05-11] query | 算子融合是否越大越好

- 读取概念页：`wiki/concepts/算子融合.md`
- 读取概念页：`wiki/concepts/CUDA Kernel.md`
- 读取概念页：`wiki/concepts/Tiling.md`
- 读取概念页：`wiki/concepts/Occupancy.md`
- 更新概念页：`wiki/concepts/算子融合.md`
- 本次 query 将面试追问整理为“融合边界”口径：算子融合的收益主要来自减少 launch 和中间张量 HBM 往返，但过度融合可能带来寄存器/shared memory 压力、occupancy 下降、register spilling、warp divergence、编译调试复杂度，以及不同阶段共用次优 tile/线程布局的问题；应先做数据流和 roofline 账本，再决定融合范围。

## [2026-05-11] query | PD 分离下 Chunked Prefill 是否仍有价值

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/PD分离.md`
- 读取概念页：`wiki/concepts/vLLM V1 统一调度器.md`
- 读取概念页：`wiki/concepts/Continuous Batching.md`
- 参考官方文档：vLLM `Optimization and Tuning` 与 `Disaggregated Prefilling`
- 创建概念页：`wiki/concepts/Chunked Prefill.md`
- 更新概念页：`wiki/concepts/PD分离.md`
- 更新概念页：`wiki/concepts/Continuous Batching.md`
- 更新概念页：`wiki/concepts/vLLM V1 统一调度器.md`
- 本次 query 将面试追问整理为层级区分：PD 分离是资源池/架构隔离，Chunked Prefill 是 prefill 执行粒度和调度粒度控制。严格 PD 分离会削弱 chunked prefill 对 decode ITL 的直接保护作用，但不会使其失效；prefill pool 内部的 P99 TTFT、公平性、峰值显存、KV 传输流水化，以及条件路由下的 mixed path 仍可能需要 chunked prefill。

## [2026-05-11] ingest | MLA与DP Attention面试整理

- 创建原始资料：`raw/articles/MLA与DP Attention面试整理.md`
- 创建来源页：`wiki/sources/MLA与DP Attention面试整理.md`
- 创建概念页：`wiki/concepts/MLA.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 更新概念页：`wiki/concepts/Roofline 模型.md`
- 更新概念页：`wiki/concepts/DP Attention.md`
- 更新实体页：`wiki/entities/DeepSeek-AI.md`
- 更新实体页：`wiki/entities/SGLang.md`
- 更新索引页：`wiki/index.md`
- 本次 ingest 将对话中的 `MHA/GQA/MLA` 算术强度、MLA 计算流程、decoupled RoPE、`W_UK/W_UV` 矩阵吸收、latency 瓶颈迁移和 `DP Attention` serving 策略整理成面试专题笔记；未发现与现有 wiki 的直接冲突，真实性能边界标记为需结合具体模型、硬件和 attention backend profiling。

## [2026-05-11] query | DP Attention 与 MLA serving

- 读取索引页：`wiki/index.md`
- 参考 SGLang 官方文档：`DP, DPA and SGLang DP Router`
- 创建概念页：`wiki/concepts/DP Attention.md`
- 更新概念页：`wiki/concepts/Tensor Parallelism.md`
- 更新实体页：`wiki/entities/SGLang.md`
- 本次 query 将 `DP Attention / DPA` 整理为系统并行策略：它不改变 attention 数学，而是在多 GPU serving 中重新组织 attention/KV cache 的分布，缓解 DeepSeek/MLA 类模型在普通 TP attention 下 latent KV cache 重复、batch size 受限的问题。它与 MLA 的关系是“模型结构压缩 KV cache + 系统策略避免并行复制浪费”。

## [2026-05-11] query | MLA 与 MHA decode 算术强度

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取概念页：`wiki/concepts/Roofline 模型.md`
- 读取概念页：`wiki/concepts/FlashAttention.md`
- 更新概念页：`wiki/concepts/Roofline 模型.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 本次 query 将面试问题整理为 roofline 口径：标准 `MHA` decode attention 粗略约 `1 FLOP/byte`，通常 memory-bound；`GQA/MQA` 通过 KV 复用提高 intensity；`MLA` 通过压缩 KV cache 显著减少 HBM 读流量，在实现能高效复用 latent cache 时可能转向 compute-bound。该结论为面试级估算，真实边界需结合具体模型参数、dtype、硬件 ridge point 和 kernel 实现 profiling。

## [2026-05-08] ingest | PageAttention代码走读

- 读取原始资料：`raw/articles/PageAttention代码走读.md`
- 创建来源页：`wiki/sources/PageAttention代码走读.md`
- 更新概念页：`wiki/concepts/PagedAttention.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 更新实体页：`wiki/entities/vLLM.md`
- 更新索引页：`wiki/index.md`
- 本次 ingest 将 `PageAttention代码走读` 原文与对话校正整理为可复用笔记：补入 prefill/decode 路径差异、`logical block / physical block / block table / slot_mapping`、decode kernel 中 `query` 与 `num_generation_tokens` 的含义、softmax 维度、K/V cache layout 差异、warp/thread group 分工和 `Copy-on-Write`。原文标题使用 `PageAttention`，wiki 概念页统一使用 `PagedAttention`；版本相关函数名和 kernel 参数标记为待按具体 commit 核实

## [2026-05-07] ingest | SGLang 与 vLLM 区别截图整理

- 创建原始资料：`raw/articles/SGLang 与 vLLM 区别截图整理.md`
- 创建来源页：`wiki/sources/SGLang 与 vLLM 区别截图整理.md`
- 创建概念页：`wiki/concepts/SGLang 与 vLLM 对比.md`
- 更新实体页：`wiki/entities/SGLang.md`
- 更新实体页：`wiki/entities/vLLM.md`
- 更新概念页：`wiki/concepts/LLM Programs.md`
- 更新概念页：`wiki/concepts/RadixAttention.md`
- 更新概念页：`wiki/concepts/PagedAttention.md`
- 更新索引页：`wiki/index.md`
- 本次 ingest 将用户提供的 `SGLang vs vLLM` 截图整理为可复用对比口径：图片大方向正确，但需避免把二者简化成“vLLM=简单问答、SGLang=Agent”。补充边界包括：`vLLM` 也支持多类复杂 serving 能力，`SGLang` 也可作为 production-level serving runtime；`RadixAttention` 和 structured output 的收益均依赖任务结构、缓存命中和约束形式

## [2026-05-07] ingest | SGLang：LLM推理引擎发展新方向

- 读取原始资料：`raw/articles/SGLang：LLM推理引擎发展新方向.md`
- 创建来源页：`wiki/sources/SGLang：LLM推理引擎发展新方向.md`
- 创建概念页：`wiki/concepts/LLM Programs.md`
- 创建概念页：`wiki/concepts/RadixAttention.md`
- 创建概念页：`wiki/concepts/Constrained Decoding.md`
- 更新实体页：`wiki/entities/SGLang.md`
- 更新实体页：`wiki/entities/vLLM.md`
- 更新实体页：`wiki/entities/TensorRT-LLM.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 更新概念页：`wiki/concepts/Prefix Caching.md`
- 更新概念页：`wiki/concepts/Speculative Decoding.md`
- 更新索引页：`wiki/index.md`
- 未发现与现有 wiki 的直接冲突；本次主要把 `SGLang` 从普通 serving engine 扩展到 `LLM Programs runtime` 视角，并补入 `RadixAttention`、压缩 FSM constrained decoding 与 API speculative execution。`SGLang V2` 性能接近或超过 `TensorRT-LLM` 的说法保留为来源声称，待官方 benchmark 或本地复现实验核实

## [2026-05-07] ingest | Gemma 4：Drafter 详解

- 复制原始图片：`raw/images/Gemma 4 Drafter 详解/`
- 创建原始资料：`raw/articles/Gemma 4：Drafter 详解.md`
- 创建来源页：`wiki/sources/Gemma 4：Drafter 详解.md`
- 创建概念页：`wiki/concepts/MTP Drafter.md`
- 更新实体页：`wiki/entities/Gemma 4.md`
- 更新概念页：`wiki/concepts/Speculative Decoding.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 更新概念页：`wiki/concepts/Shared KV Cache.md`
- 更新概念页：`wiki/concepts/Per-Layer Embeddings.md`
- 本次 ingest 将小红书 19 张截图整理为 raw article，并把 Gemma 4 `MTP Drafter` 作为独立概念入库；核对备注指出：`最高 3x`、target activation 复用、KV cache 共享和 E2B/E4B efficient embedder 与 Google 官方文章一致，但具体数值规格仍需按模型配置核实

## [2026-05-07] ingest | vLLM v0 与 vLLM v1 调度架构差异截图整理

- 创建原始资料：`raw/articles/vLLM v0 与 vLLM v1 调度架构差异截图整理.md`
- 创建来源页：`wiki/sources/vLLM v0 与 vLLM v1 调度架构差异截图整理.md`
- 创建概念页：`wiki/concepts/vLLM V1 统一调度器.md`
- 更新实体页：`wiki/entities/vLLM.md`
- 更新概念页：`wiki/concepts/Continuous Batching.md`
- 更新概念页：`wiki/concepts/PagedAttention.md`
- 本次 ingest 将用户提供的三张截图转写为 raw article，并把 `vLLM v0/v1` 调度差异整理成可复用概念页；核对备注指出：v0 开启 chunked prefill 后也可混合 prefill/decode，v1 的 `token quota` 应理解为每步动态 token 分配，且多 GPU 能力不能简化为只支持数据并行

## [2026-05-07] query | 量化剪枝推理瓶颈Nsight与异构集群面试整理

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Roofline 模型.md`
- 读取概念页：`wiki/concepts/PD分离.md`
- 读取既有报告：`output/reports/算子与GPU优化、推理优化补充.md`
- 读取既有报告：`output/reports/多卡GPU监控与SM执行模型面试整理.md`
- 参考外部资料：`GPTQ / AWQ / SmoothQuant / LLM.int8 / Marlin` 论文，NVIDIA Hopper/A100 官方资料，以及 H20 公开规格报道
- 创建查询报告：`output/reports/量化剪枝推理瓶颈Nsight与异构集群面试整理.md`
- 创建来源页：`wiki/sources/量化剪枝推理瓶颈Nsight与异构集群面试整理.md`
- 本次输出围绕量化、GPTQ/AWQ、剪枝/稀疏/蒸馏、推理瓶颈 roofline、Nsight 多卡分析、Marlin、PD 分离以及 A100/H20 异构集群差异形成一份面试专题稿；H20 规格和实习 PD 分离细节标记为待用户按真实环境核实
- 追加补充：数据预加载与 Paddle Fluid Dataset 抽象、Agent 沙箱秒级启动设计、推理显存优化优先级、TTFT 优化和投机解码；`Fluid` 与沙箱 GPU 隔离细节标记为待按实际面试上下文核实

## [2026-05-07] ingest | 多卡GPU监控与SM执行模型面试整理

- 整理查询报告：`output/reports/多卡GPU监控与SM执行模型面试整理.md`
- 创建来源页：`wiki/sources/多卡GPU监控与SM执行模型面试整理.md`
- 更新概念页：`wiki/concepts/Profiling.md`
- 更新概念页：`wiki/concepts/GPU执行模型.md`
- 更新概念页：`wiki/concepts/Occupancy.md`
- 更新概念页：`wiki/concepts/Warp Divergence.md`
- 更新概念页：`wiki/concepts/混合精度训练与推理.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 更新概念页：`wiki/concepts/Continuous Batching.md`
- 本次 ingest 将连续两轮面试问答整理为可长期保存的 Obsidian 报告，并建立到 GPU profiling、SM/warp 执行模型、混合精度推理和 KV cache 的交叉引用；未发现与现有 wiki 的直接冲突

## [2026-05-07] query | SM、线程束与 LLM 推理指标关系

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/GPU执行模型.md`
- 读取概念页：`wiki/concepts/Occupancy.md`
- 读取概念页：`wiki/concepts/Warp Divergence.md`
- 读取概念页：`wiki/concepts/CUDA内存层次.md`
- 读取概念页：`wiki/concepts/FlashAttention.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 更新概念页：`wiki/concepts/GPU执行模型.md`
- 更新概念页：`wiki/concepts/Occupancy.md`
- 更新概念页：`wiki/concepts/Warp Divergence.md`
- 本次 query 围绕 `SM / warp / occupancy / SM Active / SM Issue` 的区别，以及它们如何映射到 LLM prefill、decode、GEMM、FlashAttention、KV cache 和变长序列场景展开

## [2026-05-07] query | 多卡集群 GPU 指标监测与 LLM 推理退化诊断

- 读取索引页：`wiki/index.md`
- 读取日志页：`wiki/log.md`
- 读取概念页：`wiki/concepts/Profiling.md`
- 读取概念页：`wiki/concepts/GPU执行模型.md`
- 读取概念页：`wiki/concepts/混合精度训练与推理.md`
- 读取概念页：`wiki/concepts/Roofline 模型.md`
- 读取既有报告：`output/reports/算子与GPU优化、推理优化补充.md`
- 更新概念页：`wiki/concepts/Profiling.md`
- 本次 query 围绕多卡 GPU 指标监控、`nvidia-smi` 显存口径、`Tensor Active / FP32 pipe active / BF16 pipe active` 的解释，以及 LLM 推理是否因 dtype、shape、layout 不匹配退化到低效 kernel 路径展开

## [2026-05-06] ingest | CUDA内存层次与动态共享内存问答整理

- 创建原始资料：`raw/articles/CUDA内存层次与动态共享内存问答整理.md`
- 创建来源页：`wiki/sources/CUDA内存层次与动态共享内存问答整理.md`
- 创建概念页：`wiki/concepts/CUDA内存层次.md`
- 创建概念页：`wiki/concepts/动态共享内存.md`
- 更新概念页：`wiki/concepts/CUDA Kernel.md`
- 更新概念页：`wiki/concepts/GPU执行模型.md`
- 更新概念页：`wiki/concepts/Occupancy.md`
- 更新概念页：`wiki/concepts/Bank Conflict.md`
- 更新概念页：`wiki/concepts/内存合并访问.md`
- 更新索引页：`wiki/index.md`
- 未发现与现有 wiki 的直接冲突；本次主要把 CUDA kernel launch 第三个参数、`extern __shared__`、静态/动态 shared memory、CUDA memory hierarchy、SM 示意图边界和 occupancy 直觉整理为可追溯页面

## [2026-05-06] ingest | Attention Residuals

- 读取原始资料：`raw/papers/2603.15031v1.pdf`
- 参考已有相关来源页：`wiki/sources/Kimi新作《Attention Residuals》：对Transformer中残差结构的调整.md`
- 创建来源页：`wiki/sources/Attention Residuals.md`
- 更新概念页：`wiki/concepts/Attention Residuals.md`
- 更新概念页：`wiki/concepts/PreNorm Dilution.md`
- 更新概念页：`wiki/concepts/流水线并行.md`
- 更新概念页：`wiki/concepts/Scaling Laws.md`
- 更新实体页：`wiki/entities/Moonshot AI.md`
- 更新索引页：`wiki/index.md`
- 未发现与现有 wiki 的直接冲突；本次主要把 `Attention Residuals` 论文本体作为一手来源单独入库，并补入 depth-wise linear/softmax attention 视角、pseudo-query 零初始化、RMSNorm key、cross-stage caching、two-phase computation 与 scaling-law 实验边界

## [2026-04-30] ingest | LLM提速利器：投机推理的原理与常见方案

- 读取原始资料：`raw/articles/LLM提速利器：投机推理的原理与常见方案.md`
- 创建来源页：`wiki/sources/LLM提速利器：投机推理的原理与常见方案.md`
- 更新概念页：`wiki/concepts/Speculative Decoding.md`
- 更新实体页：`wiki/entities/vLLM.md`
- 未发现与现有 wiki 的直接冲突；本次主要把 `Speculative Decoding` 从“draft-target 基本定义”扩展到“方案分类 / 框架代价 / vLLM 使用面”视角

## [2026-04-29] ingest | 推理的非确定性运算及vLLMSGLang控制方式

- 读取原始资料：`raw/articles/推理的非确定性运算及vLLMSGLang控制方式.md`
- 创建来源页：`wiki/sources/推理的非确定性运算及vLLMSGLang控制方式.md`
- 创建概念页：`wiki/concepts/确定性推理.md`
- 更新概念页：`wiki/concepts/Continuous Batching.md`
- 更新实体页：`wiki/entities/vLLM.md`
- 更新实体页：`wiki/entities/SGLang.md`
- 未发现与现有 wiki 的直接冲突；本次主要补入了“推理系统为何即使固定 seed 也可能不复现”以及 `vLLM / SGLang` 在确定性推理上的工程控制方式

## [2026-04-28] ingest | CUDA优化维度框架

- 读取原始资料：`raw/articles/CUDA优化维度框架.md`
- 创建来源页：`wiki/sources/CUDA优化维度框架.md`
- 更新概念页：`wiki/concepts/内存合并访问.md`
- 更新概念页：`wiki/concepts/Bank Conflict.md`
- 更新概念页：`wiki/concepts/Occupancy.md`
- 更新概念页：`wiki/concepts/Tiling.md`
- 更新概念页：`wiki/concepts/Warp Divergence.md`
- 更新概念页：`wiki/concepts/Tail Effect.md`
- 更新概念页：`wiki/concepts/CUDA Kernel.md`
- 更新概念页：`wiki/concepts/GPU执行模型.md`
- 更新概念页：`wiki/concepts/Profiling.md`
- 未发现与现有 wiki 的直接冲突；本次主要把既有 CUDA 条目从“性能原则”进一步推进到“诊断规则 / 数值估算 / profiler 信号”视角

## [2026-04-28] query | 算子与GPU优化、推理优化补充

- 读取索引页：`wiki/index.md`
- 读取日志页：`wiki/log.md`
- 读取概念页：`wiki/concepts/CUDA Kernel.md`
- 读取概念页：`wiki/concepts/算子融合.md`
- 读取概念页：`wiki/concepts/Profiling.md`
- 读取概念页：`wiki/concepts/FlashAttention.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取概念页：`wiki/concepts/Occupancy.md`
- 读取概念页：`wiki/concepts/Roofline 模型.md`
- 读取概念页：`wiki/concepts/Tiling.md`
- 读取概念页：`wiki/concepts/PagedAttention.md`
- 读取来源页：`wiki/sources/斯坦福CS336 Lecture 5 - GPUs.md`
- 读取来源页：`wiki/sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing.md`
- 读取来源页：`wiki/sources/你一定要知道：CUDA优化六要.md`
- 读取来源页：`wiki/sources/LLM推理优化核心技术.md`
- 生成查询报告：`output/reports/算子与GPU优化、推理优化补充.md`
- 创建概念页：`wiki/concepts/混合精度训练与推理.md`
- 更新概念页：`wiki/concepts/Profiling.md`
- 更新概念页：`wiki/concepts/CUDA Kernel.md`
- 更新概念页：`wiki/concepts/算子融合.md`
- 更新概念页：`wiki/concepts/FlashAttention.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 参考外部资料：`NVIDIA Mixed Precision / Nsight Systems / Nsight Compute 官方文档` 与 `FlashAttention` 论文链接
- 本次输出围绕 `特定 shape 调优 / 算子融合 / 混合精度 / Nsight 工具链 / KV Cache / FlashAttention` 形成了一份可面试复述、也可回链到 wiki 的专题报告；未发现与现有 wiki 的直接冲突

## [2026-04-23] ingest | Model Runner V2 A Modular and Faster Core for vLLM

- 读取原始资料：`raw/articles/Model Runner V2 A Modular and Faster Core for vLLM.md`
- 创建来源页：`wiki/sources/Model Runner V2 A Modular and Faster Core for vLLM.md`
- 创建概念页：`wiki/concepts/持久批处理.md`
- 创建实体页：`wiki/entities/vLLM Team.md`
- 更新实体页：`wiki/entities/vLLM.md`
- 更新概念页：`wiki/concepts/Continuous Batching.md`
- 未发现与现有 wiki 的直接冲突；本次主要把 `vLLM` 条目从 `PagedAttention / Continuous Batching` 扩展到执行核心重构视角，补入了 `MRV2`、`GPU-native input preparation` 和 `async-first` 这条主线

## [2026-04-23] query | 推理系统专题面试稿

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取概念页：`wiki/concepts/PagedAttention.md`
- 读取概念页：`wiki/concepts/Continuous Batching.md`
- 读取概念页：`wiki/concepts/Speculative Decoding.md`
- 读取实体页：`wiki/entities/TensorRT-LLM.md`
- 生成查询报告：`output/reports/推理系统专题面试稿.md`
- 更新概念页：`wiki/concepts/PagedAttention.md`
- 更新概念页：`wiki/concepts/Continuous Batching.md`
- 更新实体页：`wiki/entities/TensorRT-LLM.md`
- 本次输出围绕 `KV Cache / PagedAttention / Copy-on-Write / Batch Size / P99 延迟优化 / Speculative Decoding / TensorRT-LLM / 推理引擎调优` 形成了一份可直接复述的专题稿，并把高价值结论回填到相关页面

## [2026-04-23] query | 字节二面代码模板落地

- 基于报告：`output/reports/字节二面高压题拆解.md`
- 生成代码文件：`output/code/bytedance_pressure_round_modules.py`
- 生成测试文件：`output/code/test_bytedance_pressure_round_modules.py`
- 代码覆盖：`MHA`、`MoE`、`MLA`、`reduce`
- 运行验证：`python3 output/code/test_bytedance_pressure_round_modules.py`
- 运行验证：`python3 output/code/bytedance_pressure_round_modules.py`
- 备注：代码以“本地可运行、便于手撕复习”为目标，优先保证 shape 清晰和最小实现，而非生产级性能

## [2026-04-23] query | 字节二面高压题拆解

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/FlashAttention.md`
- 读取概念页：`wiki/concepts/Online Softmax.md`
- 读取概念页：`wiki/concepts/CUDA Kernel.md`
- 读取概念页：`wiki/concepts/Block Reduce.md`
- 读取概念页：`wiki/concepts/Warp Shuffle Reduce.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取既有报告：`output/reports/面试经验.md`
- 读取既有报告：`output/reports/大模型系统面试题全答.md`
- 生成查询报告：`output/reports/字节二面高压题拆解.md`
- 本次输出围绕 `linear attention / MoE / MHA / MLA / DeepSeek-V3 / reduce / heap sort` 给出一版高压面拆解，重点补了可手撕的最小代码模板以及 `参数量 / FLOPs / 访存` 的口算框架

## [2026-04-23] query | AI Infra面试题全答（二）

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/数据并行.md`
- 读取概念页：`wiki/concepts/Tensor Parallelism.md`
- 读取概念页：`wiki/concepts/流水线并行.md`
- 读取概念页：`wiki/concepts/ZeRO.md`
- 读取概念页：`wiki/concepts/Torch Distributed.md`
- 读取概念页：`wiki/concepts/集合通信.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取概念页：`wiki/concepts/PagedAttention.md`
- 读取概念页：`wiki/concepts/Continuous Batching.md`
- 读取概念页：`wiki/concepts/FlashAttention.md`
- 读取概念页：`wiki/concepts/Profiling.md`
- 读取实体页：`wiki/entities/NCCL.md`
- 读取实体页：`wiki/entities/vLLM.md`
- 读取实体页：`wiki/entities/SGLang.md`
- 读取实体页：`wiki/entities/TensorRT-LLM.md`
- 读取既有报告：`output/reports/大模型系统面试题全答.md`
- 读取既有报告：`output/reports/多卡与推理系统面试梳理.md`
- 生成查询报告：`output/reports/AI Infra面试题全答（二）.md`
- 本次输出补齐了更偏平台、参数服务器、调度、存储、MLOps、管理协作和行为面的题目；其中部分题采用了“可替换模板”形式，便于后续用个人真实经历替换

## [2026-04-23] query | 大模型系统面试题全答

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Roofline 模型.md`
- 读取概念页：`wiki/concepts/数据并行.md`
- 读取概念页：`wiki/concepts/Tensor Parallelism.md`
- 读取概念页：`wiki/concepts/流水线并行.md`
- 读取概念页：`wiki/concepts/ZeRO.md`
- 读取概念页：`wiki/concepts/Torch Distributed.md`
- 读取概念页：`wiki/concepts/集合通信.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取概念页：`wiki/concepts/PagedAttention.md`
- 读取概念页：`wiki/concepts/Continuous Batching.md`
- 读取概念页：`wiki/concepts/FlashAttention.md`
- 读取概念页：`wiki/concepts/Speculative Decoding.md`
- 读取实体页：`wiki/entities/NCCL.md`
- 读取实体页：`wiki/entities/vLLM.md`
- 读取实体页：`wiki/entities/SGLang.md`
- 读取实体页：`wiki/entities/TensorRT-LLM.md`
- 读取既有报告：`output/reports/面试经验.md`
- 读取既有报告：`output/reports/多卡与推理系统面试梳理.md`
- 读取既有报告：`output/reports/大模型系统面试题地图.md`
- 生成查询报告：`output/reports/大模型系统面试题全答.md`
- 本次输出按题单顺序给出一版可面试复述的全答，覆盖概念题、系统设计题、行为题和手写题；其中量化、大规模集群和平台工程部分仍属于高层答法，后续可继续拆专题

## [2026-04-23] query | 大模型系统面试题地图

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Roofline 模型.md`
- 读取概念页：`wiki/concepts/数据并行.md`
- 读取概念页：`wiki/concepts/Tensor Parallelism.md`
- 读取概念页：`wiki/concepts/流水线并行.md`
- 读取概念页：`wiki/concepts/ZeRO.md`
- 读取概念页：`wiki/concepts/Torch Distributed.md`
- 读取概念页：`wiki/concepts/集合通信.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取概念页：`wiki/concepts/PagedAttention.md`
- 读取概念页：`wiki/concepts/Continuous Batching.md`
- 读取概念页：`wiki/concepts/FlashAttention.md`
- 读取概念页：`wiki/concepts/Speculative Decoding.md`
- 读取实体页：`wiki/entities/NCCL.md`
- 读取实体页：`wiki/entities/vLLM.md`
- 读取实体页：`wiki/entities/SGLang.md`
- 读取实体页：`wiki/entities/TensorRT-LLM.md`
- 读取既有报告：`output/reports/面试经验.md`
- 读取既有报告：`output/reports/多卡与推理系统面试梳理.md`
- 生成查询报告：`output/reports/大模型系统面试题地图.md`
- 本次输出的目标不是逐题作答，而是把题单压缩成“硬件性能 -> 分布式训练 -> 推理系统 -> 平台工程”四层知识地图，并标出当前 wiki 的覆盖区与空白区

## [2026-04-23] ingest | 美团一面：请介绍 vLLM PageAttention

- 读取原始资料：`raw/articles/美团一面：请介绍 vLLM PageAttention.md`
- 创建来源页：`wiki/sources/美团一面：请介绍 vLLM PageAttention.md`
- 更新概念页：`wiki/concepts/PagedAttention.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 更新实体页：`wiki/entities/vLLM.md`
- 未发现与现有 wiki 的直接冲突，本次主要把 `PagedAttention` 从“分页式 KV 管理”推进到“logical block / physical block / block table + prefill/decode 运行流程”视角

## [2026-04-23] query | 多卡与推理系统面试梳理

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Tensor Parallelism.md`
- 读取概念页：`wiki/concepts/集合通信.md`
- 读取概念页：`wiki/concepts/Torch Distributed.md`
- 读取概念页：`wiki/concepts/Continuous Batching.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取概念页：`wiki/concepts/PD分离.md`
- 读取概念页：`wiki/concepts/Prefix Caching.md`
- 读取概念页：`wiki/concepts/缓存感知路由.md`
- 读取概念页：`wiki/concepts/PagedAttention.md`
- 读取实体页：`wiki/entities/NCCL.md`
- 读取实体页：`wiki/entities/vLLM.md`
- 读取实体页：`wiki/entities/SGLang.md`
- 读取既有报告：`output/reports/面试经验.md`
- 生成查询报告：`output/reports/多卡与推理系统面试梳理.md`
- 本次输出以面试口径为主，重点串联 `TP 通信原语 -> MoE/EP -> Continuous Batching 与 KV Cache -> 单卡 P/D 衔接 -> PD 解耦与分离 -> Prefix Caching 执行路径`
- 备注：`EP` 与 `PD 解耦 vs PD 分离` 目前仍主要是基于现有条目做的工程归纳，后续可补独立概念页或来源

## [2026-04-23] ingest | 秋招CUDA手撕题复盘（附代码）

- 读取原始资料：`raw/articles/秋招CUDA手撕题复盘（附代码）.md`
- 创建来源页：`wiki/sources/秋招CUDA手撕题复盘（附代码）.md`
- 创建概念页：`wiki/concepts/Warp Shuffle Reduce.md`
- 创建概念页：`wiki/concepts/Block Reduce.md`
- 创建概念页：`wiki/concepts/Grid-stride Loop.md`
- 创建概念页：`wiki/concepts/RMSNorm.md`
- 创建概念页：`wiki/concepts/Histogram.md`
- 更新概念页：`wiki/concepts/CUDA Kernel.md`
- 更新概念页：`wiki/concepts/Online Softmax.md`
- 更新概念页：`wiki/concepts/Tiling.md`
- 未发现与现有 wiki 的直接冲突，本次主要把 CUDA 条目从“性能优化清单”推进到“面试高频 kernel 模板”视角，补入了 `Warp Shuffle Reduce / Block Reduce / Grid-stride Loop` 这组骨架概念

## [2026-04-22] ingest | Kimi新作《Attention Residuals》：对Transformer中残差结构的调整

- 读取原始资料：`raw/articles/Kimi新作《Attention Residuals》：对Transformer中残差结构的调整.md`
- 补充原始论文：`raw/papers/2603.15031v1.pdf`
- 创建来源页：`wiki/sources/Kimi新作《Attention Residuals》：对Transformer中残差结构的调整.md`
- 创建实体页：`wiki/entities/Moonshot AI.md`
- 创建概念页：`wiki/concepts/Attention Residuals.md`
- 创建概念页：`wiki/concepts/PreNorm Dilution.md`
- 更新概念页：`wiki/concepts/流水线并行.md`
- 更新概念页：`wiki/concepts/Scaling Laws.md`
- 未发现与现有 wiki 的直接冲突，本次主要补入了 `标准残差 -> Attention Residuals -> Block AttnRes` 这条跨层聚合改造路线，以及它和 pipeline communication / scaling-law 评估的联系

## [2026-04-23] query | 面试经验

- 生成输出文档：`output/reports/面试经验.md`
- 整理主题：`SFT`、`RLHF`、`MoE 路由`、`PPO`、`大模型架构对比`
- 本次输出以面试口径为主，强调“问题 -> 结论 -> 原因 -> 权衡”的答题结构
- 后续补充：新增 `RoPE` 的五层面试追问框架，覆盖机制理解、数学直觉、实现细节、系统影响与 research tradeoff
- 后续补充：新增简历高频八股速查，覆盖 `FlashAttention`、`算子融合`、`Nsight Systems / Nsight Compute`、基础模型架构、`KV cache` 与 `per-group quantization group size`
- 后续补充：新增 `CUDA Graphs` 速查，整理其性能原理、5 个经典坑，以及 `torch.compile(mode=\"reduce-overhead\")` 作为低成本入场方式

## [2026-04-22] ingest | mHC: Manifold-Constrained Hyper-Connections

- 读取原始资料：`raw/papers/2512.24880v2.pdf`
- 创建来源页：`wiki/sources/mHC: Manifold-Constrained Hyper-Connections.md`
- 创建实体页：`wiki/entities/DeepSeek-AI.md`
- 创建概念页：`wiki/concepts/Hyper-Connections.md`
- 创建概念页：`wiki/concepts/mHC.md`
- 更新概念页：`wiki/concepts/算子融合.md`
- 更新概念页：`wiki/concepts/重计算.md`
- 更新概念页：`wiki/concepts/流水线并行.md`
- 未发现与现有 wiki 的直接冲突，本次主要补入了 `HC -> mHC` 这条宏观拓扑设计主线，以及论文里的 kernel fusion / recomputing / pipeline overlap 系统实现视角

## [2026-04-13] bootstrap | 初始化 AI Infra LLM Wiki

- 建立 `raw/`、`wiki/`、`output/`、`scripts/` 基础结构
- 增加 `AGENTS.md` 作为 schema 规则文件
- 建立 `index.md` 与 `log.md` 作为导航层

## [2026-04-13] ingest | LLM推理优化核心技术

- 读取原始资料：`raw/articles/LLM推理优化核心技术.md`
- 创建来源页：`wiki/sources/LLM推理优化核心技术.md`
- 创建实体页：`wiki/entities/vLLM.md`
- 创建实体页：`wiki/entities/SGLang.md`
- 创建实体页：`wiki/entities/Nvidia Dynamo.md`
- 创建实体页：`wiki/entities/TensorRT-LLM.md`
- 创建概念页：`wiki/concepts/KV Cache.md`
- 创建概念页：`wiki/concepts/Prefix Caching.md`
- 创建概念页：`wiki/concepts/缓存感知路由.md`
- 创建概念页：`wiki/concepts/PagedAttention.md`
- 创建概念页：`wiki/concepts/PD分离.md`
- 创建概念页：`wiki/concepts/Tensor Parallelism.md`
- 未发现直接矛盾，`PD分离` 的适用边界标记为后续待验证

## [2026-04-13] ingest | Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构

- 读取原始资料：`raw/articles/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构.md`
- 创建来源页：`wiki/sources/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构.md`
- 创建实体页：`wiki/entities/Gemma 4.md`
- 创建实体页：`wiki/entities/Google DeepMind.md`
- 创建实体页：`wiki/entities/特里斯丹井底之娃 往上爬.md`
- 创建概念页：`wiki/concepts/Per-Layer Embeddings.md`
- 创建概念页：`wiki/concepts/Shared KV Cache.md`
- 创建概念页：`wiki/concepts/混合注意力.md`
- 创建概念页：`wiki/concepts/Dual RoPE.md`
- 创建概念页：`wiki/concepts/Double-Wide MLP.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 未发现直接矛盾，`Double-Wide MLP` 作为补偿机制的解释标记为后续待验证

## [2026-04-13] query | Gemma 开源代码结构导读

- 删除原始资料：`raw/articles/Google Gemma 4 深度解析：当开源AI进入「逐层嵌入平行化」时代-程序员茄子.md`
- 更新索引：`wiki/index.md`
- 运行健康检查：`scripts/lint.py`
- 生成查询报告：`output/reports/Gemma 开源代码结构导读.md`
- 本次查询以官方源码结构为主，重点核对 `google/gemma_pytorch` 与 `transformers/models/gemma4`

## [2026-04-13] ingest | 斯坦福CS336 Lecture 5 - GPUs

- 读取原始资料：`raw/articles/斯坦福CS336 Lecture 5 - GPUs.md`
- 创建来源页：`wiki/sources/斯坦福CS336 Lecture 5 - GPUs.md`
- 创建实体页：`wiki/entities/Stanford CS336.md`
- 创建概念页：`wiki/concepts/GPU执行模型.md`
- 创建概念页：`wiki/concepts/Roofline 模型.md`
- 创建概念页：`wiki/concepts/算子融合.md`
- 创建概念页：`wiki/concepts/重计算.md`
- 创建概念页：`wiki/concepts/内存合并访问.md`
- 创建概念页：`wiki/concepts/Tiling.md`
- 创建概念页：`wiki/concepts/FlashAttention.md`
- 标注说明：B 站镜像标题中的“分布式训练基础”范围比 Stanford 官方讲义 `Lecture 5 - GPUs` 更宽
- 未发现与现有 wiki 的直接冲突，后续建议补 `Lecture 7 - Parallelism basics` 与本讲组成前后衔接

## [2026-04-13] ingest | 斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing

- 读取原始资料：`raw/articles/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing.md`
- 创建来源页：`wiki/sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing.md`
- 创建概念页：`wiki/concepts/Benchmarking.md`
- 创建概念页：`wiki/concepts/Profiling.md`
- 创建概念页：`wiki/concepts/CUDA Kernel.md`
- 创建概念页：`wiki/concepts/Triton.md`
- 创建概念页：`wiki/concepts/torch.compile.md`
- 更新实体页：`wiki/entities/Stanford CS336.md`
- 更新概念页：`wiki/concepts/算子融合.md`
- 更新概念页：`wiki/concepts/Tiling.md`
- 未发现与现有 wiki 的直接冲突，Lecture 6 已形成对 Lecture 5 的自然延续

## [2026-04-13] ingest | 斯坦福CS336 Lecture 7 - Parallelism basics

- 读取原始资料：`raw/articles/斯坦福CS336 Lecture 7 - Parallelism basics.md`
- 创建来源页：`wiki/sources/斯坦福CS336 Lecture 7 - Parallelism basics.md`
- 创建概念页：`wiki/concepts/集合通信.md`
- 创建概念页：`wiki/concepts/数据并行.md`
- 创建概念页：`wiki/concepts/ZeRO.md`
- 创建概念页：`wiki/concepts/FSDP.md`
- 创建概念页：`wiki/concepts/流水线并行.md`
- 创建概念页：`wiki/concepts/Sequence Parallelism.md`
- 更新概念页：`wiki/concepts/Tensor Parallelism.md`
- 更新实体页：`wiki/entities/Stanford CS336.md`
- 未发现与现有 wiki 的直接冲突，Lecture 7 已把课程线推进到多机多卡训练并行

## [2026-04-13] ingest | 斯坦福CS336 Lecture 8 - Distributed communication and training code

- 读取原始资料：`raw/articles/斯坦福CS336 Lecture 8 - Distributed communication and training code.md`
- 创建来源页：`wiki/sources/斯坦福CS336 Lecture 8 - Distributed communication and training code.md`
- 创建实体页：`wiki/entities/NCCL.md`
- 创建概念页：`wiki/concepts/Torch Distributed.md`
- 更新概念页：`wiki/concepts/集合通信.md`
- 更新概念页：`wiki/concepts/数据并行.md`
- 更新概念页：`wiki/concepts/Tensor Parallelism.md`
- 更新概念页：`wiki/concepts/流水线并行.md`
- 更新实体页：`wiki/entities/Stanford CS336.md`
- 未发现与现有 wiki 的直接冲突，Lecture 8 已把并行原理推进到 `torch.distributed` 与通信骨架代码层

## [2026-04-13] ingest | 斯坦福CS336 Lecture 9 - Scaling laws basics

- 读取原始资料：`raw/articles/斯坦福CS336 Lecture 9 - Scaling laws basics.md`
- 创建来源页：`wiki/sources/斯坦福CS336 Lecture 9 - Scaling laws basics.md`
- 创建概念页：`wiki/concepts/Scaling Laws.md`
- 创建概念页：`wiki/concepts/数据缩放定律.md`
- 创建概念页：`wiki/concepts/Critical Batch Size.md`
- 创建概念页：`wiki/concepts/Chinchilla Scaling.md`
- 更新实体页：`wiki/entities/Stanford CS336.md`
- 未发现与现有 wiki 的直接冲突，Lecture 9 已把课程线从系统实现推进到模型设计与资源分配规律

## [2026-04-13] ingest | 斯坦福CS336 Lecture 10 - Inference systems and optimization

- 读取原始资料：`raw/articles/斯坦福CS336 Lecture 10 - Inference systems and optimization.md`
- 创建来源页：`wiki/sources/斯坦福CS336 Lecture 10 - Inference systems and optimization.md`
- 创建概念页：`wiki/concepts/Continuous Batching.md`
- 创建概念页：`wiki/concepts/Speculative Decoding.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 更新概念页：`wiki/concepts/PagedAttention.md`
- 更新实体页：`wiki/entities/vLLM.md`
- 未发现与现有 wiki 的直接冲突，Lecture 10 已把课程线推进到推理系统与动态 workload 优化

## [2026-04-14] ingest | Flash Attention 详细解释推演与Pytorch代码实现

- 读取原始资料：`raw/articles/Flash Attention 详细解释推演与Pytorch代码实现.md`
- 创建来源页：`wiki/sources/Flash Attention 详细解释推演与Pytorch代码实现.md`
- 创建概念页：`wiki/concepts/Online Softmax.md`
- 更新概念页：`wiki/concepts/FlashAttention.md`
- 更新概念页：`wiki/concepts/Tiling.md`
- 未发现与现有 wiki 的直接冲突，本次主要补强了 `FlashAttention` 的 IO-aware 视角、`Online Softmax` 的状态合并逻辑，以及 `FA1/FA2` 的工作分配差异

## [2026-04-22] ingest | 十分钟读懂旋转编码（RoPE）

- 读取原始资料：`raw/articles/十分钟读懂旋转编码（RoPE）.md`
- 创建来源页：`wiki/sources/十分钟读懂旋转编码（RoPE）.md`
- 创建概念页：`wiki/concepts/RoPE.md`
- 更新概念页：`wiki/concepts/Dual RoPE.md`
- 未发现与现有 wiki 的直接冲突，本次主要补齐了通用 `RoPE` 概念锚点，并把现有 `Dual RoPE` 收束回更基础的位置编码语境

## [2026-04-22] ingest | 你一定要知道：CUDA优化六要

- 读取原始资料：`raw/articles/你一定要知道：CUDA优化六要.md`
- 创建来源页：`wiki/sources/你一定要知道：CUDA优化六要.md`
- 创建概念页：`wiki/concepts/Bank Conflict.md`
- 创建概念页：`wiki/concepts/Occupancy.md`
- 创建概念页：`wiki/concepts/Warp Divergence.md`
- 创建概念页：`wiki/concepts/Tail Effect.md`
- 更新概念页：`wiki/concepts/CUDA Kernel.md`
- 更新概念页：`wiki/concepts/GPU执行模型.md`
- 更新概念页：`wiki/concepts/内存合并访问.md`
- 更新概念页：`wiki/concepts/Tiling.md`
- 未发现与现有 wiki 的直接冲突，本次主要把 CUDA 优化经验条目整理成可复用的性能排障概念集

## [2026-04-22] ingest | Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

- 读取原始资料：`raw/papers/2601.07372v1.pdf`
- 创建来源页：`wiki/sources/Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models.md`
- 创建实体页：`wiki/entities/Engram.md`
- 创建概念页：`wiki/concepts/Conditional Memory.md`
- 创建概念页：`wiki/concepts/Sparsity Allocation.md`
- 未发现与现有 wiki 的直接冲突，本次主要补入了 `conditional memory` / `Engram` / `Sparsity Allocation` 这一条新的稀疏建模主线

## [2026-04-22] ingest | engram_demo_v1

- 读取原始资料：`raw/code/engram_demo_v1.py`
- 创建来源页：`wiki/sources/engram_demo_v1.md`
- 更新实体页：`wiki/entities/Engram.md`
- 更新概念页：`wiki/concepts/Conditional Memory.md`
- 未发现与现有 wiki 的直接冲突，本次主要把 `Engram` 从论文摘要推进到 demo 代码实现视角
