# 知识库操作日志

按时间记录 ingest、query、lint 等操作，帮助 LLM 与人类共同追踪知识库的演化过程。

## [2026-06-22] query | Split-KV 在现代 decode attention 中是否常见

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Flash Decoding.md`、`wiki/concepts/FlashAttention.md`、`wiki/concepts/Decode Context Parallel.md`
- 搜索本地资料中的 `Split-KV / Flash Decoding / KV split`
- 参考外部资料：PyTorch Flash-Decoding 博客、FlashInfer attention 文档、TensorRT-LLM GPT attention 文档
- 更新概念页：`wiki/concepts/Flash Decoding.md`
- 本次 query 澄清：`Split-KV` 已经是现代 LLM serving decode attention 后端中的常见技巧，尤其适合长上下文、小 batch、`Q length` 很短的 decode；但它不是所有 attention 的默认路径。Prefill / training 更常沿 query block、head、batch 等维度并行；decode 是否启用 Split-KV 通常取决于 backend heuristic、上下文长度、batch、head 数、cache layout、硬件和合并开销。

## [2026-06-22] query | DP Attention 概念复查

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/DP Attention.md`、`wiki/concepts/MLA.md`、`wiki/concepts/Tensor Parallelism.md`
- 读取来源页：`wiki/sources/MLA与DP Attention面试整理.md`
- 参考官方资料：SGLang `DP, DPA and SGLang DP Router` 文档、SGLang DeepSeek V3/V3.1/R1 使用文档
- 本次 query 复用既有结论：`DP Attention / DPA` 是 attention component 级别的数据并行策略，不是把单个请求的 context 切开；它让不同 attention DP replica 处理不同请求/batch，并分别维护 KV cache。它适合 DeepSeek/MLA 等在普通 TP attention 下容易复制 latent KV cache 的模型，常与 MoE 的 [[Expert Parallelism]] 组合，用于提升多请求吞吐和可承载 batch size。

## [2026-06-22] query | PD 分离与 Chunked Prefill 对 TPOT 的作用

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/PD分离.md`、`wiki/concepts/Chunked Prefill.md`、`wiki/concepts/Continuous Batching.md`
- 更新概念页：`wiki/concepts/PD分离.md`、`wiki/concepts/Chunked Prefill.md`
- 本次 query 澄清：PD 分离和 Chunked Prefill 都是在缓解 prefill 对 decode 的阻塞，主要保护 decode ITL / TPOT 和 tail latency，而不是减少单个 decode token 的模型计算；PD 分离靠资源池隔离，Chunked Prefill 靠调度粒度切分。二者可能把成本转移到 TTFT、prefill 吞吐、KV cache 传输、调度开销或资源利用率上。

## [2026-06-22] query | MTP 推理验证规则

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Speculative Decoding.md`、`wiki/concepts/Multi-Token Prediction.md`
- 更新概念页：`wiki/concepts/Speculative Decoding.md`
- 本次 query 澄清：MTP 推理中的验证由 target model 完成，而不是 MTP 自己判断；target 对 `prefix + draft` 做一次 forward，逐位置 score draft token。greedy 场景可用 argmax 一致性做连续接受；采样场景需要按 `min(1, p/q)` 的 speculative sampling 接受规则以及拒绝时的 `(p-q)_+` 替代分布，才能保持 target model 分布。runtime 形状上，verify 常把普通 decode 的 `qlen=1` 变成 `qlen=draft_len` 的 chunk decode，历史 prefix 仍来自 KV cache。

## [2026-06-22] query | DeepSeek-V3 MTP 实现口径

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Multi-Token Prediction.md`、`wiki/concepts/MTP Drafter.md`
- 参考外部资料：DeepSeek-V3 Technical Report、DeepSeek-V3 官方 `README_WEIGHTS.md`、NVIDIA Megatron-LM MTP 文档
- 更新概念页：`wiki/concepts/Multi-Token Prediction.md`
- 本次 query 澄清：DeepSeek-V3 的 MTP 不是简单并排多个 linear head，而是顺序 MTP module；每个 module 通过共享 embedding、projection、Transformer block、共享 output head 保留 causal chain。开源 V3 权重中 `num_nextn_predict_layers=1`，MTP 作为 `model.layers.61` 追加在 61 层主模型之后，并共享主模型 embedding 与 output head。

## [2026-06-22] query | DCP 和 Flash Decoding 区别复查

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Decode Context Parallel.md`、`wiki/concepts/Flash Decoding.md`、`wiki/concepts/KV Cache.md`
- 读取来源页：`wiki/sources/vllm并行策略之DCP(Decode Context Parallel).md`，并参考原始 Flash Decoding 资料片段
- 本次 query 复用既有结论：二者都沿 context/KV 维切分并用 online softmax / log-sum-exp 合并局部结果；`Flash Decoding` 更偏 decode attention kernel / split-KV 算法思路，目标是提高小 `Q`、长 `KV` 时的并行度；`DCP` 更偏多 GPU serving 并行策略，目标是在 TP group 内减少小 `num_kv_heads` 模型的 KV cache 重复，并处理 interleaved KV cache、跨 rank 通信、prefill 写 cache 等系统问题。

## [2026-06-16] query | MHA/GQA/MLA prefill 计算量对比

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/MLA.md`、`wiki/concepts/KV Cache.md`
- 读取来源页：`wiki/sources/MLA与DP Attention面试整理.md`、`wiki/sources/斯坦福CS336 Lecture 10 - Inference systems and optimization.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 本次 query 澄清：在输入 `[B, S, D]` 的单层 causal prefill 中，`MHA` 和 `GQA/MQA` 的 `QK^T + P @ V` 主体算术量同阶，近似 `B * D * S^2` MACs；`GQA/MQA` 主要减少 K/V 投影和 KV cache 写入/读取。`MLA` 的 absorbed latent 路径会把历史 cache 从完整 K/V 压到 `C + R` 元素，但 attention core 可能变成 `0.5 * B * S * (S + 1) * H_q * (2C + R)` MACs，因此它更像用额外或重排后的计算换更少 HBM 访问，收益尤其体现在 decode。

## [2026-06-15] query | CuTe DSL 概念补页

- 读取索引页：`wiki/index.md`
- 搜索本地资料中的 `CuTe / CuTeDSL / CUTLASS / DSL`
- 参考官方资料：NVIDIA CUTLASS CuTe DSL / Python DSL 文档与 NVIDIA 技术博客
- 创建概念页：`wiki/concepts/CuTe DSL.md`
- 更新概念页：`wiki/concepts/CODA.md`、`wiki/concepts/Triton.md`、`wiki/concepts/Torch Compile.md`
- 更新来源页：`wiki/sources/还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速.md`
- 本次 query 将 CuTe DSL 整理为 CUTLASS 4.x 中面向 GPU kernel authoring 的 Python-native 低级 DSL：它把 CuTe / CUTLASS C++ 的 layout、tensor、atom、tiled operation 等抽象搬到 Python 语法和 JIT / MLIR / `ptxas` 编译路径中；它比 Triton 更贴近 CUTLASS/CuTe 的 layout algebra 与硬件 atom 组合，不应理解为自动图优化器。

## [2026-06-12] ingest | Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B

- 读取原始资料：`raw/articles/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B 1.md`
- 创建来源页：`wiki/sources/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B.md`
- 创建概念页：`wiki/concepts/Megakernel.md`、`wiki/concepts/Programmatic Dependent Launch.md`
- 创建实体页：`wiki/entities/HazyResearch.md`、`wiki/entities/Megakernels.md`
- 更新概念页：`wiki/concepts/CUDA Kernel.md`、`wiki/concepts/算子融合.md`、`wiki/concepts/Tail Effect.md`、`wiki/concepts/Roofline 模型.md`、`wiki/concepts/CUDA内存层次.md`、`wiki/concepts/GPU执行模型.md`、`wiki/concepts/Profiling.md`
- 更新实体页：`wiki/entities/vLLM.md`、`wiki/entities/SGLang.md`
- 本次 ingest 将该文整理为“低延迟 batch size 1 Llama-1B decode 中 kernel 边界本身成为瓶颈”的系统案例：作者通过整模型 [[Megakernel]]、on-GPU interpreter、shared memory paging 和显式 counter 同步减少 memory pipeline bubbles；H100 `78%` memory bandwidth、相对 vLLM/SGLang `1.5x+`、B200 `<680 us` 等性能数字均保留为来源声称，需绑定硬件、dtype、prompt/generation 长度、baseline 配置和代码版本。

## [2026-06-12] query | vLLM cudagraph_mode NONE 与 enforce eager

- 读取索引页：`wiki/index.md`
- 搜索本地资料中的 `CUDA Graph / cudagraph / enforce_eager / torch.compile`
- 参考官方资料：vLLM `debug_vllm_compile` 与 `CUDA Graphs` 设计文档
- 更新实体页：`wiki/entities/vLLM.md`
- 更新概念页：`wiki/concepts/Torch Compile.md`
- 本次 query 澄清：`cudagraph_mode=NONE` 只关闭 CUDA Graphs，通常用于排查 graph capture / replay / graph memory pool 问题；`--enforce-eager` 是更强的 vLLM eager 总开关，会关闭 `torch.compile` 集成和 CUDA Graphs，因此二者在“无 CUDA Graph”这一点上相同，但在是否仍允许 compile / Inductor / vLLM compile 路径上不同。

## [2026-06-12] query | NVFP4 与 MXFP4 格式区别

- 读取索引页：`wiki/index.md`
- 搜索本地资料中的 `NVFP4 / MXFP4 / FP4 / 量化`
- 参考外部资料：NVIDIA Transformer Engine / NVIDIA Technical Blog、OCP Microscaling Formats MX v1.0、AMD ROCm MXFP4/MXFP6 说明、Triton block-scaled matmul 文档
- 更新概念页：`wiki/concepts/混合精度训练与推理.md`
- 本次 query 澄清：`MXFP4` 是 OCP microscaling FP4 格式，通常由 `FP4 E2M1` 元素加每 32 个元素共享的 `E8M0` scale 组成；`NVFP4` 是 NVIDIA Blackwell/Transformer Engine 的 FP4 recipe，由 `FP4 E2M1` 元素、每 16 个元素共享的 `FP8 E4M3` block scale 和 per-tensor `FP32` global scale 组成。二者都不是裸 4-bit float，差异主要在 block size、scale dtype、是否有 global scale、硬件/框架支持与量化误差。

## [2026-06-11] query | vLLM enforce eager 与不开 CUDA Graph 的区别

- 读取索引页：`wiki/index.md`
- 搜索本地资料中的 `CUDA Graph / eager / capture`
- 参考官方资料：vLLM engine args / serve args 文档
- 本次 query 澄清：`--enforce-eager` 是 vLLM 级别的强制 eager 执行开关，会禁用 CUDA Graph 并让模型始终走 PyTorch eager；默认 `enforce_eager=False` 时，vLLM 是 CUDA Graph 与 eager 的混合模式，能 capture 的固定路径走 graph，超过 capture 范围或不适合 capture 的动态路径回退 eager。“不开 CUDA Graph”若只是把 capture size 设小或让请求形态落到未 capture 范围，行为上会更多回退 eager，但语义上不一定等同于 `enforce_eager=True`。

## [2026-06-11] query | Mamba page size 与时序 state 的关系

- 读取索引页：`wiki/index.md`
- 搜索本地资料中的 `Mamba / hybrid / page size / state cache`
- 参考外部资料：vLLM Hybrid KV Cache Manager 设计文档、vLLM `MambaSpec` / KV cache utils 源码、PyTorch/vLLM hybrid models 博客
- 本次 query 澄清：Mamba / linear attention 的算法记忆仍是 recurrent state，不是按历史 token 存完整 KV；但 vLLM 的 cache manager 需要把 attention KV cache 与 Mamba state cache 放进统一 allocator / block table / page accounting，因此给 Mamba state 引入 `page_size_bytes`。这里的 page 是物理内存管理单位，不等同于 Mamba 算法上的 token 历史分页。

## [2026-06-11] query | TRTLLM kernel 与 FlashInfer

- 读取索引页：`wiki/index.md`
- 读取实体页：`wiki/entities/TensorRT-LLM.md`
- 读取概念页：`wiki/concepts/CUDA Kernel.md`、`wiki/concepts/算子融合.md`
- 参考官方资料：NVIDIA TensorRT-LLM 文档、FlashInfer 文档 / GitHub README、NVIDIA FlashInfer 技术博客
- 创建实体页：`wiki/entities/FlashInfer.md`
- 更新实体页：`wiki/entities/TensorRT-LLM.md`
- 本次 query 澄清：`TRTLLM kernel` 通常不是一个具体算子名，而是指 TensorRT-LLM 内部或来源于 TensorRT-LLM 的高性能推理 kernel 集合；FlashInfer 则更像可被不同 serving engine 集成的 LLM kernel 库 / generator，覆盖 attention、GEMM、MoE、sampling、通信等热点路径。

## [2026-06-11] query | Qwen3Next config 参数量估算

- 读取索引页：`wiki/index.md`
- 读取用户提供的 `Qwen3NextForCausalLM` config
- 本次 query 按 Hugging Face 风格的 per-layer MoE 参数组织估算：若 `num_experts=512` 表示每个 MoE 层各自拥有 512 个 routed experts，则该 config 总参数约 `1,579.55B`，即约 `1.58T`；其中 MoE experts 占主导，约 `1,549.46B`。每 token 激活口径下，按 `top_k=10` 加 shared expert 粗算，层内 active 参数约 `59.49B`，若连 untied embedding 与 lm_head 一并计入约 `63.56B`。该结果说明此 config 与常见 `80B-A3B` 口径不一致，需确认是否为缩放/实验 config 或 experts 是否存在跨层共享、分片元信息等额外约定。

## [2026-06-11] query | Qwen3Next chunked GDN 算子形状与优化

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Chunked Gated Delta Rule.md`
- 参考用户提供的 `Qwen3NextForCausalLM` config，并核对 Hugging Face Transformers / FLA 相关实现
- 更新概念页：`wiki/concepts/Chunked Gated Delta Rule.md`
- 本次 query 将 `chunked GDN` 落到 Qwen3Next shape 口径：`hidden [B,T,8192]` 经投影、causal depthwise conv 后形成 `q/k/v [B,T,128,128]`、`g/beta [B,T,128]`，`chunk_gated_delta_rule` 输出 `[B,T,128,128]`，再经 gated RMSNorm 与 `out_proj` 回到 `[B,T,8192]`；其 cache state 为 `[N,128,128,128]`，是常数长度 recurrent state，不是随上下文线性增长的 KV cache。

## [2026-06-09] query | Linear Attention 是否有 Prefix Cache

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Prefix Caching.md`、`wiki/concepts/KV Cache.md`、`wiki/concepts/Chunked Gated Delta Rule.md`
- 参考外部资料：`Transformers are RNNs`、`Parallelizing Linear Transformers with the Delta Rule over Sequence Length`、`Kimi Linear`
- 本次 query 澄清：线性注意力通常不需要标准 Transformer 那种逐 token 增长的 `KV Cache`；decode 路径更常缓存每层 recurrent/state 表示。若多个请求共享完全相同前缀，可以缓存该前缀结束处的 state 并作为后续 continuation 的初始状态，因此功能上有“prefix state cache”；但它不是传统 `KV prefix cache`，最长前缀匹配、分块 checkpoint、回滚与混合 attention 层的实现都依赖具体 runtime。

## [2026-06-09] query | CuTe DSL 抽象层级

- 读取索引页：`wiki/index.md`
- 搜索本地资料中的 `CuTe / CuTeDSL / CUTLASS / DSL`
- 参考官方资料：NVIDIA CUTLASS CuTe DSL / CUTLASS 4.x Python DSL 文档与 NVIDIA 技术博客
- 本次 query 澄清：CuTe DSL 是 CUTLASS 4.x 中面向 GPU kernel authoring 的 Python-native 低级 DSL，基本沿用 CuTe C++ 的 layout、tensor、atom、tiled operation 等抽象；它相对 CuTe C++ 更像“Python 语法外壳 + JIT/MLIR 编译路径”，不是像 Triton/TVM 那样明显更高层的张量程序 DSL。

## [2026-06-09] query | DCP 与 Flash Decoding 区别

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Decode Context Parallel.md`、`wiki/concepts/Flash Decoding.md`、`wiki/concepts/KV Cache.md`、`wiki/concepts/Tensor Parallelism.md`
- 更新概念页：`wiki/concepts/Decode Context Parallel.md`
- 本次 query 澄清：DCP 和 Flash Decoding 的共同点是都沿 context/KV 维切分并用 online softmax / log-sum-exp 合并局部结果；区别是 Flash Decoding 更偏 attention kernel / 算法思路，目标是小 `Q` decode 下增加 `KV split` 并行度，DCP 更偏多 GPU serving 并行策略，目标是在 TP group 内减少小 `num_kv_heads` 模型的 KV cache 复制，并处理 process group、interleaved KV cache、prefill 写入和跨 rank 通信。

## [2026-06-09] query | DP Attention 是否一定加速

- 读取概念页：`wiki/concepts/DP Attention.md`
- 更新概念页：`wiki/concepts/DP Attention.md`
- 本次 query 澄清：DP Attention 更偏多并发吞吐和可承载 batch size 优化，不保证单请求 latency 下降；当请求数少、batch 填不满或 router/通信/调度开销较高时，DPA 不一定加速，甚至可能让单请求变慢。

## [2026-06-09] query | DP Attention size 与请求数关系

- 读取概念页：`wiki/concepts/DP Attention.md`
- 更新概念页：`wiki/concepts/DP Attention.md`
- 本次 query 澄清：`dp attention = 8` 更准确表示有 8 个 attention data-parallel replica / 分片可承载请求流，而不是每个 decode step 固定只处理或正好处理 8 个请求。每个 replica 内部仍可 continuous batching 多个 active requests；单个长请求通常不会自动被 DPA 切成 8 份，这更接近 DCP 的问题。

## [2026-06-09] ingest | vllm并行策略之DCP(Decode Context Parallel)

- 读取原始资料：`raw/articles/vllm并行策略之DCP(Decode Context Parallel).md`
- 创建来源页：`wiki/sources/vllm并行策略之DCP(Decode Context Parallel).md`
- 创建实体页：`wiki/entities/梦初AI Infra.md`
- 更新概念页：`wiki/concepts/Decode Context Parallel.md`、`wiki/concepts/KV Cache.md`、`wiki/concepts/Tensor Parallelism.md`、`wiki/concepts/Flash Decoding.md`、`wiki/concepts/Chunked Prefill.md`、`wiki/concepts/Prefix Caching.md`、`wiki/concepts/MLA.md`
- 更新实体页：`wiki/entities/vLLM.md`
- 本次 ingest 将 vLLM DCP 整理为“复用 TP group 的 decode context/KV 分片策略”：它通过 interleaved KV cache 存储把同一请求的历史 token KV 按 `token_idx % cp_world_size` 分到不同 DCP rank，decode 时各 rank 计算 partial output 与 `lse`，再合并为全局 attention output。来源中的 CUDA backend 支持、PCP 状态、`dcp_all2all` 通信和 Chunked Prefill / Prefix Cache 兼容性均标记为版本相关待核实。

## [2026-06-09] query | DP Attention 是否能替代 DCP

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/DP Attention.md`、`wiki/concepts/Decode Context Parallel.md`、`wiki/concepts/Tensor Parallelism.md`、`wiki/concepts/MLA.md`
- 参考官方资料：SGLang `DP, DPA and SGLang DP Router` 文档、vLLM Context Parallel Deployment 文档
- 更新概念页：`wiki/concepts/DP Attention.md`、`wiki/concepts/Decode Context Parallel.md`
- 本次 query 澄清：DP Attention 和 DCP 都能缓解普通 TP 下 attention/KV cache 组织不理想的问题，但粒度不同。DP Attention 是请求/batch 级 replica，让不同 DP attention 副本处理不同请求并维护独立 KV cache；DCP 是单请求 context 级分片，把同一个长上下文请求的历史 KV 沿 token/context 维拆到多个 GPU 上，并合并局部 softmax 统计。二者不是简单替代关系。

## [2026-06-09] query | DCP 与 TP 关系

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Tensor Parallelism.md`、`wiki/concepts/DP Attention.md`、`wiki/concepts/Flash Decoding.md`、`wiki/concepts/KV Cache.md`
- 参考官方资料：vLLM Context Parallel Deployment 文档
- 创建概念页：`wiki/concepts/Decode Context Parallel.md`
- 更新概念页：`wiki/concepts/Tensor Parallelism.md`、`wiki/concepts/Flash Decoding.md`、`wiki/concepts/KV Cache.md`
- 本次 query 澄清：DCP 不是替代 TP，而是在 TP 已经拉大、`KV heads` 维度切分受限时，面向 decode 阶段进一步沿 context/token 维切分 `KV Cache`。普通 TP 解决层内权重/计算切分；DCP 解决长上下文 decode 下 KV cache 重复保存和 batch size 受限问题。

## [2026-06-09] query | Flash Decoding 与 FlashAttention Split-KV 关系

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/FlashAttention.md`、`wiki/concepts/PagedAttention.md`、`wiki/concepts/Online Softmax.md`
- 读取原始资料：`raw/articles/推理长序列利器：ChunkedPrefill&FlashDecoding原理详解.md`
- 创建概念页：`wiki/concepts/Flash Decoding.md`
- 更新概念页：`wiki/concepts/FlashAttention.md`、`wiki/concepts/PagedAttention.md`
- 本次 query 澄清：Flash Decoding 可以面试级理解为 decode 场景下的 Split-KV FlashAttention-family 思路；它把历史 `KV Cache` 沿 context 维切成多个 split 并行计算局部 attention，再用 online softmax / log-sum-exp 统计合并为精确全局输出。边界是：它不是 FlashAttention 的简单改名，也不是 speculative decoding；重点是小 `Q` decode 下增加 `KV/context` 维并行度。

## [2026-06-09] query | chunk_gated_delta_rule 中 chunk 含义

- 读取索引页：`wiki/index.md`
- 搜索本地资料中的 `chunk_gated_delta_rule / gated_delta_rule / chunk`
- 参考外部资料：vLLM `fla.ops.chunk.chunk_gated_delta_rule` API、GatedDeltaNet 论文、FLA/GatedDeltaNet 相关实现文档
- 创建概念页：`wiki/concepts/Chunked Gated Delta Rule.md`
- 本次 query 澄清：`chunk_gated_delta_rule` 里的 `chunk` 不是 RAG 文本切片，也不是 serving 层的 `Chunked Prefill`，而是算子内部沿时间/token 维切块。它把 Gated Delta Rule 的长递推链拆成块内矩阵化并行 + 块间 state 传递，主要服务 prefill/training 等长序列并行计算。

## [2026-06-09] query | MTP 层结构

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Multi-Token Prediction.md`、`wiki/concepts/MTP Drafter.md`
- 读取原始资料：`raw/articles/LLM提速利器：投机推理的原理与常见方案.md`、`raw/articles/RTP-LLM：阿里开源工业级 LLM 推理引擎，模型加载提速 6.3 倍、TTFT 降低 37%，吞吐量领先 vLLM 与 SGLang！.md`、`raw/articles/Gemma 4：Drafter 详解.md`
- 更新概念页：`wiki/concepts/Multi-Token Prediction.md`
- 本次 query 将 MTP 层结构拆成三种口径：最简单的共享 trunk + 多个未来 token 输出头；DeepSeek-V3 风格的顺序 MTP module，融合主模型 hidden state 与未来 token embedding 后经过轻量 block 逐步预测更远 token；Gemma 4 风格的独立小 drafter，复用 target activation / KV cache 后生成 draft token。

## [2026-06-09] query | MTP 与投机解码关系

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Speculative Decoding.md`、`wiki/concepts/MTP Drafter.md`
- 读取来源页：`wiki/sources/Gemma 4：Drafter 详解.md`、`wiki/sources/LLM提速利器：投机推理的原理与常见方案.md`、`wiki/sources/RTP-LLM.md`
- 创建概念页：`wiki/concepts/Multi-Token Prediction.md`
- 更新概念页：`wiki/concepts/Speculative Decoding.md`、`wiki/concepts/MTP Drafter.md`
- 本次 query 澄清：MTP 是 `Multi-Token Prediction`，可以是训练辅助目标，也可以在推理中作为候选 token 生成机制；投机解码是“候选生成 + target 验证 + 接受/回退”的执行框架。MTP 只有被用作 drafter/proposer 并交给 target model 验证时，才是投机解码的一种实现路线。

## [2026-06-09] query | EP all-to-all 与 EP/TP size 关系

- 读取索引页：`wiki/index.md`
- 读取/更新概念页：`wiki/concepts/Expert Parallelism.md`、`wiki/concepts/集合通信.md`
- 参考概念页：`wiki/concepts/MoE.md`、`wiki/concepts/Tensor Parallelism.md`、`wiki/concepts/DP Attention.md`
- 本次 query 澄清：EP 使用 all-to-all 是因为每个 rank 的 token 会根据 router 被动态发送到任意 expert 所在 rank，通信是多对多 activation dispatch / combine，而不是 TP 式 partial result all-reduce。`EP size` 和 `TP size` 切分维度不同，不要求相等；单机单副本 serving 中 `EP=TP` 常见，但在 DP/DPA、跨节点或 hybrid MoE parallel 下会分开设计。

## [2026-06-09] query | SP 下 attention 依赖如何保证

- 读取/更新概念页：`wiki/concepts/Sequence Parallelism.md`
- 本次 query 澄清：SP 将部分激活沿 token/sequence 维保存为 `[T/P,H]`，但标准 causal self-attention 不能只看本地 token shard；常见 TP+SP 实现会在 attention 或 column-parallel linear 前 all-gather 成完整 `[T,H]` / 完整 K/V 视图，做完 attention output projection 后再 reduce-scatter 回 `[T/P,H]`。真正把 attention 序列维本身分布式计算并交换 K/V block 的方案应和 Context Parallelism、Ring Attention、Ulysses 等区分。

## [2026-06-09] query | TP 中输入激活是否复制

- 更新概念页：`wiki/concepts/Tensor Parallelism.md`
- 本次 query 澄清：常见 Megatron-style TP 里，Transformer 子层边界处的 hidden activation 往往是每个 TP rank 都有完整 `[tokens, hidden]`；TP 主要切 column/row linear 权重，并让中间激活 shard 化，row-parallel 输出后再 all-reduce。若启用 Sequence Parallelism，激活更多沿 token/sequence 维切为 `[tokens_local, hidden]`，而不是所有子层统一把 hidden 维切成 `[tokens, hidden/TP]`。

## [2026-06-09] query | 大 EP / Expert Parallelism 解释

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/MoE.md`、`wiki/concepts/Tensor Parallelism.md`、`wiki/concepts/DP Attention.md`
- 创建概念页：`wiki/concepts/Expert Parallelism.md`
- 更新概念页：`wiki/concepts/MoE.md`、`wiki/concepts/Tensor Parallelism.md`、`wiki/concepts/DP Attention.md`
- 本次 query 将口语里的“大 EP”解释为 `EP size` 开得较大的 MoE 专家并行：expert 按 expert 维度分散到很多 GPU / rank，token 根据 router 结果跨卡发送到 expert 所在设备计算；收益是分散 expert 权重与提升系统吞吐，代价是 token dispatch / all-to-all、负载不均、小 batch GEMM 和跨节点 tail effect。

## [2026-06-09] query | Qwen3.5-MoE TP8 shape

- 读取/更新概念页：`wiki/concepts/MoE.md`
- 本次 query 补充 Qwen3.5-MoE 在 `TP=8` 下的 MoE shape 账本：router 在该口径下按复制理解，输出 `[T,512]` 与 top-k `[T,10]`；routed expert 的 `gate_up_proj [512,2048,4096]` 按 intermediate 维切为每 rank `[512,256,4096]`，`down_proj [512,4096,1024]` 切为 `[512,4096,128]`；每个 expert 的本地中间激活为 `[n_e,128]`，down 后得到 `[n_e,4096]` partial，再经 TP all-reduce 合成完整 `[n_e,4096]`。

## [2026-06-09] query | Qwen3.5-MoE 计算流程与 hidden/intermediate 维度

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/MoE.md`
- 参考官方 Hugging Face Transformers `qwen3_5_moe` 配置与建模源码
- 更新概念页：`wiki/concepts/MoE.md`
- 本次 query 将给定 `Qwen3_5MoeForConditionalGeneration` config 落到 shape 口径：文本侧 `hidden_size=4096` 是 decoder 主干宽度；`moe_intermediate_size=1024` 是每个 routed expert 的 SwiGLU 中间宽度；`num_experts=512` 且 `num_experts_per_tok=10` 表示每个 token 只路由到 10 个专家，并额外经过 shared expert。视觉侧 `hidden_size=1152`、`intermediate_size=4304` 属于 vision encoder 内部，`out_hidden_size=4096` 用于对齐文本主干。

## [2026-06-09] ingest | RTP-LLM：阿里开源工业级 LLM 推理引擎，模型加载提速 6.3 倍、TTFT 降低 37%，吞吐量领先 vLLM 与 SGLang！

- 读取原始资料：`raw/articles/RTP-LLM：阿里开源工业级 LLM 推理引擎，模型加载提速 6.3 倍、TTFT 降低 37%，吞吐量领先 vLLM 与 SGLang！.md`
- 创建来源页：`wiki/sources/RTP-LLM.md`
- 创建实体页：`wiki/entities/RTP-LLM.md`
- 创建实体页：`wiki/entities/阿里巴巴.md`
- 创建概念页：`wiki/concepts/分层 KV Cache.md`
- 更新概念页：`wiki/concepts/PD分离.md`、`wiki/concepts/KV Cache.md`、`wiki/concepts/Prefix Caching.md`、`wiki/concepts/缓存感知路由.md`、`wiki/concepts/Speculative Decoding.md`、`wiki/concepts/混合精度训练与推理.md`
- 更新实体页：`wiki/entities/vLLM.md`、`wiki/entities/SGLang.md`、`wiki/entities/Qwen VL.md`
- 本次 ingest 将 RTP-LLM 整理为生产级推理 serving 系统案例：其价值点不只是某个 kernel，而是文件顺序驱动模型加载、中心化调度、跨 worker 前缀匹配、分层 KV cache、PD 分离、模块化推测解码、KV cache 量化和多模态 EPD 解耦的组合。文中的性能数字均保留为来源声称，后续引用需补模型、硬件、并行配置、流量形态和框架版本。

## [2026-06-08] ingest | 还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速

- 读取原始资料：`raw/articles/还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速.md`
- 创建来源页：`wiki/sources/还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速.md`
- 创建概念页：`wiki/concepts/CODA.md`
- 更新概念页：`wiki/concepts/算子融合.md`、`wiki/concepts/CUDA Kernel.md`、`wiki/concepts/RMSNorm.md`、`wiki/concepts/Triton.md`、`wiki/concepts/Torch Compile.md`
- 本次 ingest 将 CODA 整理为“围绕 GEMM epilogue 的代数重写 + 融合抽象”：它不是简单把相邻 PyTorch op 拼大，而是利用 Transformer 中部分 memory-bound 小操作的代数性质，将 RMSNorm、SwiGLU、RoPE、残差、交叉熵等重写进 `GEMM + epilogue` 程序，减少中间张量 HBM 往返。文章中的 `1.6x-1.8x` backward 加速和 `5%-20%` Transformer 层前向加速均标记为待按原论文、代码仓库和 benchmark 配置核实。

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

## [2026-07-08] query | MoE 通信-计算重叠与 MegaKernel 技术谱系

- 讨论 5 份资料：DeepFusionKernel、Megakernel(HazyResearch)、Ladder-Residual、NVIDIA Wide-EP、UniEP
- 创建报告：`output/reports/MoE通信-计算重叠与MegaKernel技术谱系.md`
- 沉淀概念澄清：SwiGLU vs MoE、fused_moe 融合边界、通信-计算重叠的细粒度原理、Triton-distributed、UniEP 能否用于推理
- 未新建独立概念/实体页（报告内已列出后续 ingest 建议）；相关已有页：Megakernel / MoE / Expert Parallelism / 算子融合
- 待核实：各框架 fused_moe 融合边界、UniEP/Triton-distributed 晚于知识截止的 API 细节

## [2026-07-12] query | Prefill Attention 的 CUDA 并行映射

- 读取概念页：`FlashAttention`、`GPU执行模型`、`Online Softmax`、`Tiling`、`CUDA Kernel`
- 读取来源页与原始资料：`Flash Attention 详细解释推演与Pytorch代码实现`、`Stanford CS336 Lecture 5 - GPUs`
- 使用 Dao-AILab/flash-attention 官方 CUDA 源码交叉检查典型 forward grid 的 `Q tile × batch × head` 映射
- 创建报告：`output/reports/Prefill Attention 的 CUDA 并行映射.md`
- 更新概念页：`wiki/concepts/FlashAttention.md`，补充 block/warp/thread 层级、单 block 内 `QK → online softmax → PV` 数据流、SM 驻留与 split-KV 特例
- 未发现与现有 wiki 的直接冲突；显式标注 tile size、warp 分工和 split-KV 策略依赖具体版本与硬件

## [2026-07-13] query | `cu_seqlens: [2]` 的 request 含义

- 检查当前仓库，未找到该 JSON 字段的具体上下文
- 更新报告：`output/reports/Prefill Attention 的 CUDA 并行映射.md`，补充 varlen packed batch 中 `cu_seqlens` 的解读规则
- 标准 FlashAttention-style 约定下，`cu_seqlens` 应以 `0` 开头，request 数为 `len(cu_seqlens)-1`；单个 2-token request 应为 `[0, 2]`
- 字面 `[2]` 对标准格式来说不完整；只有在某框架明确省略起始 `0` 的日志约定下，才表示 1 个长度为 2 的 request

## [2026-07-25] ingest | vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署

- 读取原始资料：`raw/articles/vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署.md`
- 创建来源页：`wiki/sources/vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署.md`
- 创建概念页：`wiki/concepts/Attention-FFN 分离.md`，整理逐层数据流、hidden states 传输、与 PP/PD 的区别及与 EP/TP 的组合
- 创建实体页：`wiki/entities/vLLM AFD Plugin.md`
- 更新概念与实体页：`MoE`、`Expert Parallelism`、`Tensor Parallelism`、`流水线并行`、`PD分离`、`vLLM`、`NCCL`
- 未发现与现有 wiki 的直接冲突；benchmark 受模拟规模、强制均衡路由和裁剪模型限制

## [2026-07-25] ingest | NVIDIA 开源 NCCL Extensions：MoE EP 与跨 Mesh 权重重分片

- 读取原始资料：`raw/articles/NVIDIA 开源 NCCL Extensions：把 MoE 专家路由与跨 Mesh 权重重分片推进到 GPU 设备侧.md`
- 创建来源页：`wiki/sources/NVIDIA 开源 NCCL Extensions：把 MoE 专家路由与跨 Mesh 权重重分片推进到 GPU 设备侧.md`
- 创建实体页：`wiki/entities/NCCL Extensions.md`
- 创建概念页：`wiki/concepts/跨 Mesh 权重重分片.md`、`wiki/concepts/通信-计算重叠.md`
- 更新页面：`Expert Parallelism`、`MoE`、`集合通信`、`NCCL`，补充 nccl_ep 与 nccl_m2n 的定位和双向链接
- 讨论归纳：在线 RL 中 M2N 负责 Training -> Rollout 的策略权重同步，不负责轨迹回传或逐 token 中间状态
- 待核实：精确 API、CUDA/NCCL/硬件支持范围及 RING/DIRECT 性能以对应仓库 commit 为准

## [2026-07-25] ingest | SGLang 的 KDA/GDN 状态管理与 Prefix Cache

- 读取完整截图资料：`raw/articles/SGLang的KDA管理与Prefix Cache难题.md`（14 张正文图片）
- 创建来源页：`wiki/sources/SGLang的KDA管理与Prefix Cache难题.md`
- 创建概念页：`wiki/concepts/线性注意力递归状态.md`，区分 Conv State 与长期矩阵状态并整理 GDN 更新机制
- 创建概念页：`wiki/concepts/递归状态 Prefix Caching.md`，整理 checkpoint、共同恢复边界、重算与显存权衡
- 更新页面：`Chunked Gated Delta Rule`、`Prefix Caching`、`KV Cache`、`混合注意力`、`Speculative Decoding`、`SGLang`
- 机制归纳：GDN 的矩阵状态按 fast-weight/key-value 关联记忆解释；Conv State 为 causal depthwise convolution 保存短窗口
- 待核实：KDA 状态 S 的官方数学定义，以及 SGLang MambaPool/UnifiedRadixCache/暂存提交路径的具体源码版本

## [2026-07-25] ingest | Nvidia Rubin 架构分析预览

- 读取原始资料：`raw/articles/Nvidia Rubin架构分析预览.md`
- 创建来源页：`wiki/sources/Nvidia Rubin架构分析预览.md`
- 创建硬件实体页：`wiki/entities/NVIDIA Rubin.md`
- 更新概念页：`MoE`、`Programmatic Dependent Launch`、`GPU执行模型`、`CUDA内存层次`、`通信-计算重叠`、`算子融合`、`混合精度训练与推理`
- 重点整理 MoE 优化链：TMA runtime override、Expert 权重 L2 priority、SFU epilogue 与 counted dispatch/combine
- 未发现直接冲突；counted fabric 的 Rubin 产品叙事与 PTX 9.3 sm_100+ target 需要区分
- 待核实：PTX 9.4 最终语义、Rubin 实机性能、作者反推的频率与调度器实现

## [2026-07-25] ingest | 2026 年 MoE 架构关键变化：LatentMoE

- 读取完整截图资料：`raw/articles/2026 年MoE 架构正在发生一次关键变化.md`（9 张正文图片）
- 创建来源页：`wiki/sources/2026 年MoE 架构正在发生一次关键变化.md`
- 创建概念页：`wiki/concepts/LatentMoE.md`
- 更新页面：`MoE`、`Expert Parallelism`、`Sparsity Allocation`、`混合精度训练与推理`、`Moonshot AI`
- 讨论澄清：d 是模型主干 hidden size，ℓ 是 routed expert 潜在输入输出维度，m 是 expert intermediate size
- 机制归纳：潜在空间可按 ℓ/d 缩小 expert 权重主导项和 EP activation payload，但不等于端到端获得 d/ℓ 倍加速
- 待核实：LatentMoE 官方定义、投影共享方式、Nemotron 3 Super/Kimi K3 规格及 benchmark
