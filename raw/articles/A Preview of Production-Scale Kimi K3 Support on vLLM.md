---
title: "A Preview of Production-Scale Kimi K3 Support on vLLM"
source: "https://vllm.ai/blog/2026-07-22-kimi-k3-preview#the-hardest-part-prefix-caching-for-kda"
author:
  - "[[vLLM Team]]"
published: 2026-07-22
created: 2026-08-02
description: "A preview of production-scale Kimi K3 support in vLLM, including KDA-aware prefix caching, fused kernels, optimized MXFP4 MoE, multimodal integration, and initi"
tags:
  - "clippings"
---
Last week, Moonshot AI [introduced Kimi K3](https://www.kimi.com/blog/kimi-k3), a 2.8-trillion-parameter model with native vision support, a 1-million-token context window, Kimi Delta Attention (KDA), Attention Residuals (AttnRes), and a highly sparse Mixture-of-Experts architecture. The announcement immediately drew [global attention](https://x.com/Kimi_Moonshot/status/2077830229968683203), and the open-source community is extremely excited that open-weight models are advancing quickly to catch up with the best proprietary models.  
上周，Moonshot AI [发布了 Kimi K3](https://www.kimi.com/blog/kimi-k3) ，这是一个拥有 2.8 万亿参数的模型，具备原生视觉支持、100 万个标记的上下文窗口、Kimi Delta Attention (KDA)、Attention Residuals (AttnRes) 以及高度稀疏的混合专家架构。该发布立即引起了 [全球关注](https://x.com/Kimi_Moonshot/status/2077830229968683203) ，开源社区也对此感到无比兴奋，因为开源权重模型正在迅速发展，力图赶上最优秀的专有模型。

Moonshot AI has announced that the full model weights will be released by July 27, 2026. In the meantime, vLLM, Moonshot AI, NVIDIA, AMD, and the broader community are working through the final integration and validation so the open-source community can serve Kimi K3 from day 0.  
Moonshot AI 宣布将于 2026 年 7 月 27 日发布完整的模型权重。与此同时，vLLM、Moonshot AI、NVIDIA、AMD 和更广泛的社区正在努力完成最终的集成和验证，以便开源社区能够从一开始就为 Kimi K3 提供服务。

This post is a preview and performance optimization is ongoing, but the core model path, KDA-aware prefix caching, multimodal integration, tool calling parsers, and hardware-specific optimizations are already taking shape. Selected trusted partners, approved by both Moonshot AI and the vLLM/Inferact team, have also begun deployment validation using the same code that is being prepared for open source.  
本文仅为预览，性能优化仍在进行中，但核心模型路径、KDA 感知前缀缓存、多模态集成、工具调用解析器以及硬件特定优化已初具雏形。经 Moonshot AI 和 vLLM/Inferact 团队共同认可的精选合作伙伴也已开始使用即将开源的相同代码进行部署验证。

As stated in the announcement blog, KDA poses new challenges for conventional prefix caching, and the Moonshot AI team has contributed a corresponding implementation to the vLLM project, to be released alongside the model weights. We will dedicate a future blog post to explaining the design.  
正如公告博客中所述，KDA 给传统的前缀缓存带来了新的挑战，而 Moonshot AI 团队已为 vLLM 项目贡献了相应的实现，该实现将与模型权重一同发布。我们将在后续的博客文章中详细解释其设计。

## TL;DR 太长不看

- **Day-0 open-source serving:** vLLM is preparing model implementation, Docker images, deployment recipes, and production validation for the Kimi K3 weight release.  
	**第 0 天开源服务：** vLLM 正在为 Kimi K3 权重发布准备模型实现、Docker 镜像、部署方案和生产验证。
- **A new hybrid architecture:** Kimi K3 combines KDA-dominant linear attention with periodic full-attention layers, AttnRes across depth, Stable LatentMoE, and native vision support.  
	**一种新的混合架构：** Kimi K3 将 KDA 主导的线性注意力与周期性全注意力层、跨深度的 AttnRes、稳定的 LatentMoE 和原生视觉支持相结合。
- **Prefix caching required core changes:** vLLM now separates the physical KDA state-block size from prefix-match granularity, enabling useful partial prefix-cache hits without storing recurrent state at every small attention block.  
	**前缀缓存需要核心更改：** vLLM 现在将物理 KDA 状态块大小与前缀匹配粒度分开，从而实现有用的部分前缀缓存命中，而无需在每个小的注意力块中存储循环状态。
- **Kernel work across the stack:** the release branch includes FlashKDA integration, fused KDA decode, fused KDA projections and convolution, fused AttnRes, reimplemented MLA module, SiTU-enabled MXFP4 MoE execution, and optimized expert routing.  
	**整个堆栈的内核工作：** 发布分支包括 FlashKDA 集成、融合 KDA 解码、融合 KDA 投影和卷积、融合 AttnRes、重新实现的 MLA 模块、支持 SiTU 的 MXFP4 MoE 执行以及优化的专家路由。
- **NVIDIA and AMD support:** NVIDIA-specific kernels are under final tuning, while an initial AMD implementation with a FlyDSL MoE kernel is already in place and moving through broader validation.  
	**NVIDIA 和 AMD 支持：** NVIDIA 专用内核正在进行最后的调整，而采用 FlyDSL MoE 内核的 AMD 初始实现已经到位，并正在进行更广泛的验证。

## Kimi K3 at a GlanceKimi K3 概览

Kimi K3 is not a larger version of Kimi K2. Kimi K3 changes the serving problem in several dimensions at once.  
Kimi K3 不是 Kimi K2 的放大版。Kimi K3 从多个维度同时改变了发球问题。

| Property 财产 | Kimi K3 configuration Kimi K3 配置 | Serving implication 服务含义 |
| --- | --- | --- |
| **Model scale 模型比例** | **2.8T parameters 2.8T 参数** | Requires large-scale expert parallelism and high-bandwidth accelerator domains   需要大规模专家并行处理和高带宽加速器域 |
| **Context length 上下文长度** | **1M tokens 100万个代币** | Makes cache capacity, prefix reuse, chunked prefill, and prefill/decode disaggregation first-order concerns   将缓存容量、前缀重用、分块预填充和预填充/解码分解作为首要考虑因素。 |
| **Attention 注意力** | **Hybrid KDA and full attention   混合 KDA 和全神贯注** | Requires both recurrent state caches and paged KV caches to advance on exactly the same logical prefix   需要循环状态缓存和分页键值缓存同时在同一个逻辑前缀上进行推进 |
| **Depth 深度** | **Attention Residual 注意力残差** | Adds cross-layer representation reads and writes that need dedicated kernels   增加了需要专用内核的跨层表示读取和写入操作。 |
| **MoE 教育部** | **896 routed experts, 16 active per token, plus shared experts   896 位路由专家，每个令牌 16 位活跃专家，外加共享专家** | Makes routing, dispatch, load balance, and MoE kernels central to end-to-end performance   使路由、调度、负载均衡和 MoE 内核成为端到端性能的核心。 |
| **Quantization 量子化** | **MXFP4 weights in the provided release configuration   提供的发布配置中的 MXFP4 权重** | Needs an efficient FP4 MoE path with Kimi K3’s SiTU activation   需要一条高效的 FP4 MoE 路径，并激活 Kimi K3 的 SiTU。 |
| **Multimodality 多模态** | **Native vision with a vision tower   本土视野与愿景塔** | Requires multimodal preprocessing (image-only) and a robust vision parallelism strategy   需要多模态预处理（仅图像）和稳健的视觉并行策略 |

For inference systems, each of these choices moves cost somewhere new. KDA reduces the need to retain a conventional KV pair for every past token, but introduces a large recurrent state. AttnRes reduces the limitations of a single residual stream, but creates additional cross-layer memory traffic. Extreme MoE sparsity avoids activating all 2.8T parameters for every token, but raises the stakes for routing and communication. vLLM's job is to make all of these pieces work together behind one familiar serving API.  
对于推理系统而言，每一种选择都会将成本转移到新的位置。KDA 减少了为每个历史令牌保留传统键值对的需求，但引入了一个庞大的循环状态。AttnRes 突破了单个残差流的限制，但增加了跨层内存流量。极高的 MoE 稀疏性避免了为每个令牌激活全部 2.8T 个参数，但提高了路由和通信的难度。vLLM 的任务是让所有这些组件在一个熟悉的 API 背后协同工作。

## A Collaboration Built Over Multiple Kimi Generations历经数代 Kimi 家族成员的合作

Kimi K3 continues a long collaboration between Moonshot AI and the vLLM community.  
Kimi K3 延续了 Moonshot AI 与 vLLM 社区之间的长期合作。

- At [GOSIM 2024](https://china2024.gosim.org/schedules/vllm-in-moonshot.html), Moonshot AI engineers presented how vLLM was used at scale inside Moonshot AI and discussed the vLLM + Mooncake prefill/decode-disaggregated architecture.  
	在 [GOSIM 2024](https://china2024.gosim.org/schedules/vllm-in-moonshot.html) 上，Moonshot AI 的工程师们展示了 vLLM 在 Moonshot AI 内部的大规模应用，并讨论了 vLLM + Mooncake 预填充/解码-解耦架构。
- Moonshot AI later shared Kimi K2 training and inference practices at the [vLLM Beijing Meetup](https://pytorch.org/blog/vllm-beijing-meetup-advancing-large-scale-llm-deployment/), including operating under strict SLOs while serving online traffic and supporting reinforcement-learning workloads.  
	Moonshot AI 后来在 [vLLM 北京聚会](https://pytorch.org/blog/vllm-beijing-meetup-advancing-large-scale-llm-deployment/) 上分享了 Kimi K2 的训练和推理实践，包括在严格的 SLO 下运行，同时服务于在线流量并支持强化学习工作负载。
- vLLM has been a day-0 launch partner for Kimi K2, Kimi K2-Thinking, Kimi K2.5, Kimi Linear, and so on.  
	vLLM 一直是 Kimi K2、Kimi K2-Thinking、Kimi K2.5、Kimi Linear 等产品的首发合作伙伴。
- vLLM has deep technical collaboration with Moonshot AI engineers, including [Kimi K2 tool-calling accuracy](https://vllm.ai/blog/Kimi-K2-Accuracy) for correctness, [improved CUDA debugging](https://vllm.ai/blog/improved-cuda-debugging) for development, [decode context parallelism](https://github.com/vllm-project/vllm/pull/23734), Mooncake-based PD disaggregation, and large-scale performance validation. Kimi K2.5 has also appeared in public [InferenceX serving results](https://inferencex.semianalysis.com/inference?g_rundate=2026-04-07&g_model=Kimi-K2.5&g_runid=24100518225&i_gpus=gb200_dynamo-vllm&i_dstart=2026-04-07&i_dend=2026-04-07).  
	vLLM 与 Moonshot AI 工程师开展了深入的技术合作，包括 [提升 Kimi K2 工具调用准确性](https://vllm.ai/blog/Kimi-K2-Accuracy) 以确保正确性、 [改进 CUDA 调试](https://vllm.ai/blog/improved-cuda-debugging) 以方便开发、 [实现解码上下文并行性](https://github.com/vllm-project/vllm/pull/23734) 、基于 Mooncake 的 PD 分解以及大规模性能验证。Kimi K2.5 也已在公开的 [InferenceX 服务结果](https://inferencex.semianalysis.com/inference?g_rundate=2026-04-07&g_model=Kimi-K2.5&g_runid=24100518225&i_gpus=gb200_dynamo-vllm&i_dstart=2026-04-07&i_dend=2026-04-07) 中亮相。

That history matters. Day-0 support is rarely one pull request written after a release announcement. It comes from model and inference teams sharing architecture details early, testing real checkpoints under realistic parallelism, identifying gaps in the serving engine, and upstreaming improvements that remain useful after one launch. **vLLM is proud to be a long-term partner of Moonshot AI and a popular inference engine for Kimi-series models.**  
历史至关重要。首日支持并非仅仅指发布公告后提交的一个拉取请求。它源于模型和推理团队尽早分享架构细节，在真实的并行环境下测试实际检查点，识别服务引擎的不足，并将改进成果提交到上游，这些改进在一次发布后仍然有效。vLLM **很荣幸成为 Moonshot AI 的长期合作伙伴，并为 Kimi 系列模型提供广受欢迎的推理引擎。**

Now, let’s dive into one of the most interesting technical challenges we ran into.  
现在，让我们深入探讨一下我们遇到的最有趣的技术挑战之一。

## The Hardest Part: Prefix Caching for KDA最难的部分：KDA 的前缀缓存

Conventional full attention and KDA remember a prefix in very different ways.  
传统全神贯注和 KDA 记忆前缀的方式截然不同。

In full attention, a prefix is represented by per-token key and value vectors. vLLM stores those vectors in paged blocks, hashes complete token blocks, and can reuse a matching sequence of blocks for another request.  
在完全关注的情况下，前缀由每个标记的键值向量表示。vLLM 将这些向量存储在分页块中，对完整的标记块进行哈希处理，并且可以为另一个请求重用匹配的块序列。

KDA is recurrent. Instead of retaining a conventional KV pair for every token, each KDA layer advances a matrix-like recurrent state, together with a short convolution state. To resume from a cached prefix, the engine needs the KDA state *at the exact prefix boundary*. Replaying an earlier state to reach that boundary would erase much of the benefit of prefix caching.  
KDA 是循环的。它不像传统算法那样为每个词元保留一个键值对，而是每个 KDA 层都递进一个类似矩阵的循环状态，以及一个简短的卷积状态。为了从缓存的前缀恢复执行，引擎需要 *精确到达前缀边界的* KDA 状态。如果为了到达边界而重放之前的状态，就会大大降低前缀缓存的优势。

![How conventional attention and KDA represent cached prefixes](https://vllm.ai/blog-assets/figures/2026-07-22-kimi-k3-preview/kda-prefix-state.png)

How conventional attention and KDA represent cached prefixes 传统注意力机制和 KDA 如何表示缓存前缀

The straightforward solution—store KDA state at every small attention-cache boundary—is too expensive. A KDA state is much larger than one ordinary token's KV entry, so implementations use a relatively large physical state block to amortize storage. Before the current work, that physical block size also constrained where a prefix-cache hit could land. With a multi-thousand-token state block, two requests sharing almost the entire prompt could still miss the reusable prefix because their common boundary did not fill the same physical block.  
最直接的解决方案——在每个小的注意力缓存边界存储 KDA 状态——成本太高。KDA 状态远大于一个普通 token 的 KV 条目，因此实现通常使用相对较大的物理状态块来分摊存储成本。在当前工作之前，这种物理块大小也限制了前缀缓存命中的位置。即使状态块包含数千个 token，两个几乎共享整个提示符的请求仍然可能错过可重用的前缀，因为它们的公共边界并未填满同一个物理块。

The new vLLM design separates three concepts that used to move together:  
新的 vLLM 设计将以前一起发展的三个概念分开：

- **Physical block size:** how KDA state and full-attention KV are allocated on the GPU.  
	**物理块大小：** KDA 状态和完全注意力 KV 在 GPU 上的分配方式。
- **Scheduler alignment:** where execution must stop so all cache groups remain consistent.  
	**调度器对齐：** 执行必须停止的位置，以确保所有缓存组保持一致。
- **Prefix-match unit:** the finer token interval at which a shared prefix is hashed and may be matched.  
	**前缀匹配单元：** 用于对共享前缀进行哈希处理并进行匹配的更精细的标记间隔。
![Fine-grained prefix matching inside a larger physical KDA state block](https://vllm.ai/blog-assets/figures/2026-07-22-kimi-k3-preview/fine-grained-prefix-cache.png)

Fine-grained prefix matching inside a larger physical KDA state block 在较大的物理 KDA 状态块内进行细粒度前缀匹配

This lets vLLM register a valid KDA state at a fine-grained boundary inside a larger physical state block. When a later request hits that partial block, the cached state is copied into a private destination before the request extends it. This copy-on-write rule preserves the shared cached prefix while allowing the new request to continue generation safely.  
这使得 vLLM 能够在较大的物理状态块内，以细粒度的边界注册有效的 KDA 状态。当后续请求到达该部分状态块时，缓存的状态会在请求扩展之前被复制到私有目标位置。这种写时复制规则既保留了共享的缓存前缀，又允许新请求安全地继续生成状态。

The implementation also handles details that are easy to miss:  
该实现方案还处理了一些容易被忽略的细节：

- The scheduler stops at the right block and hash boundaries so the recurrent state being registered really corresponds to the advertised token prefix.  
	调度器会在正确的区块和哈希边界处停止，因此注册的循环状态确实与公布的令牌前缀相对应。
- Full-attention and KDA cache groups agree on one `num_computed_tokens`, even though their physical block sizes differ.  
	即使物理块大小不同，全注意力缓存组和 KDA 缓存组也对 `num_computed_tokens` 达成一致。
- Partial cache entries use chained, fine-grained hashes so a boundary identifies the entire prefix, not only the tail tokens.  
	部分缓存条目使用链式细粒度哈希，因此边界标识整个前缀，而不仅仅是尾部标记。
- Same-step reuse is deferred until the state copy is safe, avoiding races between cache registration and extension.  
	同步骤重用被推迟到状态副本安全之后，从而避免缓存注册和扩展之间的竞争。
- Cache transfer and disaggregated prefill/decode paths can carry the same logical prefix across workers.  
	缓存传输和解耦的预填充/解码路径可以在工作进程之间携带相同的逻辑前缀。

This work was motivated by Kimi K3 and many other hybrid attention models, but it is core vLLM infrastructure rather than a model-specific shortcut. The vLLM team and the Moonshot AI team collaborated deeply on the design. The two teams will publish a separate post with the design, invariants, and benchmarks in more detail.  
这项工作的灵感来源于 Kimi K3 和许多其他混合注意力模型，但它是 vLLM 的核心基础设施，而非特定模型的捷径。vLLM 团队和 Moonshot AI 团队在设计上进行了深入合作。这两个团队将另行发布一篇博文，更详细地介绍设计、不变量和基准测试。

## Performance Work: Removing the New Bottlenecks绩效工作：消除新的瓶颈

Our current progress can be summarized into this table:  
我们目前的进展可以总结在下表中：

| Area 区域 | Current status 当前状态 |
| --- | --- |
| **Model and configuration 模型和配置** | Kimi K3 language and vision model definitions are integrated, with separate **NVIDIA** and **AMD** implementations where hardware paths differ   Kimi K3 语言和视觉模型定义已集成，但针对不同的硬件路径， **NVIDIA** 和 **AMD** 分别进行了单独的实现。 |
| **Optimized MLA module for native PD disaggregation deployment   针对原生 PD 解耦部署优化的 MLA 模块** | Optimized MLA module with manual kernel fusion and separate prefill/decode paths. Gate projection runs in parallel with attention, with multi-stream support in decode and a fused epilogue in prefill—highly optimized for PD disaggregation deployment.   优化的 MLA 模块，支持手动内核融合和独立的预填充/解码路径。门投影与注意力机制并行运行，解码时支持多流，预填充时采用融合尾声——针对 PD 解耦部署进行了高度优化。 |
| **Serving semantics 服务语义** | Kimi K3 chat rendering, tokenizer integration, streaming parsing, tool calls, reasoning output, and structured-output paths are implemented and under **final end-to-end validation**   Kimi K3 的聊天渲染、分词器集成、流式解析、工具调用、推理输出和结构化输出路径均已实现，目前正在进行 **最终的端到端验证。** |
| **KDA prefill KDA 预填充** | FlashKDA and Triton paths are integrated; final backend selection and numerical validation are **in progress**   FlashKDA 和 Triton 路径已集成；最终后端选择和数值验证 **正在进行中。** |
| **KDA decode KDA 解码** | A fused **NVIDIA** decode kernel covering convolution, the recurrent KDA update, gating, and normalization is integrated, with portable fallback paths retained   集成了一个融合的 **NVIDIA** 解码内核，涵盖卷积、循环 KDA 更新、门控和归一化，并保留了可移植的回退路径。 |
| **Prefix caching 前缀缓存** | Fine-grained partial prefix hits for hybrid full-attention + recurrent-state caches are integrated; disaggregated and offload scenarios are **being validated**   混合型全注意力机制+循环状态缓存的细粒度部分前缀命中已集成；正在 **验证** 解耦和卸载场景。 |
| **Attention Residuals 注意力残差** | Triton and **NVIDIA** kernels are integrated, including fusion of residual addition and output RMSNorm on supported shapes   Triton 和 **NVIDIA** 内核已集成，包括对支持的形状进行残差加法融合和输出 RMSNorm。 |
| **MoE 教育部** | Kimi K3’s **SiTU** activation is wired into **MXFP4 TRTLLM-Gen** and **DeepGEMM** paths; optimized grouped top-k routing is integrated. **AMD** implements FlyDSL’s **MLIR** kernel stack with hardware-tuned **A16W4/A8W4** fused operators and **SiTU** activation   Kimi K3 的 **SiTU** 激活功能集成到 **MXFP4 TRTLLM-Gen** 和 **DeepGEMM** 路径中；并集成了优化的分组 top-k 路由。AMD 实现了 **FlyDSL** 的 **MLIR** 内核协议栈，并采用了硬件优化的 **A16W4/A8W4** 融合运算符和 **SiTU** 激活功能。 |
| **Production stack 生产环境堆栈** | Non-disaggregated serving is working; Dynamo + vLLM + Mooncake disaggregated serving, expert parallelism, and vendor verification are in the **final validation loop**   非解耦式服务运行正常；Dynamo + vLLM + Mooncake 解耦式服务、专家并行处理和供应商验证正处于 **最终验证循环** 中。 |

Kimi K3 changes the hot path, so the team has optimized more than the attention kernel itself. Below are details of the progress in each area.  
Kimi K3 改变了热路径，因此团队优化的内容远不止注意力内核本身。以下是各个领域进展的详细信息。

### KDA prefill and decodeKDA 预填充和解码

The prefill path integrates FlashKDA and Flash Linear Attention (FLA). Around the core recurrence, vLLM fuses the input projections and causal convolution, and gathers initial recurrent states in one operation.  
预填充路径集成了 FlashKDA 和 Flash Linear Attention (FLA)。围绕核心循环，vLLM 融合了输入投影和因果卷积，并在一次操作中收集初始循环状态。

Decode uses a fused NVIDIA kernel on supported architectures and shapes. Instead of launching separate operations for the short convolution, KDA state update, output gate, and normalization for every generated token, the fused path performs them together. This is especially important because Kimi K3 contains many KDA layers; a small per-layer launch or memory penalty quickly becomes a large TPOT penalty.  
在支持的架构和形状上，解码过程使用融合的 NVIDIA 内核。与为每个生成的 token 单独执行短卷积、KDA 状态更新、输出门和归一化等操作不同，融合路径会将这些操作一起执行。这一点尤为重要，因为 Kimi K3 包含许多 KDA 层；即使每个层启动操作或内存开销很小，也会迅速累积成巨大的 TPOT 开销。

### Attention Residuals 注意力残差

AttnRes retrieves from representations written by earlier layer blocks rather than relying on only one uniformly accumulated residual stream. A naive implementation creates extra reads, writes, reductions, and normalization launches throughout the 93-layer network.  
AttnRes 从先前层块写入的表示中检索数据，而不是仅依赖于一个均匀累积的残差流。一个简单的实现会在整个 93 层网络中产生额外的读取、写入、归约和归一化操作。

The release branch includes a Triton implementation and an NVIDIA kernel that fuse residual update, AttnRes mixing, and output RMSNorm for supported cases. Sequence-parallel work also shards the attention-residual traffic across ranks. Early kernel-level results are encouraging, while end-to-end gains are still being measured across prefill lengths and parallel configurations.  
发布分支包含一个 Triton 实现和一个 NVIDIA 内核，它们融合了残差更新、AttnRes 混合以及针对支持情况的输出 RMSNorm。序列并行工作还将注意力残差流量分片到各个进程。早期内核级结果令人鼓舞，而端到端性能提升仍在针对不同的预填充长度和并行配置进行评估。

### Optimized MLA module for native PD disaggregation deployment针对原生 PD 解耦部署优化的 MLA 模块

Kimi K3 still uses MLA attention every four layers. In the previous model, vLLM relied heavily on a `torch.compile` custom-fusion path to map small kernels into fused kernels, which slowed startup and still left many kernels unfused. In this release, we implement a new MLA module that fuses these kernels manually. MLA also requires different kernel launch orders for prefill and decode, so we implement two code paths with different fusion patterns, specialized for PD-disaggregated deployment. Furthermore, Kimi K3 introduces a gate projection that can execute in parallel with the main attention path. We optionally add multi-stream support for the gate projection in the decode path, while in the prefill path—where multi-stream overlap is not optimal—we fuse the elementwise multiply and sigmoid into the gate-projection epilogue.  
Kimi K3 仍然每四层使用一次 MLA 注意力机制。在之前的模型中，vLLM 严重依赖于 `torch.compile` 自定义融合路径将小内核映射到融合内核，这导致启动速度变慢，并且仍然有很多内核未融合。在此版本中，我们实现了一个新的 MLA 模块，可以手动融合这些内核。MLA 还要求预填充和解码使用不同的内核启动顺序，因此我们实现了两条具有不同融合模式的代码路径，专门用于 PD 解耦部署。此外，Kimi K3 引入了一个门投影，它可以与主注意力路径并行执行。我们为解码路径中的门投影添加了可选的多流支持，而在预填充路径中（多流重叠并非最佳选择），我们将逐元素乘法和 sigmoid 函数融合到门投影的尾声部分。

### MXFP4 MoE

Kimi K3's release configuration uses MXFP4 weights and the SiTU activation. Before this work, the MXFP4 TRTLLM-Gen path did not support SiTU and would fall back to a slower implementation. vLLM now maps Kimi K3's SiTU parameters into the optimized FP4 expert path and also handles large token-by-top-k launch grids by safely chunking the workload.  
Kimi K3 的发布配置使用 MXFP4 权重和 SiTU 激活。在此之前，MXFP4 TRTLLM-Gen 路径不支持 SiTU，会回退到速度较慢的实现。vLLM 现在将 Kimi K3 的 SiTU 参数映射到优化的 FP4 专家路径，并通过安全地分块工作负载来处理大型的基于 token-by-top-k 的启动网格。

This has already been validated on a 16-GPU DP16+EP16 configuration, where all ranks selected the optimized MXFP4 backend and passed correctness checks.  
这已经在 16-GPU DP16+EP16 配置上得到了验证，其中所有进程都选择了优化的 MXFP4 后端并通过了正确性检查。

On the AMD side, Kimi K3 MoE is supported on FlyDSL's MLIR Python kernel stack. This includes hardware-tuned A16W4/A8W4 quantized fused operators and a SiTU activation implementation, all built on FlyDSL's modular abstractions.  
在 AMD 方面，FlyDSL 的 MLIR Python 内核栈支持 Kimi K3 MoE。这包括硬件优化的 A16W4/A8W4 量化融合算子和 SiTU 激活实现，所有这些都基于 FlyDSL 的模块化抽象构建。

## What to Expect on Open-Source Day开源日有哪些值得期待的事

The planned day-0 package includes:  
计划中的第一天服务包包括：

- vLLM model, parser, cache, and kernel integration;  
	vLLM 模型、解析器、缓存和内核集成；
- initial open-source Docker images;  
	初始开源 Docker 镜像；
- validated launch recipes for NVIDIA configurations;  
	针对 NVIDIA 配置的已验证启动方案；
- an initial AMD path with FlyDSL MoE kernel, with more ROCm tuning to follow;  
	最初采用 AMD FlyDSL MoE 内核，后续将进行更多 ROCm 调优；
- multimodal, tool-use, reasoning, and structured-output examples;  
	多模态、工具使用、推理和结构化输出示例；
- initial performance results.  
	初步性能结果。

Trusted deployment partners are already exercising the release candidate under a dual-approval process from Moonshot AI and vLLM/Inferact. This provides real production feedback without distributing prerelease model artifacts broadly. It also gives us a chance to test the complete serving system—frontend semantics, batching, cache transfer, expert parallelism, observability, and failure handling—not only isolated kernels.  
受信任的部署合作伙伴已在 Moonshot AI 和 vLLM/Inferact 的双重审批流程下对候选版本进行测试。这无需广泛分发预发布模型工件，即可获得真实的生产反馈。此外，它还使我们有机会测试完整的服务系统——包括前端语义、批处理、缓存传输、专家并行处理、可观测性和故障处理——而不仅仅是孤立的内核。

## Acknowledgements 致谢

Kimi K3 day-0 support is a joint effort across the model vendor, inference engine, and hardware communities.  
Kimi K3 的第一天支持是模型供应商、推理引擎和硬件社区共同努力的结果。

We thank the **Moonshot AI team** for creating Kimi K3, sharing architecture details ahead of the weight release, contributing the initial model integration and KDA prefix-caching work, and collaborating closely on correctness and production validation.  
我们感谢 **Moonshot AI 团队** 创建了 Kimi K3，在权重发布之前分享了架构细节，贡献了初始模型集成和 KDA 前缀缓存工作，并在正确性和生产验证方面密切合作。

We thank the **Inferact team** for integrating the model into vLLM, extending the core cache manager for partial hybrid prefix hits, implementing serving semantics and multimodal support, building deployment recipes, and driving end-to-end performance optimization.  
我们感谢 **Inferact 团队** 将该模型集成到 vLLM 中，扩展了核心缓存管理器以支持部分混合前缀命中，实现了服务语义和多模态支持，构建了部署方案，并推动了端到端的性能优化。

We thank the **NVIDIA team** for KDA decode and Attention Residual kernels, MXFP4 MoE collaboration, and performance work across the board.  
我们感谢 **NVIDIA 团队** 提供的 KDA 解码和注意力残差内核、MXFP4 MoE 合作以及全面的性能改进工作。

We thank the **AMD team** for initial day-0 ROCm support and for continuing to expand Kimi K3 across AMD GPUs.  
我们感谢 **AMD 团队** 在 ROCm 推出之初就给予的支持，并继续在 AMD GPU 上扩展 Kimi K3。

Most importantly, we thank the broader open-source community for the anticipation, testing, and feedback already surrounding Kimi K3. We look forward to putting the weights and the inference engine support in your hands.  
最重要的是，我们要感谢广大开源社区对 Kimi K3 的期待、测试和反馈。我们期待着尽快将权重和推理引擎支持交付到你们手中。

## One More Thing: Why the Announcement and Open-Source Release Are Separated还有一件事：为什么公告和开源版本发布要分开进行？

Kimi K3 also features a release process that we hope more model vendors will consider: announce the model first, then release the weights and inference engine support later.  
Kimi K3 还采用了一种我们希望更多模型供应商能够考虑的发布流程：先发布模型，然后再发布权重和推理引擎支持。

The vLLM team proposed this separation, and Moonshot AI agreed and executed. The reason is practical. A frontier-model announcement has unavoidable last-mile uncertainty. The model team is simultaneously stabilizing its own products, APIs, evaluations, safety work, documentation, and commercial launch. If open-source weights and open-source support must land at the exact same moment, a community project such as vLLM suffers from the moving deadline.  
vLLM 团队提出了这种分离方案，Moonshot AI 同意并执行了该方案。原因很实际。前沿模型的发布不可避免地存在“最后一公里”的不确定性。模型团队同时还要稳定自身的产品、API、评估、安全工作、文档编写和商业发布。如果开源权重和开源支持必须在同一时间发布，像 vLLM 这样的社区项目就会因截止日期的不断变化而受到影响。

Separating the two timelines gives both sides a better contract:  
将两条时间线分开，对双方都有利，能使合同更加完善：

1. The model vendor can concentrate on its product launch and freeze the final checkpoint, configuration, tokenizer, and serving semantics.  
	模型供应商可以专注于产品发布，并冻结最终检查点、配置、分词器和服务语义。
2. The open-source inference engine team gets a stable integration window for correctness tests, performance tuning, Docker builds, and recipe validation.  
	开源推理引擎团队获得了一个稳定的集成窗口，用于正确性测试、性能调优、Docker 构建和配方验证。
3. The community gets a public, bounded expectation instead of an ambiguous “coming soon.”  
	社区得到的是一个公开的、明确的预期，而不是模糊的“即将推出”。

The separation is not a retreat from day-0 support. It is a more sustainable way to deliver day-0 support against the artifact that users will actually download. We encourage more model vendors to follow!  
此次分离并非放弃首日支持，而是一种更可持续的方式，能够针对用户实际下载的产品提供首日支持。我们鼓励更多模型厂商效仿！