# vLLM

## 一句话说明

面向大语言模型推理与 serving 的高吞吐开源系统，以 `PagedAttention` 和高效调度能力著称。

## 类型

- 项目 / 推理框架

## 核心信息

- 常被用作生产级 LLM serving 引擎，强调吞吐、显存利用率和多请求调度效率。
- 与 `PagedAttention` 关系非常紧密，这也是它在社区中的标志性设计之一。
- 在本文语境里，vLLM 被提到支持 `Prefix Caching`。
- 在 Stanford CS336 推理课的语境里，vLLM 代表的是“把 paging、continuous batching 和现代 attention kernel 结合起来的推理系统”。
- 新增来源进一步把它的 `PagedAttention` 讲清楚为 `logical block / physical block / block table` 的组合，这也是面试里最常见的解释路径之一。
- 新来源 `MRV2` 进一步说明，`vLLM` 的优化重点不只在 `PagedAttention`，还在于执行核心本身：包括 `persistent batching`、GPU-native input preparation、async-first scheduling 和更模块化的 `ModelState` 抽象。
- 新增来源还补入了 `vLLM` 在可复现性上的一条工程主线：除了给采样设置 `seed`，还可以通过关闭 `V1 multiprocessing`、开启 `Batch Invariance` 等方式减少调度与 kernel 路径带来的非确定性；但这通常会带来性能回退，且支持范围有限。
- 新来源补充了 `vLLM` 在 speculative decoding 上的使用面：它不仅支持小 draft model，也支持 `ngram / suffix / MTP / EAGLE` 等多类 speculative 配置，但不同版本和并行策略存在能力边界。
- 新增截图整理补足了 `vLLM v0 -> vLLM v1` 的调度架构变化：v1 以 `{request_id: num_tokens}` 形式统一 prompt/output token 的每步调度决策，更自然地支持 chunked prefill、prefix caching 和 speculative decoding；但 `token quota`、chunked prefill 默认行为和优先级调度能力都需要按具体版本核实。
- 新来源 `SGLang：LLM推理引擎发展新方向` 把 `vLLM` 放在推理框架演化史中讨论：它因 `PagedAttention`、PyTorch 生态易用性、开源社区和多硬件支持成为现象级系统，但也可能像早期 `Caffe` 一样在新使用范式和硬件压力下继续被重构。
- 新增截图整理校正了一个常见误解：`vLLM` 的抽象重心偏 serving engine，但这不等于它只能做单轮简单问答；它也在支持 prefix caching、structured output、speculative decoding、多模态等能力。
- 新增 `PageAttention代码走读` 从源码实现角度补充了 decode kernel 视角：`vLLM` 通过 `block table` 间接读取 paged KV cache，kernel 对每个 sequence/head 遍历历史 KV blocks，并在历史 token 维度完成 attention softmax。
- 新增 `vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现` 从 CUDA 并行算法角度补充 `PAv1`：每个 thread block 负责一个 `sequence/head` 输出行，warp/thread group 分摊 paged KV cache 的 QK 与 PV 两段计算，并指出 PAv1 与 FlashAttention/FlashDecoding 的任务切分差异。
- 新来源 `RTP-LLM` 将 `vLLM` 作为模型加载、TTFT、推测解码和多模态吞吐的对比基线；这些对比应限定在原文给出的模型、硬件、并行配置和框架版本下。
- 新来源 `vllm并行策略之DCP` 补充了 `vLLM` 的 decode context parallel 口径：DCP 复用 TP group，通过 `--decode-context-parallel-size` 在 decode 阶段沿 `seq_len` 维分片 KV cache，适合 `MLA/MQA/GQA` 这类 `num_kv_heads` 较小、纯 TP 容易复制 KV cache 的场景。
- `vLLM` 的 `--enforce-eager` 与 `cudagraph_mode=NONE` 不完全等价：前者是运行在 eager mode 的总开关，会关闭 `torch.compile` 集成和 CUDA Graphs；后者只关闭 CUDA Graphs，仍可能保留 `torch.compile` / vLLM compile 的其他路径。
- `Look Ma, No Bubbles!` 将 vLLM 作为 Llama-3.2-1B、batch size 1、BF16 低延迟 decode baseline，指出在该极窄场景中许多短 kernel 边界会限制可用 HBM 带宽；该结论不能直接外推到高并发 serving。

## 相关概念

- [[Continuous Batching]]
- [[PagedAttention]]
- [[持久批处理]]
- [[Prefix Caching]]
- [[缓存感知路由]]
- [[确定性推理]]
- [[Speculative Decoding]]
- [[vLLM V1 统一调度器]]
- [[LLM Programs]]
- [[SGLang 与 vLLM 对比]]
- [[分层 KV Cache]]
- [[Decode Context Parallel]]
- [[Chunked Prefill]]
- [[Torch Compile]]
- [[Megakernel]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/斯坦福CS336 Lecture 10 - Inference systems and optimization]]
- [[../sources/美团一面：请介绍 vLLM PageAttention]]
- [[../sources/Model Runner V2 A Modular and Faster Core for vLLM]]
- [[../sources/推理的非确定性运算及vLLMSGLang控制方式]]
- [[../sources/LLM提速利器：投机推理的原理与常见方案]]
- [[../sources/vLLM v0 与 vLLM v1 调度架构差异截图整理]]
- [[../sources/SGLang：LLM推理引擎发展新方向]]
- [[../sources/SGLang 与 vLLM 区别截图整理]]
- [[../sources/PageAttention代码走读]]
- [[../sources/vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现]]
- [[../sources/RTP-LLM]]
- [[../sources/vllm并行策略之DCP(Decode Context Parallel)]]
- [[../sources/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B]]

## 冲突与备注

- 本文只点到 vLLM 的相关能力，没有展开版本差异、调度细节和具体实现限制，后续需要结合官方文档补足
- 目前 wiki 里的 vLLM 条目已经覆盖了 `PagedAttention`、`Continuous Batching`、`Prefix Caching` 三个面试高频锚点，但 `chunked prefill`、`disaggregated prefilling` 还可继续补
- `MRV2` 说明 vLLM 正在把运行时架构从“功能累加”往“模块化、GPU-native、async-first”的方向收束，但截至来源发布时它仍是实验态，并未完全替代旧 runner
- 关于 `Batch Invariance` 的支持模型、硬件条件和性能代价，目前库里仍主要来自经验文章摘要，后续宜补官方文档核实
- 关于 speculative decoding 的方法矩阵和版本限制，目前库里仍主要来自经验文章整理；若后续要写 `vLLM vs SGLang` 对比，宜再补官方文档
- 关于 `vLLM v0/v1` 调度差异，应避免把 v0 简化成“完全不能混合 prefill/decode”，也避免把 v1 说成“prefill/decode 在计算上完全相同”；更准确的边界是调度表示从阶段中心转向 token budget 中心
- 新来源对 `vLLM` 的“Caffe 类比”是作者判断，不是技术事实；可作为框架演化视角保留，但不宜直接当作性能或生命周期结论
- 对比 `SGLang` 时，应把差异落到抽象层和负载结构：`vLLM` 更偏高吞吐 serving，`SGLang` 更偏 LLM Programs runtime；不要写成能力互斥
- `PagedAttention` 的具体 kernel 名称、cache layout 和线程组织属于版本相关实现细节；长期笔记中应保留机制层结论，并在精确引用时补具体 commit
- `PAv1` 适用条件、`PAv2` 切换启发式、以及 MQA/GQA 下是否重复读取 KV cache，均可能随 vLLM 版本和 backend 改动；引用时应落到具体源码版本。
- RTP-LLM 文章中的横向比较属于来源 benchmark 声称，不宜覆盖既有 `vLLM` 条目中关于 PagedAttention、MRV2、统一调度器等机制层总结。
- DCP 来源中的 CUDA backend 支持、PCP 状态、`dcp_all2all` 通信和 Chunked Prefill / Prefix Cache 兼容性都应按具体 vLLM 版本、PR 或官方文档复核。
- `cudagraph_mode=NONE` 只表示不 capture/replay CUDA Graph；如果排查的是 `torch.compile` / Dynamo / Inductor / vLLM compile 本身的问题，应使用 `--enforce-eager` 或进一步设置 compilation mode，而不是只关 CUDA Graphs。
- `Look Ma, No Bubbles!` 中关于 vLLM H100 带宽利用率和相对 megakernel 的性能差距，均应作为来源 benchmark 声称引用，并保留 prompt 长度、生成长度、dtype、硬件和 baseline 配置。
