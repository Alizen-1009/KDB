---
type: entity
entity_type: 框架
topic: 推理服务
sources: 7
updated: 2026-06-12
---

# SGLang

## 一句话说明

面向大模型推理和程序化交互的系统，在缓存组织与执行效率方面有强工程取向。

## 类型

- 项目 / 推理框架

## 核心信息

- 文中提到 SGLang 通过 `RadixTree` 思路支持前缀复用。
- 在缓存命中、长前缀复用和高效调度场景中，SGLang 经常与 vLLM 被拿来对照。
- 它属于 AI infra 语境下典型“系统实现型实体”，适合作为概念页的落点。
- 新来源补入了 `SGLang` 在确定性推理上的支持：服务端可开启 `deterministic inference`，客户端则能通过 `sampling_seed` 为不同请求指定可复现采样路径，这很适合 `RL rollout` 或精度对齐场景。
- 新来源 `SGLang：LLM推理引擎发展新方向` 进一步把它定位为面向 [[LLM Programs]] 的 runtime：前端用嵌入 Python 的 DSL 表达多次 LLM 调用、分支、合并和结构化输出，后端通过 [[RadixAttention]]、[[Constrained Decoding]] 与 API speculative execution 优化执行。
- 该来源还转述了 `SGLang V2` 在部分 H100 Llama3 serving benchmark 中接近甚至超过 `TensorRT-LLM`、明显快于 `vLLM` 的结果；但作者没有实测，应作为“来源声称 / 待复现”的性能备注。
- 新增截图整理把 `SGLang vs vLLM` 的常见二分法校正为：`SGLang` 更偏可编程 runtime 和复杂 LLM workflow，但它同样是 production-level serving framework；不能简单理解成“只适合 Agent”。
- 官方文档语境里，SGLang 对 DeepSeek/MLA serving 还强调 [[DP Attention]]：通过 attention 侧数据并行减少 TP 下 latent KV cache 重复，提升可承载 batch size 和吞吐。
- 新来源 `RTP-LLM` 将 `SGLang` 作为模型加载、TTFT、推测解码和多模态吞吐的对比基线；这些结果应视为特定 benchmark/生产流量下的来源声称。
- `Look Ma, No Bubbles!` 将 SGLang 作为 Llama-3.2-1B、batch size 1、BF16 低延迟 decode baseline，指出 megakernel 在该特定场景中能进一步减少 kernel 边界带来的 pipeline bubble。

## 相关概念

- [[Prefix Caching]]
- [[RadixAttention]]
- [[LLM Programs]]
- [[SGLang 与 vLLM 对比]]
- [[Constrained Decoding]]
- [[Speculative Decoding]]
- [[缓存感知路由]]
- [[确定性推理]]
- [[DP Attention]]
- [[分层 KV Cache]]
- [[Megakernel]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/推理的非确定性运算及vLLMSGLang控制方式]]
- [[../sources/SGLang：LLM推理引擎发展新方向]]
- [[../sources/SGLang 与 vLLM 区别截图整理]]
- [[../sources/MLA与DP Attention面试整理]]
- [[../sources/RTP-LLM]]
- [[../sources/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B]]

## 冲突与备注

- 虽然新来源强调 `SGLang V2` 的调度实现优势，但没有展开内部调度语义，后续可专门 ingest 官方设计文档、论文或代码路径
- 当前条目对 `deterministic inference` 的理解仍偏功能层摘要；若后续需要比较 `SGLang vs vLLM` 的确定性实现差异，建议再补官方文档或代码路径
- `RadixAttention` 与普通 [[Prefix Caching]] 应区分：前者强调 radix tree runtime 结构和 program 分支复用，后者是更通用的跨请求公共前缀复用概念
- 关于和 `vLLM` 的区别，推荐回链 [[SGLang 与 vLLM 对比]]；避免把差异简化为“简单问答 vs Agent”
- RTP-LLM 来源中的对比不应被写成 `SGLang` 的通用弱点；SGLang 在 LLM Programs、RadixAttention、结构化输出和 DP Attention 上仍有独立设计重心。
- `Look Ma, No Bubbles!` 中关于 SGLang 的性能差距只应作为来源实验观察，不代表 SGLang 在高 batch、长上下文或 LLM Programs workload 下的通用表现。
