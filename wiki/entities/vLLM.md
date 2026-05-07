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

## 相关概念

- [[Continuous Batching]]
- [[PagedAttention]]
- [[持久批处理]]
- [[Prefix Caching]]
- [[缓存感知路由]]
- [[确定性推理]]
- [[Speculative Decoding]]
- [[vLLM V1 统一调度器]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/斯坦福CS336 Lecture 10 - Inference systems and optimization]]
- [[../sources/美团一面：请介绍 vLLM PageAttention]]
- [[../sources/Model Runner V2 A Modular and Faster Core for vLLM]]
- [[../sources/推理的非确定性运算及vLLMSGLang控制方式]]
- [[../sources/LLM提速利器：投机推理的原理与常见方案]]
- [[../sources/vLLM v0 与 vLLM v1 调度架构差异截图整理]]

## 冲突与备注

- 本文只点到 vLLM 的相关能力，没有展开版本差异、调度细节和具体实现限制，后续需要结合官方文档补足
- 目前 wiki 里的 vLLM 条目已经覆盖了 `PagedAttention`、`Continuous Batching`、`Prefix Caching` 三个面试高频锚点，但 `chunked prefill`、`disaggregated prefilling` 还可继续补
- `MRV2` 说明 vLLM 正在把运行时架构从“功能累加”往“模块化、GPU-native、async-first”的方向收束，但截至来源发布时它仍是实验态，并未完全替代旧 runner
- 关于 `Batch Invariance` 的支持模型、硬件条件和性能代价，目前库里仍主要来自经验文章摘要，后续宜补官方文档核实
- 关于 speculative decoding 的方法矩阵和版本限制，目前库里仍主要来自经验文章整理；若后续要写 `vLLM vs SGLang` 对比，宜再补官方文档
- 关于 `vLLM v0/v1` 调度差异，应避免把 v0 简化成“完全不能混合 prefill/decode”，也避免把 v1 说成“prefill/decode 在计算上完全相同”；更准确的边界是调度表示从阶段中心转向 token budget 中心
