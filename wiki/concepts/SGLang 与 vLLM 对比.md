---
type: concept
topic: 推理服务
sources: 4
updated: 2026-05-07
---

# SGLang 与 vLLM 对比

## 定义

`SGLang 与 vLLM 对比` 是围绕两个开源 LLM 推理系统的抽象重心、缓存机制、适用负载和工程选择进行区分：`vLLM` 更偏通用 serving engine，`SGLang` 更偏可编程 LLM runtime。

## 一句话区分

`vLLM` 的主线是把模型服务跑得稳、吞吐高、易部署；`SGLang` 的主线是把复杂 LLM 程序跑得更高效、更可控。

## 核心对比

| 维度 | [[../entities/vLLM]] | [[../entities/SGLang]] |
| --- | --- | --- |
| 核心定位 | 高吞吐、显存高效的 LLM serving engine | 高性能 serving + 面向 [[LLM Programs]] 的 runtime |
| 代表技术 | [[PagedAttention]]、[[Continuous Batching]]、OpenAI-compatible server | [[RadixAttention]]、frontend DSL、[[Constrained Decoding]] / structured output |
| 缓存重心 | 用分页/块表管理 KV cache，减少碎片并支撑动态 batch | 用 radix tree 做前缀搜索、复用、插入和驱逐，吃到程序分支和共享前缀收益 |
| 抽象层级 | 请求级 serving、调度、显存管理、API 服务 | 多次生成调用、控制流、并行、分支、结构化输出 |
| 适合场景 | 通用 chat/completion API、高并发在线服务、快速部署 | Agent、RAG pipeline、多轮工作流、多分支推理、严格结构化输出 |
| 主要风险 | 复杂 workflow 需要应用层自行编排，系统未必理解调用图 | DSL/runtime 学习和迁移成本更高，收益依赖任务结构和缓存命中 |

## 图片说法的校正

- 可以说 `vLLM` 更偏“通用高并发推理引擎”，但不应说它只适合“单轮次、简单问答”。`vLLM` 也支持 chat、prefix caching、guided/structured output、speculative decoding、多模态和多种部署能力。
- 可以说 `SGLang` 更适合表达复杂任务，但不应说它不是普通 serving framework。`SGLang` 也支持 OpenAI-compatible API 和生产级 serving。
- `RadixAttention` 的数倍收益是条件性收益，主要来自共享前缀、多调用、多分支或固定模板；普通单轮请求不一定能获得数倍提升。
- structured output / constrained decoding 能提高格式正确性，但不能保证语义正确；性能提升取决于约束形式和模板中可压缩的确定性片段。
- `PagedAttention` 显著降低 KV cache 碎片和浪费是正确方向；具体“70%+ 到 10%-”这类数字需要回到论文或 benchmark 核实。

## 工程选择

- 如果目标是快速搭建稳定、高并发的模型 API 服务，且交互逻辑主要在应用层完成，优先评估 `vLLM`。
- 如果业务请求天然包含大量共享前缀、多次生成调用、分支/合并、RAG 上下文复用或强结构化输出，优先评估 `SGLang`。
- 如果已有服务已经跑在 `vLLM` 上，可以先定位瓶颈是否来自复杂 workflow 的重复 prefill、结构化输出开销或多调用延迟，再决定是否把部分负载迁移到 `SGLang` 或做路由分流。

## 相关实体

- [[../entities/vLLM]]
- [[../entities/SGLang]]

## 相关来源

- [[../sources/SGLang 与 vLLM 区别截图整理]]
- [[../sources/SGLang：LLM推理引擎发展新方向]]
- [[../sources/LLM推理优化核心技术]]
- [[../sources/Model Runner V2 A Modular and Faster Core for vLLM]]

## 相关概念

- [[PagedAttention]]
- [[RadixAttention]]
- [[LLM Programs]]
- [[Constrained Decoding]]
- [[KV Cache]]
- [[Prefix Caching]]
- [[Continuous Batching]]

## 研究备注

- 二者边界正在变模糊：`vLLM` 在补 structured output、speculative decoding、prefix caching 等复杂能力；`SGLang` 也不只是 Agent DSL，而是生产级 serving runtime。
- 后续如果要做严格选型，需要补同一模型、同一硬件、同一 workload 下的实测，而不是只引用不同博客的单点 benchmark。
