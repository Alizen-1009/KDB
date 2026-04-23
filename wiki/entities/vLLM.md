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

## 相关概念

- [[Continuous Batching]]
- [[PagedAttention]]
- [[Prefix Caching]]
- [[缓存感知路由]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/斯坦福CS336 Lecture 10 - Inference systems and optimization]]
- [[../sources/美团一面：请介绍 vLLM PageAttention]]

## 冲突与备注

- 本文只点到 vLLM 的相关能力，没有展开版本差异、调度细节和具体实现限制，后续需要结合官方文档补足
- 目前 wiki 里的 vLLM 条目已经覆盖了 `PagedAttention`、`Continuous Batching`、`Prefix Caching` 三个面试高频锚点，但 `chunked prefill`、`disaggregated prefilling` 还可继续补
