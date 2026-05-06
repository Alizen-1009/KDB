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

## 相关概念

- [[Prefix Caching]]
- [[缓存感知路由]]
- [[确定性推理]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/推理的非确定性运算及vLLMSGLang控制方式]]

## 冲突与备注

- 文中没有深入展开 SGLang 的内部调度语义，后续可专门 ingest 官方设计文档
- 当前条目对 `deterministic inference` 的理解仍偏功能层摘要；若后续需要比较 `SGLang vs vLLM` 的确定性实现差异，建议再补官方文档或代码路径
