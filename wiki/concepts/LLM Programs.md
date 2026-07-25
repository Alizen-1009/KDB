---
type: concept
topic: 推理服务
sources: 2
updated: 2026-05-07
---

# LLM Programs

## 定义

`LLM Programs` 是把 LLM 调用组织成带控制流、结构化输入输出和多步组合逻辑的程序化使用范式，而不是只把模型当成单轮聊天接口。

## 它解决什么问题

- 表达 `multi-round planning`、reasoning、工具调用、多模态输入、self-consistency、tree-of-thought 等复杂交互模式。
- 让多个 LLM 调用之间可以组合、分支、并行、合并，并把结果以结构化形式交给下游软件系统。
- 给推理 runtime 提供更丰富的全局信息，使其有机会做 lazy scheduling、缓存复用和受约束生成优化。

## 核心机制

- 一个 program 通常包含多个 `gen / select / fork / merge` 之类的 LLM 调用原语。
- 调用之间穿插普通控制流，用于根据中间结果决定后续 prompt、分支或输出格式。
- 输入和输出更偏结构化：例如图片路径、论文文本、JSON schema、正则表达式约束和评分字段。
- 在 `SGLang` 中，前端 DSL 嵌入 Python，后端 runtime 可以把多调用程序映射到更高效的执行计划。
- 和 `vLLM` 对比时，`LLM Programs` 是理解 `SGLang` 差异的关键：差异不是“是否能聊天”，而是系统是否把多调用、控制流、结构化输出等上层调用图纳入 runtime 优化。

## 关键权衡

- 表达能力强于单次 API 调用，但 runtime、状态管理和调试复杂度更高。
- 如果程序结构能暴露大量共享前缀或固定输出模板，可以显著受益于缓存和 constrained decoding；反之收益可能有限。
- 程序化抽象越强，越需要清楚地区分“模型能力提升”和“系统执行优化”两类收益。

## 相关实体

- [[../entities/SGLang]]

## 相关来源

- [[../sources/SGLang：LLM推理引擎发展新方向]]
- [[../sources/SGLang 与 vLLM 区别截图整理]]

## 相关概念

- [[RadixAttention]]
- [[SGLang 与 vLLM 对比]]
- [[Constrained Decoding]]
- [[Speculative Decoding]]
- [[KV Cache]]
- [[Prefix Caching]]

## 研究备注

- 后续可补 `SGLang` 论文、DSPy、Guidance、LMQL 等系统，比较不同 `LLM Programs` 抽象对 runtime 优化的暴露程度。
- 需要避免把所有“多轮 prompt 工程”都泛化为 LLM Programs；关键区别在于是否有明确的程序结构、状态管理和可组合输出。
