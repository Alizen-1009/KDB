---
type: concept
topic: KV Cache
sources: 3
updated: 2026-05-07
---

# RadixAttention

## 定义

`RadixAttention` 是 `SGLang` 提出的 KV cache prefix reuse 机制：用类似 radix tree 的结构保留 prompt 和生成结果的 KV cache，以支持运行时前缀搜索、复用、插入与驱逐。

## 它解决什么问题

- `LLM Programs` 中多个分支、fork 副本或 program 实例常共享长前缀，重复 prefill 会浪费大量计算。
- 普通请求完成后直接丢弃 KV cache，会错过系统提示、RAG 模板、固定 JSON 前缀等可复用结构。
- 复杂多调用程序需要一种系统化方式自动识别可复用前缀，而不是靠应用层手写缓存逻辑。

## 核心机制

- 将 token 前缀和对应 KV cache 组织在 radix tree 中。
- 新请求或 program 分支到来时，先做最长前缀匹配，命中部分直接复用 KV cache。
- 执行完成后，把新的 prompt 片段和生成结果继续插入树中，供后续分支或请求复用。
- 结合 LRU 驱逐和 cache-aware scheduling，在显存有限时优先保留更可能命中的缓存。

## 关键权衡

- 缓存命中时可以减少 prefill 计算和 TTFT，但收益依赖 prompt/program 结构的共享程度。
- radix tree 状态、显存驱逐和调度策略会增加 runtime 复杂度。
- 与 [[Prefix Caching]] 高度相关，但 `RadixAttention` 更强调运行时内的系统化树结构和 program 分支场景。
- 和 `vLLM` 的 [[PagedAttention]] 相比，二者关注点不同：`PagedAttention` 主要解决 KV cache 的分页显存管理和动态调度，`RadixAttention` 主要解决共享前缀的运行时复用。

## 相关实体

- [[../entities/SGLang]]
- [[../entities/vLLM]]

## 相关来源

- [[../sources/SGLang：LLM推理引擎发展新方向]]
- [[../sources/LLM推理优化核心技术]]
- [[../sources/SGLang 与 vLLM 区别截图整理]]

## 相关概念

- [[KV Cache]]
- [[Prefix Caching]]
- [[缓存感知路由]]
- [[LLM Programs]]
- [[SGLang 与 vLLM 对比]]

## 研究备注

- 文章指出 Mooncake、MemServe 等系统也采用相似的 KV cache prefix sharing 思路，但不同系统的共享边界可能不同：有的偏 program 内部复用，有的偏跨请求、跨用户或 prefilling pool。
- 后续需要补 SGLang 论文或代码，核实 radix tree key、block 管理、eviction 粒度和调度策略的具体实现。
