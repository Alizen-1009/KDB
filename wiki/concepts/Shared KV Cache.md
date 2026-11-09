---
type: concept
topic: KV Cache
sources: 3
updated: 2026-05-07
---

# Shared KV Cache

## 定义

在模型内部让最后若干层共享同一套 K/V 表示，而不是为每一层单独维护完整 KV Cache 的机制。

## 它解决什么问题

- 降低长上下文推理时的 KV Cache 内存占用
- 减少某些层重复进行 K/V 投影的计算成本

## 核心机制

- 前面的非共享层正常计算并存储 K/V
- 后面的共享层直接复用特定非共享层的 K/V
- 通过层间复用而不是请求间复用来压缩缓存成本
- 在 `MTP Drafter` 语境中还会出现另一类共享：drafter 通过 cross-attention 复用 target model 已计算的 KV cache。这和本文定义的“目标模型内部层间共享”不是同一层级，但都服务于减少重复上下文计算。

## CSA2：缓存共享与索引共享分开

在 DeepSeek-V4.1-Flash 的 [[CSA2]] 中，全局主 KV 与 Indexer K 只由少数来源层生成；Reindex 可共享缓存但重新选 Top-K，Reuse 同时复用缓存与选择。每层主 Q、局部 SWA KV 与主注意力仍需计算，共享选择不意味着共享权重或输出。

[[Causal Encoder-Decoder]] 进一步使 Decoder 全局 KV 来自 Encoder 边界表示，改变 Prefill 的依赖；全局缓存共享不替代局部 SWA 状态恢复。这与既有 Gemma 层间共享同属减少独立 KV 副本的方向，但具体层调度与局部支路不同。

## 关键权衡

- 能减少缓存占用和部分投影计算
- 可能牺牲部分层级表达独立性，因此需要其它机制补偿

## 相关实体

- [[../entities/DeepSeek-V4.1-Flash]]

- [[../entities/Gemma 4]]

## 相关来源

- [[../sources/详细分析 DeepSeek-V4.1 Flash 技术报告(1)  模型结构]]

- [[../sources/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构]]
- [[../sources/Gemma 4：Drafter 详解]]

## 相关概念

- [[KV Cache]]
- [[Double-Wide MLP]]
- [[MTP Drafter]]

## 研究备注

- 需要明确区分它和 `Prefix Caching`、`Cache-aware routing` 这类服务层缓存复用机制
- 还需要区分它和 target-drafter 之间的 KV cache sharing：前者是模型内部层间复用，后者是 speculative decoding 辅助模型复用目标模型状态
