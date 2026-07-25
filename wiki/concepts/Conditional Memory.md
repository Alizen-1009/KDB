---
type: concept
topic: 模型架构
sources: 2
updated: 2026-05-17
---

# Conditional Memory

## 定义

`Conditional Memory` 是一种与条件计算互补的稀疏建模方式：模型不是为每个 token 激活更多计算路径，而是根据输入局部上下文只检索固定数量的静态记忆条目。

## 它解决什么问题

- 让 Transformer 获得一种更原生的“知识查找”原语，而不是总通过多层计算去模拟检索
- 把局部、固定、可表驱动的模式存储从主干计算中分离出来
- 在不显著增加每 token FLOPs 的情况下扩大模型的静态容量

## 核心机制

- 从输入局部上下文构造确定性检索键，例如压缩后的后缀 `N-gram`
- 通过哈希或其它固定寻址机制，把巨大 `N-gram` 空间压到有限 memory slots；查找形式类似 `idx = hash(compressed_ngram) % table_size`
- 记忆表本身类似超大 embedding table，是模型参数的一部分；训练时被命中的 rows 参与反向传播并被更新，推理时只做确定性数组索引
- 因为每个 token 只计算固定数量的 hash / head，并只读取固定数量的 rows，所以单 token memory lookup 是 `O(1)`，不是按表大小线性扫描
- 用当前 hidden state 对检索结果做 context-aware gating，抑制碰撞噪声与语义不匹配
- 将门控后的记忆结果作为残差支路注入主干，而不是替代 attention / FFN / MoE
- 在官方 demo 中，这条路径被具体实现为：`CompressedTokenizer -> NgramHashMapping -> MultiHeadEmbedding -> gate -> ShortConv -> residual`

## 关键权衡

- 能把静态模式与动态推理解耦，但需要维护巨大的静态表和潜在的哈希碰撞
- 更适合处理局部、稳定、可查表的知识或模式，不能替代需要深层推理的动态计算
- 真正释放价值往往依赖系统共设计，例如表分片、异步预取和多级缓存
- 和普通 token embedding 相比，优势是 key 能表达局部上下文；代价是 `N-gram` 空间爆炸，必须依赖压缩、哈希和碰撞抑制。

## 相关实体

- [[../entities/Engram]]

## 相关来源

- [[../sources/Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models]]
- [[../sources/engram_demo_v1]]

## 相关概念

- [[Sparsity Allocation]]

## 研究备注

- 后续可继续补它与 `MoE`、普通 token embedding、`RAG`、`KV Cache`、`Prefix Caching` 的边界对比，以及 demo 中 token 压缩、质数哈希表和门控实现对系统设计的启发
