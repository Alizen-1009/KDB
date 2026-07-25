---
type: map
topic: KV Cache
---

# KV Cache

## 导读

[[../concepts/KV Cache|KV Cache]] 是总纲，读完它再按“谁在省什么”分三路：

- **显存管理**：[[../concepts/PagedAttention|PagedAttention]]（分页消碎片）
- **跨请求复用**：[[../concepts/Prefix Caching|Prefix Caching]] → [[../concepts/RadixAttention|RadixAttention]]（前缀树共享）→ [[../concepts/缓存感知路由|缓存感知路由]]（多副本时把请求打到有缓存的那台）
- **递归模型复用**：[[../concepts/递归状态 Prefix Caching|递归状态 Prefix Caching]] —— KDA/GDN 不能按任意 token 行回退，需要保存 Conv State 与矩阵状态 checkpoint
- **模型侧减量**：[[../concepts/分层 KV Cache|分层 KV Cache]]、[[../concepts/Shared KV Cache|Shared KV Cache]]

前两路是运行时能改的，第三路要改模型结构，代价不同。

<!-- BEGIN AUTO：以下由 scripts/update_index.py 生成，改动会被覆盖 -->

## 概念（8）

- [[../concepts/KV Cache|KV Cache]]
- [[../concepts/PagedAttention|PagedAttention]]
- [[../concepts/Prefix Caching|Prefix Caching]]
- [[../concepts/RadixAttention|RadixAttention]]
- [[../concepts/Shared KV Cache|Shared KV Cache]]
- [[../concepts/分层 KV Cache|分层 KV Cache]]
- [[../concepts/缓存感知路由|缓存感知路由]]
- [[../concepts/递归状态 Prefix Caching|递归状态 Prefix Caching]]

## 来源（4）

- [[../sources/PageAttention代码走读|PageAttention代码走读]]
- [[../sources/SGLang的KDA管理与Prefix Cache难题|SGLang的KDA管理与Prefix Cache难题]]
- [[../sources/vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现|vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现]]
- [[../sources/美团一面：请介绍 vLLM PageAttention|美团一面：请介绍 vLLM PageAttention]]
