# SGLang RadixCache 如何组织前缀复用、引用与淘汰？

- 整理状态：已整理
- 题目出处：用户提供的一面面经（SGLang / CUDA 系统专题）

## 其他问法

- RadixCache 详细介绍一下。
- RadixAttention 如何复用 KV Cache？

## 30 秒回答

SGLang 用 radix tree 按 token 前缀组织可复用 KV Cache。新请求先做最长前缀匹配，命中的节点返回对应 KV slot，未命中的后缀才执行 prefill；完成后再把新 token 与 KV slot 插回树中。节点需要记录树结构、KV slot、引用或锁定状态和 LRU 信息：正在被请求使用的节点不能驱逐，显存不足时优先从未被引用的叶子按 LRU 回收。它解决的是跨请求前缀复用，底层 slot/page allocator 解决的是 KV 物理显存分配，二者不能混成同一层。

## 深入解释

Radix tree 的边不是单个字符，而可以是一段 token；插入时若新 key 与已有边只有部分公共前缀，需要拆分节点。逻辑树保存的是 token 前缀与 KV slot 索引，不是把整块 KV 数据复制进树节点。命中收益主要是省掉重复 prefill，因此取决于系统提示、多轮对话、RAG 模板等前缀的共享程度。

## 面试官可能追问

- [[SGLang 显存不足时如何回收 KV Cache，哪些显存不能被 RadixCache 淘汰|SGLang 显存不足时如何回收 KV Cache？]]
- [[SGLang 的 varlen metadata 如何映射到物理 KV slot|SGLang 的 varlen metadata 如何映射到物理 KV slot？]]
- RadixCache 与普通 PagedAttention 分别解决什么问题？

## 关联知识

- [[../../../../wiki/concepts/RadixAttention|RadixAttention]]
- [[../../../../wiki/concepts/Prefix Caching|Prefix Caching]]
- [[../../../../wiki/entities/SGLang|SGLang]]

## 参考来源与待核实

- [[../../../../wiki/concepts/RadixAttention|RadixAttention 概念页]]
- [SGLang RadixCache 源码](https://github.com/sgl-project/sglang/blob/575759d90af942eeff89f1c8c33a1fcdb2da4181/python/sglang/srt/mem_cache/radix_cache.py)
- 实现字段与驱逐细节绑定 SGLang commit `575759d9`；不同版本、ChunkCache、HiCache、SWA 或混合递归状态路径可能不同。

## 所属题单

- [[../sets/SGLang与CUDA系统一面|SGLang 与 CUDA 系统一面]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
