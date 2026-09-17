# SGLang 显存不足时如何回收 KV Cache，哪些显存不能被 RadixCache 淘汰？

- 整理状态：已整理
- 题目出处：用户提供的一面面经（SGLang / CUDA 系统专题）

## 其他问法

- SGLang 显存不足怎么处理？有没有看过源码？
- SGLang 的显存主要由哪些部分组成？

## 30 秒回答

先区分权重、KV pool、运行时 workspace/激活和框架保留显存。生成过程中如果 KV slot 不够，SGLang 可以先从 RadixCache 驱逐未被请求锁定的低价值前缀，归还 page/slot；仍不足时需要 retract/preempt 部分请求、降低并发或 token budget，之后通过已有前缀或重计算恢复。RadixCache 只能回收可驱逐的 KV 前缀，不能淘汰模型权重、正在执行请求锁定的 KV，也不能自动解决 workspace 峰值 OOM。

## 深入解释

源码层建议沿四个对象看：

1. `ReqToTokenPool`：请求逻辑 token 位置到 KV slot 的映射。
2. `TokenToKVPoolAllocator` / `PagedTokenToKVPoolAllocator`：空闲 slot/page 的分配与回收。
3. `RadixCache`：哪些历史前缀仍值得保留、哪些节点被引用、哪些可以 LRU 驱逐。
4. Scheduler：申请失败后的 eviction、request retract、准入和重计算策略。

工程措施还包括降低 `mem_fraction_static`/并发/最大上下文或 chunked-prefill token budget、使用 GQA/MLA/KV 量化、扩容或采用分层 KV。必须先用显存账本确认是权重容量、KV 增长还是临时 workspace 峰值。

## 关联知识

- [[../../../../wiki/concepts/KV Cache|KV Cache]]
- [[../../../../wiki/concepts/RadixAttention|RadixAttention]]
- [[../../../../wiki/entities/SGLang|SGLang]]

## 参考来源与待核实

- [SGLang KV allocator 源码](https://github.com/sgl-project/sglang/blob/575759d90af942eeff89f1c8c33a1fcdb2da4181/python/sglang/srt/mem_cache/allocator/paged.py)
- [SGLang cache allocation orchestration](https://github.com/sgl-project/sglang/blob/575759d90af942eeff89f1c8c33a1fcdb2da4181/python/sglang/srt/mem_cache/common.py)
- [[LLM 推理显存由哪些部分组成，如何按对象优化|LLM 推理显存组成题]]
- 具体 OOM 回退顺序、启动参数和 HiCache 行为会随版本与 backend 改变；回答时需绑定部署版本。

## 所属题单

- [[../sets/SGLang与CUDA系统一面|SGLang 与 CUDA 系统一面]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
