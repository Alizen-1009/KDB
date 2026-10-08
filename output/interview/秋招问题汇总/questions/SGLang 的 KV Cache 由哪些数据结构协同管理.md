# SGLang 的 KV Cache 由哪些数据结构协同管理？

- 整理状态：已整理
- 题目出处：用户提供的一面面经（PyTorch 图编译与推理系统专题）

## 30 秒回答

SGLang 的 KV Cache 可以分成四层：物理 KV tensor pool 保存各层 K/V；token/page allocator 管理空闲 physical slot；`ReqToTokenPool` 保存请求逻辑 token 位置到 slot 的二维映射；RadixCache 在其上保存可共享前缀及对应 slot，并负责匹配、引用保护和驱逐。Scheduler 为 prefill/decode 申请 slot，`ForwardBatch` 把 varlen 长度、请求行号和 `out_cache_loc` 交给 attention/cache kernel。

## 深入解释

```text
RadixCache：哪些前缀可复用、哪些可驱逐
    ↓ 持有 slot indices
ReqToTokenPool：request + logical position → slot
    ↓
Allocator：分配/回收 token slot 或 page
    ↓
KV tensor pool：实际 K/V 数据与 layout
```

因此 RadixCache 与 Paged KV 不是竞争关系：前者解决跨请求前缀复用，后者解决物理显存分配与间接寻址。不同 attention backend 可以把同一核心映射转换成自己的 block table、indptr 或 KV indices。

## 面试官可能追问

- [[SGLang RadixCache 如何组织前缀复用、引用与淘汰|RadixCache 如何复用和淘汰？]]
- [[SGLang 的 varlen metadata 如何映射到物理 KV slot|varlen metadata 如何映射到 slot？]]
- [[一个 KV page 内的多个 slot 在显存中是否连续|page 内 slot 是否连续？]]
- [[SGLang 显存不足时如何回收 KV Cache，哪些显存不能被 RadixCache 淘汰|显存不足如何回收？]]

## 关联知识

- [[../../../../wiki/concepts/KV Cache|KV Cache]]
- [[../../../../wiki/concepts/RadixAttention|RadixAttention]]
- [[../../../../wiki/concepts/PagedAttention|PagedAttention]]
- [[../../../../wiki/entities/SGLang|SGLang]]

## 参考来源与待核实

- [SGLang memory pool @ `575759d9`](https://github.com/sgl-project/sglang/blob/575759d90af942eeff89f1c8c33a1fcdb2da4181/python/sglang/srt/mem_cache/memory_pool.py)
- [SGLang paged allocator @ `575759d9`](https://github.com/sgl-project/sglang/blob/575759d90af942eeff89f1c8c33a1fcdb2da4181/python/sglang/srt/mem_cache/allocator/paged.py)
- 混合 Mamba/KDA、SWA、HiCache 和 PD disaggregation 会增加额外 pool/metadata，不能把四层简图当作所有模型的完整对象表。

## 所属题单

- [[../sets/昆仑芯AI Infra一面|昆仑芯 AI Infra 一面]]
- [[../sets/PyTorch图编译与推理系统一面|PyTorch 图编译与推理系统一面]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
