# 一个 KV page 内的多个 slot 在显存中是否连续？

- 整理状态：已整理
- 题目出处：用户提供的一面面经（SGLang / CUDA 系统专题）

## 30 秒回答

在 SGLang 当前 paged allocator 的索引语义里，一个 page 内的 slot 是连续编号的：`slot = page_id * page_size + offset`，同一页内 offset 从 `0` 到 `page_size-1`。但一个请求的相邻逻辑 page 可以来自不同 free page，因此跨 page 不保证连续；而且“slot 连续”只表示沿 KV pool 的 slot 维连续，实际字节地址还要乘 tensor stride，不能说所有层、K/V、head 的数据都挤在一段无间隔字节里。

## 深入解释

假设 `page_size=4`，请求拿到物理 page `3` 和 `9`：

```text
逻辑 token 0..3 -> slots 12,13,14,15
逻辑 token 4..7 -> slots 36,37,38,39
```

所以：

- 页内 slot 连续：是。
- 请求的跨页物理地址连续：不一定。
- 不同请求的页面可以交错：可以。
- K cache 与 V cache 的完整字节布局完全相同：不一定，取决于 pool layout 和 kernel 访问方向。

## 面试官可能追问

- page size 变大为何降低管理开销，却增加尾页内部碎片？
- `slot_mapping` 更偏读路径还是写路径？
- block table、page ID、slot ID 各是什么关系？

## 关联知识

- [[../../../../wiki/concepts/PagedAttention|PagedAttention]]
- [[SGLang 的 varlen metadata 如何映射到物理 KV slot|varlen metadata 与 slot 映射]]

## 参考来源与待核实

- [SGLang `PagedTokenToKVPoolAllocator`](https://github.com/sgl-project/sglang/blob/575759d90af942eeff89f1c8c33a1fcdb2da4181/python/sglang/srt/mem_cache/allocator/paged.py)
- 结论限定于该 commit 的 allocator 索引语义；不同 backend 的 K/V tensor layout 和量化 packing 需要分别核对。

## 所属题单

- [[../sets/SGLang与CUDA系统一面|SGLang 与 CUDA 系统一面]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
