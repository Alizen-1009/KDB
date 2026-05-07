# PagedAttention

## 定义

把 KV Cache 按固定大小分页并通过索引表访问的显存管理与注意力执行方式，用于减少碎片并提升动态调度能力。

## 它解决什么问题

- 降低 KV Cache 的显存碎片问题
- 支持动态增长、共享和更灵活的批处理
- 避免传统按 `max_seq_len` 连续预分配导致的大量预留槽位和难复用空间浪费

## 核心机制

- 将 KV 张量切成固定大小 page
- 通过映射表而不是连续物理地址来定位 page
- 让请求生命周期和显存布局解耦
- 支持 prefix sharing、copy-on-write 和更适合动态请求流的块级管理
- 从运行时视角看，它通常可拆成三层对象：逻辑上的 `logical KV blocks`、显存中的 `physical KV blocks`，以及维护两者映射和填充状态的 `block table`
- `prefill` 先把 prompt 对应的 KV 写入若干逻辑块并映射到物理块；`decode` 再通过 `block table` 读取历史 KV，并把新 token 追加到已有块或新分配的块中
- 当多个序列共享同一前缀时，不同序列的 `block table` 可以同时指向同一批 prefix 物理块；只有在后续写入分叉时，才通过 `Copy-on-Write` 复制出新块
- 在 `vLLM V1` 的统一调度语境里，PagedAttention/KV block 管理仍是 token-level scheduler 能工作的底层条件：调度器决定本轮处理多少 token，KV cache manager 则决定这些 token 对应的 block 是否可分配、复用或需要触发抢占/重计算。

## 关键权衡

- 提高显存利用率和动态调度灵活性
- 需要额外的页表与运行时管理复杂度
- 把“连续地址访问”变成“块表寻址”，因此 kernel 与调度器都要适配这种间接访问模式
- 和 [[RadixAttention]] 对比时，`PagedAttention` 不是“复杂任务编排”机制，而是底层 KV cache 显存管理机制；它可以支撑 prefix sharing 和动态 batch，但抽象重心仍在 serving runtime。

## 相关实体

- [[../entities/vLLM]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/斯坦福CS336 Lecture 10 - Inference systems and optimization]]
- [[../sources/美团一面：请介绍 vLLM PageAttention]]
- [[../sources/vLLM v0 与 vLLM v1 调度架构差异截图整理]]
- [[../sources/SGLang 与 vLLM 区别截图整理]]

## 相关概念

- [[Continuous Batching]]
- [[KV Cache]]
- [[Prefix Caching]]
- [[缓存感知路由]]
- [[vLLM V1 统一调度器]]
- [[SGLang 与 vLLM 对比]]
- [[RadixAttention]]

## 研究备注

- 后续可补 vLLM 具体的 page/block 抽象、调度器配合方式与碎片率收益
- 现有来源已经能支撑一版比较好的面试回答：不仅能说“像虚拟内存”，还可以把 `block table`、`prefill/decode` 和块填充过程讲出来
- 在 Beam Search 或 prefix sharing 场景下，`Copy-on-Write` 是高频追问点：它的价值不只是正确性，还在于避免 beam 或共享前缀的 KV cache 线性膨胀
- 截图中关于 `PagedAttention` 降低碎片率的方向是对的，但具体百分比需要回到论文原文核对，不宜脱离 benchmark 直接复述
