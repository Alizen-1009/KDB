# Ring Attention 与 AllGather KV 的瓶颈、收益和退化场景是什么？

- 整理状态：已整理
- 题目出处：用户本次提供

## 30 秒回答

Ring 固定本地 Q，分轮传递 KV 块并以 online softmax 合并，可边传边算，避免每 rank 一次持有完整 KV；AllGather KV 先收齐完整 KV，再对本地 Q 计算，易调用成熟 collective，但有完整 KV 峰值和等待。Ring 适合长序列、单轮计算足以掩盖通信且内存紧张；短序列、小块、rank 多或跨节点延迟大时多轮启动和依赖链会退化。Causal 负载不均、拓扑、通信带宽和 kernel 效率都要实测，不能断言 Ring 必然更快。

依据：[[../../../../wiki/concepts/Ring Attention|Ring Attention]]（该页注明复杂度及具体实现待核实）；[[Context Parallel 中 AllGather KV 与 AllGather Q 如何选择|AllGather Q 与 KV]]。

## 所属题单

- [[../sets/长上下文与全局缓存调度面试题|长上下文与全局缓存调度面试题]]
