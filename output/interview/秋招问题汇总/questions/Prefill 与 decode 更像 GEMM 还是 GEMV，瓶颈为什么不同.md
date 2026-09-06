# Prefill 与 decode 更像 GEMM 还是 GEMV，瓶颈为什么不同？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 本页从历史原稿提炼；本轮未逐题重新核验技术事实或运行代码，不表示答案已通过事实复核。

## 其他问法

- prefill 和 decode 是 GEMM 还是 GEMV？
- SM、warp 和 LLM 的 prefill/decode 有什么关系？
- Prefill 和 Decode 的 KV Cache 与计算压力为什么不同？

## 30 秒回答

Prefill 线性层输入常是 B×S 行，形成大 GEMM，较容易提高并行度和算力利用率。Decode 输入只有 B 行，小 batch 时更像 GEMV 或小 GEMM，还要读历史 KV，因此常偏访存瓶颈。连续批处理能放大线性层矩阵，但不消除注意力与 KV 路径的带宽压力。

## 深入解释

沿文末原稿章节阅读完整机制、推导或代码；本页保留短答，不复制长篇正文。

## 参考来源与待核实

- [[../../多卡GPU监控与SM执行模型面试整理#4. SM、warp 和 LLM 的关系|多卡GPU监控与SM执行模型面试整理]]
- [[../../量化剪枝推理瓶颈Nsight与异构集群面试整理#12.3 prefill 和 decode 是 GEMM 还是 GEMV|量化剪枝推理瓶颈Nsight与异构集群面试整理]]
- [[../../面试经验#5. KV cache 的缓存和计算|面试经验]]
- [[../../推理系统专题面试稿#2. 请详细解释 KV Cache 的核心原理，为什么它能显著加速大模型推理？|推理系统专题面试稿]]

- 待核实 / 原稿边界：比较题保持整体，并聚合两份来源。原稿使用通常、更容易等条件表达，不可写成 prefill 必然 compute-bound、decode 必然 GEMV/memory-bound；需绑定 batch、序列长和具体 kernel。 原稿使用“更像”“很多时候”等定性判断；不能把所有 prefill 固定判为 compute-bound，或把所有 decode 固定判为 memory-bound。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

## 所属题单

- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
