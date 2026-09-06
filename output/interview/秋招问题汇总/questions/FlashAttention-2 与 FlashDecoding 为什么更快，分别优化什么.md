# FlashAttention-2 与 FlashDecoding 为什么更快，分别优化什么？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 本页从历史原稿提炼；本轮未逐题重新核验技术事实或运行代码，不表示答案已通过事实复核。

## 其他问法

- FlashAttention-2 / FlashDecoding 的加速原理？为什么比原生注意力快 3-10 倍
- FlashDecoding 的加速原理是什么？

## 30 秒回答

FlashAttention-2 用分块、online softmax 和本地累积减少 HBM 往返，并改进工作划分，避免物化大中间矩阵。FlashDecoding 面向短 query、长 KV 的 decode，提高 KV 遍历与归并的并行度。两者重点是数据流优化，而非简单减少数学工作量。

## 深入解释

沿文末原稿章节阅读完整机制、推导或代码；本页保留短答，不复制长篇正文。

## 关联知识

- [[../../../../wiki/concepts/FlashAttention|FlashAttention]]
- [[../../../../wiki/concepts/Flash Decoding|Flash Decoding]]

## 参考来源与待核实

- [[../../大模型系统面试题全答#6. FlashAttention-2 / FlashDecoding 的加速原理？为什么比原生注意力快 3-10 倍|大模型系统面试题全答]]
- [[../../大模型系统面试题地图#1. 硬件与性能模型|大模型系统面试题地图]]

- 待核实 / 原稿边界：保留整体对比题，不把两段答案小标题拆题。题设“3–10 倍”无配置和来源，待核实；原稿对 FLOPs 的表述为简化机制口径。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

## 所属题单

- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
