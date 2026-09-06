# TP8、PP4、DP2 下全局 batch 如何计算？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 本页从历史原稿提炼；本轮未逐题重新核验技术事实或运行代码，不表示答案已通过事实复核。

## 其他问法

- TP=8、PP=4、DP=2 时，global batch 与 micro batch、梯度累积有何关系？
- 3D 并行下，TP=8、PP=4、DP=2 时，global batch size 与 micro batch 的关系

## 30 秒回答

设 micro-batch 大小为 m、梯度累积次数为 GA，global batch 等于 DP×m×GA；题设 DP=2，故为 2mGA。若一次 pipeline flush 的 micro-batch 数 K 对应累积次数，可写 2mK。TP 不切 batch，PP 也不直接乘入公式。

## 深入解释

沿文末原稿章节阅读完整机制、推导或代码；本页保留短答，不复制长篇正文。

## 关联知识

- [[../../../../wiki/concepts/Tensor Parallelism|Tensor Parallelism]]
- [[../../../../wiki/concepts/流水线并行|流水线并行]]

## 参考来源与待核实

- [[../../大模型系统面试题全答#7. 3D 并行下，TP=8、PP=4、DP=2 时，global batch size 与 micro batch 的关系|大模型系统面试题全答]]
- [[../../大模型系统面试题地图#2. 分布式训练与内存账本|大模型系统面试题地图]]

- 待核实 / 原稿边界：原稿将 GA=K 限定为通常的 flush/累积口径；实际应确认调度与更新周期，不能把 TP=8 和 PP=4 都乘入全局 batch。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

## 所属题单

- [[../sets/分布式训练|分布式训练]]
- [[../README|秋招问题汇总]]
