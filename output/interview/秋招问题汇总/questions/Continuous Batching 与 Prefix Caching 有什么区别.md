# Continuous Batching 与 Prefix Caching 有什么区别？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 本页从历史原稿提炼；本轮未逐题重新核验技术事实或运行代码，不表示答案已通过事实复核。

## 30 秒回答

原稿将 Continuous Batching 归为调度策略，将 Prefix Caching 归为跨请求复用策略。比较时应分别看请求如何参与执行，以及不同请求之间如何复用已有内容，而不能把批处理和前缀复用当作同一种优化或同一层面的概念。

## 深入解释

沿文末原稿章节阅读完整机制、推导或代码；本页保留短答，不复制长篇正文。

## 关联知识

- [[../../../../wiki/concepts/Prefix Caching|Prefix Caching]]
- [[../../../../wiki/concepts/Continuous Batching|Continuous Batching]]

## 参考来源与待核实

- [[../../大模型系统面试题地图#5. Continuous Batching vs Prefix Caching|大模型系统面试题地图]]

- 待核实 / 原稿边界：地图只有两条概念区分；不据此补写前缀哈希、淘汰策略或命中收益。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

## 所属题单

- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
