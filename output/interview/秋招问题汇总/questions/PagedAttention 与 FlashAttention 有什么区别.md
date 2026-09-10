# PagedAttention 与 FlashAttention 有什么区别？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 本页从历史原稿提炼；本轮未逐题重新核验技术事实或运行代码，不表示答案已通过事实复核。

## 30 秒回答

PagedAttention 主要解决 KV Cache 的显存管理与碎片问题，FlashAttention 主要解决 attention kernel 的 IO 问题。原稿强调二者优化对象不同：一个关注缓存如何存放和管理，另一个关注计算过程如何减少数据搬运，不能仅因名字相近就混淆。

## 深入解释

沿文末原稿章节阅读完整机制、推导或代码；本页保留短答，不复制长篇正文。

## 关联知识

- [[../../../../wiki/concepts/FlashAttention|FlashAttention]]
- [[../../../../wiki/concepts/PagedAttention|PagedAttention]]

## 参考来源与待核实

- [[../../大模型系统面试题地图#4. PagedAttention vs FlashAttention|大模型系统面试题地图]]

- 待核实 / 原稿边界：保留比较题整体；地图没有性能数字，也没有给特定框架的组合实现。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

## 所属题单

- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
