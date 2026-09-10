# Speculative Decoding 的原理是什么，Medusa、Lookahead 与 EAGLE 有何区别？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 本页从历史原稿提炼；本轮未逐题重新核验技术事实或运行代码，不表示答案已通过事实复核。

## 其他问法

- Speculative Decoding 原理？Medusa、Lookahead Decoding、EAGLE 区别

## 30 秒回答

投机解码先提出多个候选 token，再由主模型批量验证、接受部分候选并在拒绝时回退。原稿将经典方案概括为小模型猜、大模型验；Medusa 用主模型上的多预测头，Lookahead 偏前瞻候选展开，EAGLE 偏轻量辅助预测器的 hidden-level 预测。

## 深入解释

沿文末原稿章节阅读完整机制、推导或代码；本页保留短答，不复制长篇正文。

## 关联知识

- [[../../../../wiki/concepts/Speculative Decoding|Speculative Decoding]]

## 参考来源与待核实

- [[../../大模型系统面试题全答#7. Speculative Decoding 原理？Medusa、Lookahead Decoding、EAGLE 区别|大模型系统面试题全答]]
- [[../../大模型系统面试题地图#3. 推理系统与服务编排|大模型系统面试题地图]]

- 待核实 / 原稿边界：原稿为高层比较，未给版本、接受规则与分布保证，不能据此把所有变体都认定为严格无损或同一实现。保留比较题整体。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

## 所属题单

- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
