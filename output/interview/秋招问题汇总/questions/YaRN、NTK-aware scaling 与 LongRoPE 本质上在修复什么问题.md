# YaRN、NTK-aware scaling 与 LongRoPE 本质上在修复什么问题？

- 整理状态：已整理
- 题目出处：既有面试原稿（见文末具体章节）

## 30 秒回答

它们都试图缓解预训练 RoPE 模型在更长位置上的分布外问题，同时尽量保住短上下文表现。NTK-aware 类方法重调频率尺度；YaRN 按频段处理位置插值并配合 attention scaling/微调；LongRoPE 搜索非均匀插值并采用渐进扩展与短文恢复。

## 深入解释

具体公式和是否需要继续训练不相同；不能把三者统称为“简单调大 rope_theta”。

## 关联知识

- [[../../../../wiki/concepts/RoPE|RoPE]]

## 参考来源与待核实

- [YaRN 论文](https://arxiv.org/abs/2309.00071)、[LongRoPE 论文](https://arxiv.org/abs/2402.13753)
- [[../../面试经验#5. 最后看 research tradeoff|面试经验]]


## 所属题单

- [[../sets/模型架构|模型架构]]
- [[../README|秋招问题汇总]]
