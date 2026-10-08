# partial_rotary_factor 如何决定 RoPE 旋转维度？

- 整理状态：已整理
- 题目出处：既有面试原稿（见文末具体章节）

## 其他问法

- partial_rotary_factor 是什么，如何决定 head_dim 中参与 RoPE 的维度？
- 为什么有的模型 partial_rotary_factor 不是 1.0？
- partial_rotary_factor=0.25 时，代码如何切张量？
- 多头注意力里 RoPE 作用在 head_dim 的哪些维度上？

## 30 秒回答

它指定 head_dim 中参与旋转的比例；常见实现取 rotary_dim≈head_dim×factor，并调整为可按两维配对的偶数，剩余维度原样通过。它改变承载位置信息的维度份额，不改变 head_dim 总大小。

## 深入解释

具体整数取整、是否对 Q/K 使用相同维数要以模型配置和实现为准；不能把 factor 当成 theta。

## 关联知识

- [[../../../../wiki/concepts/RoPE|RoPE]]

## 参考来源与待核实

- [RoFormer 原论文](https://arxiv.org/abs/2104.09864)
- [[../../面试经验#2. 再看能不能讲清数学直觉|面试经验]]
- [[../../面试经验#3. 再看有没有实现感|面试经验]]


## 所属题单

- [[../sets/模型架构|模型架构]]
- [[../README|秋招问题汇总]]
