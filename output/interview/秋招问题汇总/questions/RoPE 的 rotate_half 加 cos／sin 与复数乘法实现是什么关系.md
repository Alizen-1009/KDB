# RoPE 的 rotate_half 加 cos／sin 与复数乘法实现是什么关系？

- 整理状态：已整理
- 题目出处：既有面试原稿（见文末具体章节）

## 其他问法

- RoPE 的 rotate_half 加 cos/sin 与复数乘法实现是什么关系？

## 30 秒回答

对每对 (a,b)，rotate_half 给出 (−b,a)，所以 x cosφ + rotate_half(x) sinφ = (a cosφ−b sinφ, a sinφ+b cosφ)，正是 (a+ib)e^{iφ} 的实部/虚部。

## 深入解释

相邻配对和 half-split 配对都可实现；必须同步调整 rotate_half、cos/sin 与张量布局。

## 关联知识

- [[../../../../wiki/concepts/RoPE|RoPE]]

## 参考来源与待核实

- [RoFormer 原论文](https://arxiv.org/abs/2104.09864)
- [[../../面试经验#3. 再看有没有实现感|面试经验]]


## 所属题单

- [[../sets/模型架构|模型架构]]
- [[../README|秋招问题汇总]]
