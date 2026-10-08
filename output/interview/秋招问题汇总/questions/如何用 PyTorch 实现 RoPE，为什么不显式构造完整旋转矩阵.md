# 如何用 PyTorch 实现 RoPE，为什么不显式构造完整旋转矩阵？

- 整理状态：已整理
- 题目出处：既有面试原稿（见文末具体章节）

## 其他问法

- 你会怎么在 PyTorch 中实现 RoPE？

## 30 秒回答

先算每对维度的频率和 position×frequency，用 cos/sin 广播到 Q/K；再计算 x*cos + rotate_half(x)*sin。每个位置的变换是稀疏块对角二维旋转，直接广播只需 O(d) 运算/系数，显式构造 d×d 矩阵会浪费存储和计算。

## 深入解释

示意：`freq = 1 / (theta ** (arange(0,d,2)/d))`，`angle = position_ids[...,None]*freq`；接着按配对布局展开 cos/sin。真实代码还要处理 batch/head 广播、partial rotary 和 dtype。

## 关联知识

- [[../../../../wiki/concepts/RoPE|RoPE]]

## 参考来源与待核实

- [RoFormer 原论文](https://arxiv.org/abs/2104.09864)
- [[../../面试经验#3. 再看有没有实现感|面试经验]]


## 所属题单

- [[../sets/模型架构|模型架构]]
- [[../README|秋招问题汇总]]
