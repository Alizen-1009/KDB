# 如何完整实现带 Reduce 的 CUDA RMSNorm？

- 整理状态：待整理
- 题目出处：用户提供的“千问 C 端 AI Infra 一面”Coding 题

## 问题背景

要求写出完整 CUDA 实现，包括沿 hidden dimension 计算平方和的 Reduce、`rsqrt` 归一化、权重缩放、边界处理与 kernel launch；后续整理时应分别验证正确性、数值精度、不同 hidden size 和性能。

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
