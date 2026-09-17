# 如何用 Triton 实现并优化 Softmax？

- 整理状态：待整理
- 题目出处：用户提供的“千问 C 端 AI Infra 一面”Coding 题

## 问题背景

要求实现一个 Triton Softmax；后续整理时应覆盖数值稳定的减最大值、mask、行级并行、block size、访存次数、长行处理，并与 PyTorch reference 做正确性和性能验证。

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
