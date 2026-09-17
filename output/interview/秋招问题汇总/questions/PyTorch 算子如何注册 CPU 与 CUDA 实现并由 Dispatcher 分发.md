# PyTorch 算子如何注册 CPU 与 CUDA 实现并由 Dispatcher 分发？

- 整理状态：待整理
- 题目出处：用户提供的“千问 C 端 AI Infra 一面”面经

## 其他问法

- Torch 算子注册过程是什么？
- CPU、CUDA 算子如何分别注册，Dispatcher 如何选择实现？

## 问题背景

后续答案需要区分算子 schema、CPU/CUDA kernel 注册、dispatch key、autograd/meta/fake 实现，以及 Python custom op 与 C++ extension 等不同入口。

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
