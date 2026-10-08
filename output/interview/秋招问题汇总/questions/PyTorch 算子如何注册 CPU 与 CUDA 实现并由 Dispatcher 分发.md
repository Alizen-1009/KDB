# PyTorch 算子如何注册 CPU 与 CUDA 实现并由 Dispatcher 分发？

- 整理状态：已整理
- 题目出处：用户提供的“千问 C 端 AI Infra 一面”面经

## 其他问法

- Torch 算子注册过程是什么？
- CPU、CUDA 算子如何分别注册，Dispatcher 如何选择实现？

## 问题背景

后续答案需要区分算子 schema、CPU/CUDA kernel 注册、dispatch key、autograd/meta/fake 实现，以及 Python custom op 与 C++ extension 等不同入口。

## 30 秒回答

先用 `TORCH_LIBRARY` 定义算子 schema，再分别用 `TORCH_LIBRARY_IMPL(namespace, CPU, m)` 和 `TORCH_LIBRARY_IMPL(namespace, CUDA, m)` 注册同名实现。调用 `torch.ops.namespace.op` 时，Dispatcher 根据张量的 dispatch key 选择后端实现；autograd、meta/fake tensor 还可能需要单独注册。

## 深入解释

schema、设备实现、梯度和 shape 推断是不同层；只写 CUDA kernel 并不会自动成为可调用 PyTorch 算子。

## 参考来源与待核实

- [PyTorch Dispatcher 教程](https://docs.pytorch.org/tutorials/advanced/dispatcher)

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
