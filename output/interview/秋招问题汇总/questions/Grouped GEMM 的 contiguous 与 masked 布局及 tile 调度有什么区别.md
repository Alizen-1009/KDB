# Grouped GEMM 的 contiguous 与 masked 布局及 tile 调度有什么区别？

- 整理状态：已整理
- 题目出处：用户提供的 Kimi AI Infra 一面复盘

## 30 秒回答

先限定 DeepGEMM 的 MoE grouped GEMM：各 expert 的 N、K 相同，M 是收到的 token 数。`contiguous` 把各 expert 的有效 token 段拼接起来，通常要按 M tile 对齐，适合能确定段长度的训练/prefill。`masked` 给每个 expert 固定容量和设备侧有效长度，kernel 只计算有效 tile，适合 decode 加 CUDA Graph、CPU 不知道实时接收数的场景。调度器把 `(expert, M tile, N tile)` 映射给 CTA；要按实际 token 数控制空 tile 和负载不均，同时考虑 expert 权重、A 数据的缓存局部性。tile 大小需按 M 分布、N/K、对齐、资源占用和尾块浪费实测。

## 深入解释

这里的 `contiguous/masked` 是 DeepGEMM 的接口/布局术语，不是所有 grouped GEMM 库的统一分类。一般 grouped GEMM 可以有各不相同的 M/N/K；DeepGEMM 此处的 M-grouped API 更窄。

## 关联知识

- [[../../../../wiki/concepts/Tiling|Tiling]]
- [[GEMM 常用的 GPU 优化手段有哪些|GEMM 优化]]

## 参考来源与待核实

- [DeepGEMM README，Grouped GEMMs](https://github.com/deepseek-ai/DeepGEMM/blob/main/README.md)；[CUTLASS Grouped Kernel Scheduler](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/grouped_scheduler.md)。

## 所属题单

- [[../sets/Kimi AI Infra一面|Kimi AI Infra 一面]]
- [[../README|秋招问题汇总]]
