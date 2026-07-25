---
type: concept
topic: GPU 编程
sources: 1
updated: 2026-04-23
---

# Histogram

## 定义

一种把输入元素按桶计数的并行统计算子。在 GPU 上，它常被当作原子操作冲突控制和 shared-memory 私有化的经典题型。

## 它解决什么问题

- 统计离散值分布或桶频次
- 暴露全局 `atomicAdd` 冲突带来的性能问题
- 作为 shared-memory 优化、两阶段归并和 [[Grid-stride Loop]] 的综合练习题

## 核心机制

- 先在 block 内建立 shared-memory 私有直方图
- 每个线程通过 [[Grid-stride Loop]] 处理多项输入，并把局部计数打到 shared histogram
- block 内同步后，再把 shared histogram 合并回全局 histogram
- 通过“先局部、后全局”的两阶段写回降低原子冲突

## 关键权衡

- `BINS` 较大时，shared memory 占用会快速升高
- 如果数据分布极度倾斜，即使在 shared memory 中也可能出现热点桶竞争
- 最优策略与桶数、输入分布和是否支持分层私有化密切相关

## 相关来源

- [[../sources/秋招CUDA手撕题复盘（附代码）]]

## 相关概念

- [[CUDA Kernel]]
- [[Grid-stride Loop]]
- [[Tiling]]

## 研究备注

- 这类题很适合用来解释“为什么正确实现不等于高性能实现”；真正的优化重点通常是冲突模式，而不是索引计算本身
