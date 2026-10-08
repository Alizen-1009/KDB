# 什么数据适合放进 CUDA shared memory？

- 整理状态：已整理
- 题目出处：用户提供的百度 Infra 面试复盘

## 30 秒回答

我主要看三件事：数据在一个 CTA 内是否会被多次使用、是否需要重排以实现合并的 global memory 访问、线程之间是否需要交换中间结果。GEMM 的 A/B tile、转置或 stencil 的数据重排、block reduce 的部分和都是典型例子。若数据只读一次，直接 global load 已合并且 cache 命中良好，显式搬进 shared memory 往往增加搬运和同步，还会占容量、压低 occupancy。收益必须覆盖这些成本。

## 深入解释

- 线程私有的小中间量优先考虑寄存器；CTA 内协作和复用才考虑 shared memory；跨 CTA 的复用主要看 L2/cache 和调度，shared memory 不能直接跨 block 共享。
- Shared memory 也可用于 producer/consumer buffer 与归约通信，因此“只有 reuse 才能用”过于狭窄。布局还需避免 bank conflict。

## 关联知识

- [[../../../../wiki/concepts/CUDA内存层次|CUDA 内存层次]]
- [[../../../../wiki/concepts/Tiling|Tiling]]
- [[../../../../wiki/concepts/Bank Conflict|Bank Conflict]]
- [[../../../../wiki/concepts/Block Reduce|Block Reduce]]

## 参考来源与待核实

- [CUDA Best Practices：Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。

## 所属题单

- [[../sets/百度Infra面试|百度 Infra 面试]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
