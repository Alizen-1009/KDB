# 如何实现高性能行主序 GEMV？

- 整理状态：已整理
- 题目出处：用户提供的百度 Infra 面试复盘；题目为 `A[M,N] × B[N,1]`

## 30 秒回答

先确认 A 的布局。行主序下 `y[i]=Σⱼ A[i,j]B[j]`，本质是 M 个按行 dot/reduction。一个起点是每个 warp 负责一行，各 lane 沿连续 N 维读 A 和 B、各自累加，再用 warp shuffle 归约；一个 CTA 可处理多行。A 通常只读一次，B 会跨行复用，先观察 cache 效果，不必默认搬到 shared memory。再按 M、N 调整每行 warp 数、每线程累加量和 CTA 数；若 M 很小而 N 很大，可沿 N 拆分一行、写 partial sums 后二次归约以增加并行度。最终用带宽、延迟和强基线实测。

## 深入解释

例如行主序 FP32 `A[4096,4096]` 约 64 MiB，`B[4096]` 约 16 KiB。A 的单次读取通常主导数据量，基础版本算术强度低；但小 M、小 N 或数据命中 cache 时，launch、归约和并行度可能主导。若 A 是列主序，线程映射必须重选，不能照搬“一 warp 一行”。

## 常见误区

- 数学上 GEMV 是 `N_out=1` 的 GEMM，但直接套大 GEMM tile 会浪费输出维并行度。
- “B 有复用”不自动证明 shared memory 更快；要比较 cache 命中与显式搬运、同步、容量成本。

## 关联知识

- [[../../../../wiki/concepts/内存合并访问|内存合并访问]]
- [[../../../../wiki/concepts/Warp Shuffle Reduce|Warp Shuffle Reduce]]
- [[../../../../wiki/concepts/Block Reduce|Block Reduce]]
- [[Prefill 与 decode 更像 GEMM 还是 GEMV，瓶颈为什么不同|推理中的 GEMV]]

## 参考来源与待核实

- [CUDA Best Practices：Coalesced Access 与 Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。固定示例用于算账，不代表已编译或 benchmark 的 CUDA 实现。

## 所属题单

- [[../sets/百度Infra面试|百度 Infra 面试]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
