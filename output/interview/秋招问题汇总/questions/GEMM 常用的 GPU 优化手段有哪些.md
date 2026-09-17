# GEMM 常用的 GPU 优化手段有哪些？

- 整理状态：已整理
- 题目出处：用户提供的一面面经（PyTorch 图编译与推理系统专题）

## 30 秒回答

GEMM 优化先固定 `M/N/K`、dtype、layout 和目标硬件，再从数据搬运与计算映射入手：做 CTA/warp/MMA 多级 tiling，把 A/B tile 搬到 shared memory 并在寄存器复用；使用合并、对齐和向量化访问，避免 bank conflict；用 double buffering、`cp.async`/TMA 重叠搬运与 MMA；选择 Tensor Core 指令和合适的 warp 分工；最后通过 epilogue fusion、split-K 或 persistent scheduling处理特定 shape。tile 不是越大越好，还要与寄存器、shared memory、occupancy 和尾块浪费一起权衡。

## 深入解释

常见检查顺序：

1. 数学和 layout：是否能转置/重排以形成连续访问，是否命中 Tensor Core dtype/shape。
2. 多级分块：global→shared→register，增加同一 A/B tile 的复用。
3. 流水：多 stage、异步 copy、ping-pong buffer，把 load latency 藏在计算后面。
4. 线程映射：warp tile、MMA atom、K loop、边界处理和负载均衡。
5. 融合：bias、activation、quant/dequant、residual 等尽量进入 epilogue，但不能因寄存器压力导致 spill。
6. 实测：先比较 cuBLASLt/CUTLASS 等强基线，再用 profiler 看 Tensor Core、DRAM、stall、occupancy 和 wave/tail。

## 关联知识

- [[../../../../wiki/concepts/Tiling|Tiling]]
- [[../../../../wiki/concepts/算子融合|算子融合]]
- [[CUDA Thrust、CUB 与 CUTLASS 各自解决什么问题|CUTLASS]]

## 参考来源与待核实

- 本页是稳定的优化路线图，不代表任意 shape 都应使用 split-K、persistent 或相同 tile；最终以目标 GPU 和 shape sweep 为准。

## 所属题单

- [[../sets/PyTorch图编译与推理系统一面|PyTorch 图编译与推理系统一面]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
