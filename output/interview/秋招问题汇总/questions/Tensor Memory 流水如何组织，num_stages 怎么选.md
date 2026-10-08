# Tensor Memory 流水如何组织，num_stages 怎么选？

- 整理状态：已整理
- 题目出处：用户提供的 Kimi AI Infra 一面复盘

## 30 秒回答

在 Blackwell 的典型 GEMM 中，TMA 把 A/B 的下一个 K tile 搬入 shared memory，`tcgen05.mma` 消费当前 tile，累加结果留在 Tensor Memory，之后再把结果读出做 epilogue。用 buffer 和 barrier 保证“写完才能算、算完才能复用”，从双缓冲开始，让 load、MMA、epilogue 尽量重叠。`num_stages` 不是越大越好：增加 stage 能隐藏搬运延迟，也会多占 shared memory、barrier 等资源，可能减少 resident CTA。固定 shape 下扫 stage 数，结合 stall、资源占用和 kernel latency 选择。

## 深入解释

TMEM 主要服务 MMA 的 operand/accumulator 路径；`num_stages` 通常指 shared-memory 中预取 tile 的流水深度，不等于“TMEM 里放了几个完整 A/B tile”。结果回写仍要经过受控的数据消费和 epilogue。

## 关联知识

- [[../../../../wiki/concepts/Tensor Memory|Tensor Memory]]
- [[../../../../wiki/concepts/Occupancy|Occupancy]]
- [[../../../../wiki/concepts/Tiling|Tiling]]

## 参考来源与待核实

- [NVIDIA CUTLASS Pipeline Types](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_pipelines.html)；[Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)。
- 具体 TMEM 分配、同步和 stage 上限依 kernel/SM target 而定，不能给通用最优数值。

## 所属题单

- [[../sets/Kimi AI Infra一面|Kimi AI Infra 一面]]
- [[../README|秋招问题汇总]]
