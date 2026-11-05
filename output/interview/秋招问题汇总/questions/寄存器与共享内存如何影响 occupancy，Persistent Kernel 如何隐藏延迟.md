# 寄存器与共享内存如何影响 occupancy，Persistent Kernel 如何隐藏延迟？

- 整理状态：已整理
- 题目出处：用户提供的 Kimi AI Infra 一面复盘

## 30 秒回答

一个 SM 的寄存器和 shared memory 都有限；单个 CTA 用得越多，能同时驻留的 CTA/warp 可能越少，等待内存时可切换的 warp 也更少。但高 occupancy 不是目标本身，较大的 tile 和更多寄存器也可能增加复用或 ILP。Persistent Kernel 让一批 CTA 长期处理多个 work tiles，复用调度和流水状态，并把下一 tile 的搬运与当前 tile 的计算、上一 tile 的 epilogue 重叠。它本身不自动隐藏延迟；仍要有足够独立工作、异步流水与合理的资源占用。

## 关联知识

- [[../../../../wiki/concepts/Occupancy|Occupancy]]
- [[../../../../wiki/concepts/Persistent Kernel|Persistent Kernel]]

## 参考来源与待核实

- [CUDA Occupancy 说明](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#occupancy)；[CUTLASS Grouped Kernel Scheduler](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/grouped_scheduler.md)。

## 面试追问：Persistent Matmul 中一个 program 算哪些 tile？

Triton 教程示例按固定步长遍历 tile。假设有 4 个 program 和 12 个 tile，program 0 处理 0、4、8。program 在一次 launch 内连续处理多个 tile，每个 tile 的累加器重新初始化。固定步长分配无需共享任务队列。跨 launch 保存状态属于不同机制。

- [Triton Persistent Matmul](https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py)

## 所属题单

- [[../sets/推理并行、昇腾调优与 MLIR 面试追问|推理并行、昇腾调优与 MLIR 面试追问]]
- [[../sets/Kimi AI Infra一面|Kimi AI Infra 一面]]
- [[../README|秋招问题汇总]]
