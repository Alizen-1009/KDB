---
type: concept
topic: GPU 编程
sources: 6
updated: 2026-05-06
---

# Tiling

## 定义

把大张量切成更小的数据块并按块计算，使数据能在 shared memory 等更快层级中复用的优化方法。

## 它解决什么问题

- 降低矩阵乘法和 attention 中对 global memory 的重复读取
- 提高 arithmetic intensity，让更多计算在更快的存储层上完成

## 核心机制

- 按 tile 把输入矩阵或张量分块
- 先把 tile 载入 shared memory
- 在块内完成更多局部计算与复用，再继续处理下一个 tile
- 在 attention 场景里，tile 设计还决定“谁留在本地、谁流式经过”；高性能实现往往会固定 `Q` block，并让 `KV` block 按顺序流过本地状态
- 在 stencil / 图像滤波类 kernel 中，tile 往往还需要显式加载 halo 区域，避免窗口访问反复回到 global memory
- 以矩阵乘法为例，naive 实现会让每个线程独立反复从 HBM 读 `A` 的一整行和 `B` 的一整列；tiling 则让 block 协作把 `A/B` 的小块搬进 shared memory，再做多轮复用
- 当 `TILE=32` 时，一个被搬入 shared memory 的元素可被同一 block 中多条线程复用，访存量可近似降到原来的 `1/32` 量级；这也是 GEMM、attention 和归约高度依赖 tiling 的原因
- 它的本质是用 shared memory 的低延迟去承接重复访问，把“多次回 HBM”改成“搬一次，多次复用”

## 关键权衡

- 正确的 tile 设计能同时改善复用和 coalescing
- tile 大小受 shared memory 容量、矩阵维度可整除性和硬件对齐约束影响
- tile 越大，单次搬运后的复用通常越好，但 shared memory 占用也会上升，进而压低 [[Occupancy]]

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 5 - GPUs]]
- [[../sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]
- [[../sources/Flash Attention 详细解释推演与Pytorch代码实现]]
- [[../sources/你一定要知道：CUDA优化六要]]
- [[../sources/CUDA优化维度框架]]
- [[../sources/秋招CUDA手撕题复盘（附代码）]]

## 相关概念

- [[GPU执行模型]]
- [[内存合并访问]]
- [[FlashAttention]]

## 研究备注

- 后续可补 tile size、wave quantization 和矩阵维度对齐如何共同影响 GPU 利用率，以及 attention 中 `Q`/`KV` 不同驻留策略带来的差异
- 现有来源已经覆盖两类典型 tiling：一类是 GEMM / attention 的高复用 tile，另一类是均值滤波这类需要 halo 的 stencil tile
