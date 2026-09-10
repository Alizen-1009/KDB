---
type: concept
topic: GPU 编程
sources: 9
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

## Work tile 调度

Tiling 不只决定 tile 内如何复用数据，也决定 tile 间如何映射到 CTA 或 thread block cluster。[[Cluster Launch Control]] 来源把常见策略分为：每个 cluster 固定处理一个 tile 的 single-tile scheduling、一波常驻 cluster 按静态步长遍历 tile 的 static persistent scheduling，以及由活跃 cluster 取消未启动 cluster 并接手其 tile 坐标的 CLC dynamic persistent scheduling。后者主要用于 tile 成本不一致时改善负载均衡，但动态顺序也可能改变 L2 局部性。

## Head dimension 与 pipeline tile

PAI-FA 来源表明，tile shape 与 pipeline stage 不能分开选择。`head_dim=256` 时，`128×128` attention tile 的 `O/dQ/dK/dV` accumulator footprint 相比 `head_dim=128` 翻倍；若继续保留原双缓冲 stage，会超出 [[Tensor Memory|TMEM]]。该实现保留较大的计算 tile 以维持 arithmetic intensity，但将 Forward Q stage 降为 1；Backward 则让 `dQ` 使用 `128×128`、`dKdV` 使用 `128×64`，以不同 loop ownership 适配各自累加目标。

## KDA chunk 16 与 chunk 32

FlashKDA 现有资料使用 16-token chunk，把 lower-bounded cumulative decay 控制在 BF16 范围并直接适配 `m16n8k16` 小矩阵路径。[[../entities/CAKE KDA]] 通过固定 exponent anchor 把 chunk 扩到 32：anchor 在 `Mqk` 中抵消，使每个 BF16 Q/K operand 的指数半径接近未居中 chunk 16；更大 chunk 减少 recurrence 次数，但需要 `32×32` triangular/inverse 处理、五级 SMEM look-ahead 和更高单 CTA 资源占用。

CAKE 同时提供 M64/M128 physical schedules：M128 增加单 CTA 工作量和 Tensor Core 复用，M64 保留更多 grid parallelism。选择不能只看单 CTA 峰值，还要结合 batch×heads 是否足以覆盖 SM。

## 关键权衡

- 正确的 tile 设计能同时改善复用和 coalescing
- tile 大小受 shared memory 容量、矩阵维度可整除性和硬件对齐约束影响
- tile 越大，单次搬运后的复用通常越好，但 shared memory 占用也会上升，进而压低 [[Occupancy]]

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/NVIDIA Blackwell]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 5 - GPUs]]
- [[../sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]
- [[../sources/Flash Attention 详细解释推演与Pytorch代码实现]]
- [[../sources/你一定要知道：CUDA优化六要]]
- [[../sources/CUDA优化维度框架]]
- [[../sources/秋招CUDA手撕题复盘（附代码）]]
- [[../sources/Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs]]
- [[../sources/PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化]]
- [[../sources/REMINDER FF-KDA & CAKE KDA Highlights]]

## 相关概念

- [[GPU执行模型]]
- [[内存合并访问]]
- [[FlashAttention]]
- [[Cluster Launch Control]]
- [[Tensor Memory]]
- [[KDA]]

## 研究备注

- 后续可补 tile size、wave quantization 和矩阵维度对齐如何共同影响 GPU 利用率，以及 attention 中 `Q`/`KV` 不同驻留策略带来的差异
- 现有来源已经覆盖两类典型 tiling：一类是 GEMM / attention 的高复用 tile，另一类是均值滤波这类需要 halo 的 stencil tile
