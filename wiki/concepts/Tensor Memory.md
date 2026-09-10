---
type: concept
topic: GPU 编程
sources: 3
updated: 2026-08-21
---

# Tensor Memory

## 定义

`Tensor Memory（TMEM）` 是 [[../entities/NVIDIA Blackwell]] Tensor Core 数据路径中的专用片上存储，用于承载 `tcgen05` MMA 的操作数或 accumulator，缓解大型矩阵运算对通用 register file 的压力。

## 它解决什么问题

- 更大 MMA tile 和更多并行 accumulator 会迅速增加寄存器需求，压缩 [[Occupancy]]，甚至导致 spill。
- Warp-specialized kernel 希望让 Tensor Core 计算与 CUDA Core 的 Softmax、correction、epilogue 并行；若所有中间矩阵都占用普通寄存器，两类执行阶段会争夺同一 RF 容量。
- [[FlashAttention]] Backward 同时涉及 `S/dP/dS/dQ/dK/dV` 等中间状态，大 head dimension 会放大 accumulator footprint，需要显式规划其生命周期与复用。

## 核心机制

- 来源将每个 SM 的 TMEM 描述为 `128 rows × 512 cols × 4 bytes = 256KB`，高度固定为 128 行，按列分配，支持的列宽为 `32/64/128/256/512`。
- `tcgen05.mma` 可从 SMEM/TMEM 取得操作数并把累加结果保存在 TMEM，避免由整个 warp group 长时间持有大块 register accumulator。
- TMEM 需要显式分配、释放和搬运；访问具有 warp/lane 协作约束，不是普通 thread-private register 数组。
- 来源描述的数据消费路径通常是 `TMEM → RF`，最终写回 GMEM 前还需经过 RF/SMEM 与 epilogue，不能把 TMEM 当作可直接输出到 global memory 的通用缓存。

## 与其他片上存储的区别

| 存储 | 典型作用域与管理 | FA4 中的典型用途 |
| --- | --- | --- |
| Register / RF | thread 私有，主要由编译器分配 | Softmax、correction、地址与标量状态 |
| Shared Memory / SMEM | CTA 内显式共享 | Q/K/V staging、TMA 落点、CTA 内重排 |
| Distributed Shared Memory / DSMEM | thread block cluster 内跨 CTA 访问 | CTA Pair 共享操作数、交换部分 `dS` |
| Tensor Memory / TMEM | Tensor Core 专用、显式列分配 | `S/O/dQ/dK/dV` 等 MMA accumulator |
| L1/L2/HBM | 硬件缓存或全局显存 | 全局输入输出、跨 tile 数据来源 |

## 从 Hopper register accumulator 到 Blackwell TMEM

翻译来源把代际变化概括为：[[../entities/NVIDIA Hopper]] WGMMA consumers 需要在普通 register 中长期持有大块 accumulator，容易推高 RF footprint；[[../entities/NVIDIA Blackwell]] 的 `tcgen05` 则把 accumulator 放入 TMEM，使 MMA producer 与 CUDA Core/epilogue consumer 更容易解耦，并让 [[Persistent Kernel]] 跨 tiles 保持更深的软件流水。

这不代表结果可从 TMEM 直接任意写入 GMEM。实际消费通常仍需 `TMEM → RF/SMEM → GMEM`，且 TMEM 支持的指令、layout、分配粒度与访问方向比普通寄存器或 SMEM 更受限制。

## HeadDim=256 的容量压力

- 对 `128×128` attention tile，`head_dim` 从 128 增至 256 时，`S` 的形状不变，但 `O/dQ/dK/dV` 等沿 head dimension 展开的 accumulator footprint 翻倍。
- PAI-FA Forward 因此把 Q stage 从 2 降到 1，只保留一个 O tile，并用同一 Q 下不同 K tile 的 ping-pong 维持流水重叠。
- Backward 则拆成 `dQ` 与 `dKdV` 两个 kernel，避免 `dQ/dK/dV` 同时竞争 TMEM；代价是重算由约 5 次 MMA 增至约 7 次 MMA。

## KDA recurrent state 常驻

[[../entities/CAKE KDA]] 的 M128 kernel 使用 256 个 TMEM columns 保存 FP32 recurrent state、output、accumulator 和临时 operands。state 跨 32-token chunks 保持驻留，consumer 按递归顺序更新；chunk-local `Mqk`、inverse、centered Q/K 等则在 register/SMEM/TMEM 间流动。该方案消除了 preparation 与 recurrent kernel 之间的 chunk-local workspace HBM 往返，但长期占用 TMEM 也限制了可共存 accumulator、pipeline stage 和调度选择。

## 关键权衡

- TMEM 能释放普通寄存器压力并支持 Tensor Core/CUDA Core 重叠，但它本身容量固定，仍可能成为更大 head dimension 或更多 pipeline stage 的瓶颈。
- 更深 stage 能提高重叠，但会同时驻留更多 accumulator；最优 stage 取决于 MMA、Softmax、load/epilogue 的实测耗时，而不是固定模板。
- 2-CTA 可分摊 SMEM 并扩大 tile，但 TMEM layout、DSMEM 交换和 cluster 同步更复杂。
- TMEM 是架构相关资源；容量、指令语义和工具链支持必须绑定 PTX/CUDA/CUTLASS 版本。

## 相关实体

- [[../entities/NVIDIA Blackwell]]
- [[../entities/阿里云 PAI 团队]]
- [[../entities/CAKE KDA]]
- [[../entities/NVIDIA Hopper]]

## 相关来源

- [[../sources/PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化]]
- [[../sources/REMINDER FF-KDA & CAKE KDA Highlights]]
- [[../sources/译 NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell]]

## 相关概念

- [[CUDA内存层次]]
- [[GPU执行模型]]
- [[FlashAttention]]
- [[Tiling]]
- [[Occupancy]]
- [[重计算]]
- [[KDA]]
- [[Persistent Kernel]]

## 研究备注

- 当前规格与访问语义来自 PAI-FA 文章，后续应以对应版本 PTX ISA、CUDA Programming Guide 和 CUTLASS TMEM 文档逐项核实。
- 需要进一步区分 TMEM 的物理容量、按 CTA/cluster 分配语义、2-CTA MMA 的共享方式和多个驻留 CTA 对资源的竞争关系。
