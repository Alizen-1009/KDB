---
title: "[译] NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell"
source: "https://zhuanlan.zhihu.com/p/1987901646806729625"
author:
  - "[[cervol]]"
published:
created: 2026-08-21
description: "原文链接: NVIDIA’s GPUs - A look over the past 3 generations 过去五年，NVIDIA 发布了一系列新的 GPU 架构。从 2020 年推出的 Ampere 到前沿的 Blackwell，每一代架构都在计算性能、内存带宽和 AI 加速能力方…"
tags:
  - "clippings"
---
SuSun 等 81 人赞同了该文章

原文链接:

过去五年，NVIDIA 发布了一系列新的 GPU 架构。从 2020 年推出的 Ampere 到前沿的 Blackwell，每一代架构都在计算性能、内存带宽和 AI 加速能力方面实现了显著提升。本文将深入探讨 Ampere、Hopper 和 Blackwell 架构的关键特性和创新之处。

### NVIDIA's Ampere architecture

Ampere 于 2020 年 5 月发布，大幅提升了 Tensor Core 的性能（尽管 Tensor Core 早在几代之前就已经推出）。从规格上看，Ampere 具有：

- 80GB 高带宽显存（ [HBM](https://zhida.zhihu.com/search?content_id=268216047&content_type=Article&match_order=1&q=HBM&zhida_source=entity), or global memory），带宽可达 2.0 TB/s
- 108 个 SM
- 40MB 所有 SM 共享的 [L2 Cache](https://zhida.zhihu.com/search?content_id=268216047&content_type=Article&match_order=1&q=L2+Cache&zhida_source=entity)
- 每个 SM 内，具有：
	- 4 个 Tensor Core
		- 192KB 的 L1 cache，可申请作为共享内存（SMEM, or shared memory）使用
		- 65536 个寄存器
- 支持异步拷贝机制，使得可以在软件层面建立 HBM 到SMEM 的流水线。
![Ampere architecture.png](https://picx.zhimg.com/v2-59c13881c3a65291dc92e4ba613211f1_1440w.jpg)

Ampere architecture.png

### NVDIA's Hopper architecture

Hopper 于 2022 年第三季度发售，在 Ampere 的基础上进一步改进，直到今天依然广泛用于 LLM 等计算任务。Hopper 架构规格如下：

- 80GB 高带宽显存（HBM），带宽可达 3.35 TB/s
- 132 个 SM
- 50MB 所有 SM 共享的 L2 Cache
- 每个 SM 内，具有：
	- 4 个 Tensor Core
		- 256KB 的 L1 cache/共享内存
		- 65536 个寄存器
- **引入 [TMA](https://zhida.zhihu.com/search?content_id=268216047&content_type=Article&match_order=1&q=TMA&zhida_source=entity) (Tensor Memory Accelerator) 以加速块内存(block memory)拷贝**
- 不向前兼容（使用到 Hopper 架构特性的代码，不保证能在下一代架构 Blackwell 上运行）
![Hopper architecture.png](https://pic2.zhimg.com/v2-c421a201f57dfd500535ceaab31ffcdf_1440w.jpg)

Hopper architecture.png

### NVIDIA's Blackwell architecture

Blackwell 是 NVIDIA 最新的架构，也是目前 LLM 部署的最佳架构。它在 Hopper 和 Ampere 的基础上进行了改进，提高了计算能力，并提供了一些关键特性以实现更快的执行速度：

- 7.672 TB/s 高带宽显存
- 148 个 SM
- 192MB 所有 SM 共享的 L2 Cache
- 每个 SM 内，具有：
	- 4 个 Tensor Core
		- 228KB 的 L1 cache/共享内存
		- 65536 个寄存器
- 第五代 Tensor Core 架构
- **256KB Tensor Memory**
![Blackwell architecture.png](https://pic1.zhimg.com/v2-238fb87e4dad4276dead292330469b5e_1440w.jpg)

Blackwell architecture.png

### GPU comparison at a glance

为了让读者了解差异的大小，以下表格比较了点对点性能：

| Metric | A100 (Baseline) | H100 | H200 | B100 | B200 |
| --- | --- | --- | --- | --- | --- |
| Peak Memory Bandwidth | 1.0x | 1.6x | 2.4x | 3.9x | 3.9x |
| NVLink Bandwidth | 1.0x | 1.5x | 1.5x | 3.0x | 3.0x |
| Peak BF16 TFLOPS (dense) | 1.0x | 3.2x | 3.2x | 5.6x | 7.2x |
| Peak FP8 TFLOPS (dense) | N/A | 1.0x | 1.0x | 1.8x | 2.3x |

随着 GPU 在浮点运算速度（FLOPs/s）和带宽方面的性能不断提升，实现这种峰值性能的编程模型也在不断演进。  
下一节，我们回顾不同架构上的最佳操作调度方案。

### Pre-Ampere optimization

Ampere 引入了异步数据传输，而在 Ampere 之前，内存操作指令和计算指令在同一个执行流水线上，内存操作会阻塞计算，在单个 CTA 的视角下，流水线类似于：从全局加载数据 → 等待 → 计算 → 等待 → 存储结果，几乎是串行的。

![Pre-Ampere optimization.png](https://pic4.zhimg.com/v2-ff767d50f42cb9a29a95dc083a3604a7_1440w.jpg)

Pre-Ampere optimization.png

因此，在 Ampere 架构之前，常见的优化思路是人为地实现“并行 CTA”来隐藏延迟——既然单个 CTA 内部无法 overlap 内存拷贝与计算，那只能靠多个 CTA 同时驻留在一个 SM 上，以 SM 在不同 CTA 间的调度来实现 overlap，即所谓的“ [double buffering](https://zhida.zhihu.com/search?content_id=268216047&content_type=Article&match_order=1&q=double+buffering&zhida_source=entity) ”。  
使用多个 CTA 来分别实现数据传输和计算，导致 SM 的有限内存资源出现争执，并迫使在内存延迟和最优的 tiling 之间（由于一个 SM 中驻留多个 CTA，在有限的 SM 资源中 CTA 的 tile size 不能过大）做出权衡。

### Ampere optimization

借助 Ampere 的 [异步拷贝指令](https://zhida.zhihu.com/search?content_id=268216047&content_type=Article&match_order=1&q=%E5%BC%82%E6%AD%A5%E6%8B%B7%E8%B4%9D%E6%8C%87%E4%BB%A4&zhida_source=entity) (`cp.async`)，可以对数据加载和 MMA(Matrix Multiply-and-Accumulate) 操作进行流水线处理，并在单个 CTA 中实现 overlap。

![Ampere optimization.png](https://pica.zhimg.com/v2-bfd92dfe16f1030f90c30df7825bee1c_1440w.jpg)

Ampere optimization.png

  
使用 `cp.async` 指令，线程可以发出内存拷贝指令并立即继续执行下一个任务，将内存传输延迟隐藏在 mma 操作之后。

> ✅ Ampere’s win: Overlapping input loads with computation  
> ❌ Ampere’s problem: CTA launch overhead

### Hopper optimization

Hopper架构引入了新的数据传输和 MMA 指令：

1. TMA (Tensor Memory Accelerator)：TMA 是一个专用于移动 tensor 的硬件单元。与其让每个线程分别计算地址并加载一个单元的数据，TMA 以异步方式在 global memory 和 shared memory 之间传输一整个 tensor tile。
2. Asynchronous Warpgroup MMA (WGMMA)：虽然 Ampere 已经允许了 MMA（同步指令）与数据传输重叠，但 WGMMA 指令是异步的，它不仅允许 MMA 与内存访问重叠，还允许 MMA 与 Tensor Core 的计算重叠。

在其影响下出现了一种新的 matmul 开发范式，称为持久内核（Persistent Kernels）。我们将在后续文章中深入讨论这项技术，但从总体上看，它允许 CTA 驻留在 SM 上并处理多个 tile 而无需返回 host (CPU)。Persistent Kernel 消除了 kernel launch 的开销，实现了一个 work tile 的输出与下一个 work tile 的加载之间的 overlap。

![Hopper optimization.png](https://picx.zhimg.com/v2-70921b5f2b6d41ddda991d6f78beb4b7_1440w.jpg)

Hopper optimization.png

> ✅ Hopper’s win: 降低 CTA launch 的 overhead，实现跨 tile 数据传输的 overlap  
> ❌ Hopper’s problem: WGMMA consumers 使用了大量的寄存器，以及 tensor core 与 ALU 之间的资源竞争

### Blackwell optimization

Blackwell 架构引入了 `tcgen05` 指令，MMA 的运算结果存储在新的专用硬件 [tensor memory](https://zhida.zhihu.com/search?content_id=268216047&content_type=Article&match_order=1&q=tensor+memory&zhida_source=entity) 中。这打破了 WGMMA 对寄存器的依赖。现在，我们可以在不污染寄存器的情况下利用更多的流水线技术。

![Blackwell optimization.png](https://pica.zhimg.com/v2-8406c0e491ea48d6f7962300d00dc7a4_1440w.jpg)

Blackwell optimization.png

Blackwell 的三阶段流水线：

1. **Loading inputs** (TMA) - 用 TMA 加载输入数据
2. **Computing MMA** (Tensor Cores) - Tensor Cores 计算 MMA 运算，并将结果写入 tensor memory
3. **Storing outputs** (from Tensor Memory) - 将结果从 tensor memory 拷贝到 global memory

这三个阶段可以在不同的内存区域同时运行——第 N 个 tile 进行计算，第 N+1 个 tile 加载数据，第 N-1 个 tile 写回数据，从而形成软件流水线。

> ✅ Blackwell’s win: pipelining the write-out stage  
> ❌ Blackwell’s problem: Tensor memory only supports very limited instructions
> 
> 本文由 [Zhihu on Obsidian](https://zhuanlan.zhihu.com/p/1901622331102696374) 创作并发布

编辑于 2025-12-26 15:20・北京

赞同 81