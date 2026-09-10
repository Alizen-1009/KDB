# CUDA 内存层次、GPGPU 与 Cache Line 入门

## 基本关系（已知事实）

GPU 是图形处理器这一类硬件；GPGPU（General-Purpose computing on Graphics Processing Units）是用 GPU 做通用计算的方法，例如矩阵乘法、训练与推理、科学模拟。CUDA 是 NVIDIA 提供的并行计算平台和编程模型，是实现 GPGPU 的一种途径。GPGPU 不表示比 GPU 多一个部件，也不保证任意串行任务都更快。

## 内存层次

| 存储 | 位置与作用域 | 使用方式 |
|---|---|---|
| Register | SM 内，线程私有 | 编译器分配，保存累加器和临时值 |
| Shared memory | SM 内，通常 block 内共享 | 程序显式读写，用于分块复用、协作；正确同步 |
| L1 cache | SM 侧片上缓存 | 硬件管理，缓存部分 global/local 访问 |
| L2 cache | GPU 片上，多 SM 共享 | 硬件管理，减少显存访问 |
| Global memory | 通常由 HBM/GDDR 显存承载 | 存储大张量，设备线程可访问 |
| Local memory | 线程私有地址空间，通常由显存承载并可缓存 | 常用于寄存器溢出或某些局部数组 |

Constant memory 提供只读常量空间及缓存路径，适合 warp 同址读取；texture/read-only 是专门的读取路径。Host memory 是 CPU 侧内存。

普通 global load 的简化路径是：线程请求 → L1（若使用且命中则返回）→ L2（命中则返回）→ 显存。Shared memory 不在这条 cache miss 链上。变量可见性与物理位置必须分开理解，特别是 local 不等于快速片上存储。

关联：[[../../wiki/concepts/CUDA内存层次]]、[[../../wiki/concepts/GPU执行模型]]、[[../../wiki/concepts/CUDA Kernel]]。

## Cache line 与合并访问

Cache line（缓存行）是缓存用来组织和标识一块连续、对齐地址数据的单位，不是数组的逻辑行。Nsight Compute 文档描述的 L1/L2 cache line 是 128B，分为 4 个 32B sector；访问一个值并不意味着总从显存取回整条 128B。

示意：一条 128B line = [0–31][32–63][64–95][96–127]，每格是一个 sector。

以下为地址覆盖推导，不是 benchmark：假设 float 为 4B、A 起始地址按 128B 对齐、一个 warp 的 32 个线程全部活跃且执行同一条普通读取指令，lane 为 0–31。

| 访问 | 有效数据 | 覆盖 32B sectors |
|---|---:|---:|
| A[lane] | 128B | 4 |
| A[lane + 1] | 128B | 5 |
| A[lane * 32] | 128B | 32 |

最后一行线程地址间隔为 128B，每个线程落在不同 sector，只使用其中 4B；第一行则充分使用 4 个 sectors。这解释了连续、对齐访问通常更高效，但不能由 4 对 32 直接推断运行时间相差 8 倍。

合并访问减少同一 warp 指令的 sector 请求；cache hit 减少向更低层继续取数；shared memory 则由程序组织数据复用。三者解决不同问题。

## 实现差异与性能权衡

- 32B 合并访问规则按 CUDA Best Practices 的 compute capability 6.0+ 范围解释；不推广到所有历史架构或特殊访存指令。
- L1/shared memory 容量和物理资源划分、缓存策略随架构变化。
- 寄存器和 shared memory 使用过多会限制并发；shared memory 布局不当可能产生 bank conflict。
- Sector 请求量不等于 DRAM 流量；缓存命中、复用、写策略等都会影响后续传输。

## 官方来源与边界

本报告为知识库与官方文档的教学归纳，示例是理论地址计算，没有实测吞吐或延迟。未指定 GPU 型号，因此不提供具体容量、周期数或速度倍数。

- [CUDA Introduction](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html)
- [CUDA Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [CUDA Best Practices：Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [[../../wiki/concepts/内存合并访问]]
- [[../../wiki/sources/CUDA内存层次与动态共享内存问答整理]]
