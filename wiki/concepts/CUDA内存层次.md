---
type: concept
topic: GPU 编程
sources: 8
updated: 2026-06-12
---

# CUDA内存层次

## 定义

`CUDA内存层次` 指 CUDA 程序在 GPU 上可使用的多级存储体系，包括 register、local memory、shared memory、L1/L2 cache、global memory、constant memory、texture/read-only path 和 host memory 等。

## 它解决什么问题

- 帮助判断数据应放在哪里，以及为什么同样的计算逻辑会因为访存路径不同产生巨大性能差异
- 为分析 [[CUDA Kernel]] 的 latency hiding、带宽利用率、数据复用和资源占用提供基础框架

## 核心机制

- `Register` 是 thread 私有最快存储，通常由编译器为局部标量变量分配
- `Local Memory` 是 thread 私有地址空间，但物理上通常走 global memory 路径，常来自寄存器 spill、大局部数组或动态索引局部数组
- `Shared Memory` 是 block 内线程共享的片上内存，适合 tiling、block reduce、数据重排和显式复用
- `L1 Cache` 位于 SM 附近，主要由硬件缓存 global/local memory 访问；可粗略理解为硬件管理的片上缓存
- `L2 Cache` 通常由整个 GPU 共享，服务 global memory 访问、跨 SM 复用和部分原子操作路径
- `Global Memory` 是 GPU 显存中的大容量内存，所有 block 可访问，但延迟高，需要依赖 [[内存合并访问]] 和足够并发隐藏延迟
- `Constant Memory` 适合 warp 内读取相同常量地址；地址分散时收益下降
- `Texture / Read-only Cache` 适合某些只读、空间局部性较强的访问模式
- `Host Memory` 是 CPU 侧内存；pinned host memory 更适合异步 H2D / D2H 拷贝
- [[Tensor Memory]] 是 Blackwell Tensor Core 数据路径中的专用片上存储，用于承载 `tcgen05` MMA 操作数或 accumulator；它需要显式分配和协作访问，不能按普通 thread-private register 或通用 shared memory 理解
- [[Megakernel]] 来源提供了一个 shared memory 资源管理案例：把 H100 每个 SM 的部分 shared memory 切成固定页，instruction 显式申请/释放 page，用于在前一段计算收尾时尽早加载下一段权重。

## Blackwell TMEM 数据路径

PAI-FA 来源把 FA4 Forward 的主要数据流概括为 `GMEM → SMEM → TMEM → RF/SMEM → GMEM`：TMA/Load warps 负责输入 staging，Tensor Core 在 TMEM 中产生 `S/O` 等 accumulator，CUDA Core 再经 RF 做 Softmax 和 correction。`head_dim=256` 会放大沿 head dimension 展开的 accumulator，因此 tile 和 pipeline stage 必须同时服从 TMEM、SMEM 与 RF 容量，而不能只追求更深双缓冲。

## KDA 的 lifetime-based SMEM aliasing

CAKE KDA M128 来源给出一个极端片上复用案例：五个 preparation stages、两个 output buffers 与 barrier/control 合计约 `227,328 bytes（222 KiB）` SMEM。它没有为每个逻辑 tensor 分配独立 buffer，而是按“首次写入到最后一次读取”的生命周期复用物理区域，例如 raw gate 复用为 centered/decayed Q、raw K 复用为 transformed K、prefix/inverse workspace 后续复用为 `Mqk`/restore factor/V。该方法节省容量，但任一 barrier 或生命周期判断错误都会造成数据覆盖。

来源中的 FF-KDA 图片则展示 global workspace 仍存在时的另一种优化：保留 swizzled physical byte image，以 raw `cp.async.bulk` 搬运，避免 GMEM 两侧的 unswizzle/reswizzle。这减少 layout conversion 和碎片化传输，但没有消除 HBM 往返。

## Rubin 的动态 L2 生命周期提示

[[../entities/NVIDIA Rubin]] 来源介绍的 `applypriority` 允许在数据使用阶段结束后，动态调整既有 cache line/tensor footprint 的 eviction priority。MoE 中可先让当前 expert 权重倾向 `evict_last`，供多个 token/tile 重复读取；expert last-use 后再恢复 `evict_normal`，避免旧热点挤压下一个 expert 的 L2 容量。

它也可用于多阶段 GEMM、fused kernel、sliding-window attention 或 KV 生命周期，但 cache hint 只是替换倾向，不是锁定驻留保证；收益取决于 working set、L2 容量和真实复用距离。

## 关键权衡

- 越靠近执行单元通常越快，但容量越小、作用域越窄
- shared memory 可以显式提升数据复用，但使用过多会限制 [[Occupancy]]，布局不当还会触发 [[Bank Conflict]]
- 寄存器能减少访存，但过高寄存器压力会降低 occupancy，甚至 spill 到 local memory
- global memory 容量大，但需要 coalesced access 和足够 active warps 才能充分利用带宽

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/NVIDIA Rubin]]
- [[../entities/NVIDIA Blackwell]]
- [[../entities/CAKE KDA]]

## 相关来源

- [[../sources/CUDA内存层次与动态共享内存问答整理]]
- [[../sources/斯坦福CS336 Lecture 5 - GPUs]]
- [[../sources/你一定要知道：CUDA优化六要]]
- [[../sources/CUDA优化维度框架]]
- [[../sources/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B]]
- [[../sources/Nvidia Rubin架构分析预览]]
- [[../sources/PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化]]
- [[../sources/REMINDER FF-KDA & CAKE KDA Highlights]]

## 相关概念

- [[CUDA Kernel]]
- [[GPU执行模型]]
- [[动态共享内存]]
- [[Occupancy]]
- [[Bank Conflict]]
- [[内存合并访问]]
- [[Tiling]]
- [[Megakernel]]
- [[Tensor Memory]]
- [[KDA]]

## 研究备注

- 后续可补不同 NVIDIA 架构中 L1/shared memory 配置、register file 大小、L2 容量和 cache policy 的差异
- Megakernel 来源中的 shared memory paging 是特定实现策略，不等同于 CUDA 提供的通用 shared memory 分页机制；引用时应说明这是作者在 kernel 内部的资源管理抽象。
