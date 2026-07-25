---
type: concept
topic: GPU 编程
sources: 6
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
- [[Megakernel]] 来源提供了一个 shared memory 资源管理案例：把 H100 每个 SM 的部分 shared memory 切成固定页，instruction 显式申请/释放 page，用于在前一段计算收尾时尽早加载下一段权重。

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

## 相关来源

- [[../sources/CUDA内存层次与动态共享内存问答整理]]
- [[../sources/斯坦福CS336 Lecture 5 - GPUs]]
- [[../sources/你一定要知道：CUDA优化六要]]
- [[../sources/CUDA优化维度框架]]
- [[../sources/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B]]
- [[../sources/Nvidia Rubin架构分析预览]]

## 相关概念

- [[CUDA Kernel]]
- [[GPU执行模型]]
- [[动态共享内存]]
- [[Occupancy]]
- [[Bank Conflict]]
- [[内存合并访问]]
- [[Tiling]]
- [[Megakernel]]

## 研究备注

- 后续可补不同 NVIDIA 架构中 L1/shared memory 配置、register file 大小、L2 容量和 cache policy 的差异
- Megakernel 来源中的 shared memory paging 是特定实现策略，不等同于 CUDA 提供的通用 shared memory 分页机制；引用时应说明这是作者在 kernel 内部的资源管理抽象。
