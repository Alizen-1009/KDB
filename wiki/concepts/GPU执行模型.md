---
type: concept
topic: GPU 编程
sources: 7
updated: 2026-06-12
---

# GPU执行模型

## 定义

描述 GPU 如何以 `thread / warp / block / SM` 这一层级结构组织并执行并行计算的模型。

## 它解决什么问题

- 帮助理解为什么同一段 CUDA / kernel 代码在 GPU 上会表现出和 CPU 完全不同的性能特征
- 为分析控制分歧、共享内存复用和并发调度提供统一抽象

## 核心机制

- 单个线程执行最细粒度的计算
- `warp` 是 32 个连续线程共同执行的基本调度单位
- `block` 是线程组，运行在单个 `SM` 上并共享 shared memory
- 多个 `SM` 彼此独立调度不同 block，从而形成大规模吞吐并行
- 很多性能现象都能还原到这套层级上：`warp` 决定 [[Warp Divergence]] 是否发生，`block` 决定 shared memory 复用与 [[Bank Conflict]] 布局，`SM` 资源上限决定 [[Occupancy]] 与 [[Tail Effect]]
- SM 结构图里的 warp scheduler、dispatch unit、register file、执行单元、L1/shared memory 等属于硬件实现细节；具体数量和容量随 GPU 架构变化，不能直接当作所有 CUDA GPU 的通用图
- CUDA kernel 之间通常存在严格顺序边界：后一个 kernel 的 block 不会在前一个 kernel 的所有 block 完成前开始执行。这个语义简化了数据依赖管理，但在许多短 kernel 串联时会放大 launch overhead、tail effect 和 load bubble。
- [[Megakernel]] 通过在单个 kernel 内部自行调度 SM instruction 和同步依赖，绕开部分 kernel 间全局边界，但也把依赖正确性责任交给实现者。

## Rubin 架构与跨 kernel 调度

[[../entities/NVIDIA Rubin]] 来源转述 Rubin 将部分 I/O 功能移到独立 I/O Die，并把双计算 Die 的 SM 总数提高到 224。对执行模型更重要的不是 SM 数本身，而是 producer-consumer kernel 能否按更细粒度数据 ready 关系推进，从而减少前序 kernel 尾部与后序 kernel 启动之间的 bubble。

来源关于具体轮询、barrier 和 scheduler 驻留方式仍是作者推测；正式理解应区分官方产品描述、PTX 可见语义和硬件调度器内部实现。

## 关键权衡

- SIMT 模型让大量同构计算更容易扩展
- 一旦出现严重分支分歧或不规则访存，执行效率会迅速下降

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/NVIDIA Rubin]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 5 - GPUs]]
- [[../sources/你一定要知道：CUDA优化六要]]
- [[../sources/CUDA优化维度框架]]
- [[../sources/CUDA内存层次与动态共享内存问答整理]]
- [[../sources/多卡GPU监控与SM执行模型面试整理]]
- [[../sources/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B]]
- [[../sources/Nvidia Rubin架构分析预览]]

## 相关概念

- [[内存合并访问]]
- [[CUDA内存层次]]
- [[动态共享内存]]
- [[Bank Conflict]]
- [[Occupancy]]
- [[Warp Divergence]]
- [[Tail Effect]]
- [[Tiling]]
- [[FlashAttention]]
- [[Megakernel]]
- [[Programmatic Dependent Launch]]

## 研究备注

- 后续可补具体到 NVIDIA GPU 的 SM 资源组成、occupancy 和 warp scheduler 细节；若记录具体数值，需要绑定到 A100/H100 等明确硬件实体
- 面向 LLM 推理时，`thread / warp / block / SM` 可直接映射到 kernel 质量判断：prefill 中的大 GEMM 和 FlashAttention 需要足够 block/warp 去填满 SM 并命中 Tensor Core；decode 阶段常因 batch 小、KV cache 读写多而更容易出现 SM Active 不高或 DRAM 带宽先打满。不要只看“GPU 利用率”，应结合 `SM Active / SM Issue / Tensor Active / Occupancy / DRAM Bandwidth` 判断是算力没喂饱、访存受限、shape 太碎，还是通信或调度断流。
- Megakernel 来源适合补一类 Nsight Systems 时间线案例：很多短 kernel 之间的空隙、尾部低 SM Active 和权重加载不连续，可能比单个 kernel 的 micro-optimization 更影响低延迟 decode。
