---
type: concept
topic: GPU 编程
sources: 10
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

## Thread block cluster 与 CLC

[[../entities/NVIDIA Blackwell]] 的 [[Cluster Launch Control]] 把调度粒度扩展到 thread block cluster：一个活跃 cluster 可以原子取消尚未启动的 cluster，并取得后者首个 CTA 的 grid 坐标来接手其 work tile。该机制不是让 CTA 在任意时刻迁移 SM，而是围绕“尚未启动的 cluster launch”做取消与任务继承；同一 cluster 内仍需由 leader scheduler warp、shared-memory 响应、mbarrier 和消费者 warp 协同保证坐标一致。

## Blackwell 异构异步流水

PAI-FA 来源展示了比统一 SIMT 叙事更细的职责分工：Load warps 驱动 TMA/SMEM staging，MMA 侧由单线程发起 `tcgen05.mma`，Softmax/Correction warps 使用 CUDA Core 处理从 [[Tensor Memory|TMEM]] 读出的中间量，Epilogue warps 完成写回。这里的 warp specialization 不是增加数学并行度，而是让搬运、Tensor Core、CUDA Core 和输出阶段拥有更规则的指令流并通过显式 producer-consumer barrier 重叠。

2-CTA/CTA Pair 进一步把执行范围扩展到两个相邻 CTA：leader CTA 发起 MMA，两个 CTA 通过 DSMEM 共享或交换部分操作数，并分别承担分块。它能扩大 tile、分摊 SMEM，但要求 cluster 内 layout 与同步严格匹配。

### CAKE KDA 的双依赖域流水

来源称 [[../entities/CAKE KDA]] 在单 CTA 内使用 32 warps，并将其组织为五组 preparation producers 与有序 recurrence consumer；该线程配置仍需按 PR 生成 kernel 核实。五组 producer 可同时构造未来 chunks 的 gate/Q/K/`Mqk`/inverse，并写入五级 SMEM ring；consumer 必须按 chunk 顺序更新唯一 FP32 TMEM state。mbarrier 同时承担 ready notification 和 backpressure，使 slot 只有在前一 chunk 被消费后才能复用。该设计增加 CTA 内 look-ahead，但没有增加 grid 维度，因此小 batch×heads 仍可能无法覆盖全部 SM。

## Rubin 架构与跨 kernel 调度

[[../entities/NVIDIA Rubin]] 来源转述 Rubin 将部分 I/O 功能移到独立 I/O Die，并把双计算 Die 的 SM 总数提高到 224。对执行模型更重要的不是 SM 数本身，而是 producer-consumer kernel 能否按更细粒度数据 ready 关系推进，从而减少前序 kernel 尾部与后序 kernel 启动之间的 bubble。

来源关于具体轮询、barrier 和 scheduler 驻留方式仍是作者推测；正式理解应区分官方产品描述、PTX 可见语义和硬件调度器内部实现。

## 关键权衡

- SIMT 模型让大量同构计算更容易扩展
- 一旦出现严重分支分歧或不规则访存，执行效率会迅速下降

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/NVIDIA Rubin]]
- [[../entities/NVIDIA Blackwell]]
- [[../entities/CAKE KDA]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 5 - GPUs]]
- [[../sources/你一定要知道：CUDA优化六要]]
- [[../sources/CUDA优化维度框架]]
- [[../sources/CUDA内存层次与动态共享内存问答整理]]
- [[../sources/多卡GPU监控与SM执行模型面试整理]]
- [[../sources/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B]]
- [[../sources/Nvidia Rubin架构分析预览]]
- [[../sources/Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs]]
- [[../sources/PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化]]
- [[../sources/REMINDER FF-KDA & CAKE KDA Highlights]]

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
- [[Cluster Launch Control]]
- [[Tensor Memory]]
- [[KDA]]

## 研究备注

- 后续可补具体到 NVIDIA GPU 的 SM 资源组成、occupancy 和 warp scheduler 细节；若记录具体数值，需要绑定到 A100/H100 等明确硬件实体
- 面向 LLM 推理时，`thread / warp / block / SM` 可直接映射到 kernel 质量判断：prefill 中的大 GEMM 和 FlashAttention 需要足够 block/warp 去填满 SM 并命中 Tensor Core；decode 阶段常因 batch 小、KV cache 读写多而更容易出现 SM Active 不高或 DRAM 带宽先打满。不要只看“GPU 利用率”，应结合 `SM Active / SM Issue / Tensor Active / Occupancy / DRAM Bandwidth` 判断是算力没喂饱、访存受限、shape 太碎，还是通信或调度断流。
- Megakernel 来源适合补一类 Nsight Systems 时间线案例：很多短 kernel 之间的空隙、尾部低 SM Active 和权重加载不连续，可能比单个 kernel 的 micro-optimization 更影响低延迟 decode。
