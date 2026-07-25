---
type: concept
topic: GPU 编程
sources: 2
updated: 2026-06-12
---

# Programmatic Dependent Launch

## 定义

`Programmatic Dependent Launch`（PDL）是 NVIDIA CUDA 中用于让后续 kernel 在前序 kernel 尚未完全结束时提前启动准备的一种依赖启动机制。

## 它解决什么问题

- 缓解严格串行 kernel launch 中，下一个 kernel 必须等前一个 kernel 完全结束才开始的空隙。
- 让部分准备工作可以与前序 kernel 的收尾阶段重叠，减少 GPU 时间线上的无效等待。

## 核心机制

- 后续 kernel 可在前序 kernel 运行期间被依赖式启动。
- 前后 kernel 之间通过同步点保证数据依赖不被破坏。
- 在 `Look Ma, No Bubbles!` 来源中，作者认为 PDL 的 `cudaGridDependencySynchronize` 粒度仍偏粗：例如 attention 需要等待所有 Q/K/V 完成，而不能按 head 或 chunk 级别尽早消费已准备好的输入。

## Rubin 的 Thread Block 级依赖方向

[[../entities/NVIDIA Rubin]] 来源描述了比 Blackwell PDL 更数据驱动的 producer-consumer 启动方向：consumer kernel 可以更早进入等待状态，在 producer 的 tile/thread block 数据 ready 后继续工作，而不是等待更大范围依赖解除。这有潜力减少短 kernel 串联的 bubble，并缩小与 [[Megakernel]] 内部细粒度调度的差距。

但来源作者在 PTX 9.4 预览中未找到完整 Grid Scheduling 或 Tile Publication 指令；关于 consumer block 预占少量资源、轮询共享 flag/barrier、使用 acquire/release 的实现属于推测，不能作为已确认的 Rubin 微架构事实。

## 关键权衡

- 相比完全串行 launch，PDL 可以减少一部分 kernel 边界等待。
- 它仍保留 kernel 之间的边界和较粗粒度同步，难以表达 megakernel 内部那种 instruction / chunk 级依赖。
- 是否有效依赖具体 workload 的依赖图、kernel 时长、输入 ready 粒度和硬件/驱动支持。

## 相关实体

- [[../entities/Megakernels]]
- [[../entities/NVIDIA Rubin]]

## 相关来源

- [[../sources/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B]]
- [[../sources/Nvidia Rubin架构分析预览]]

## 相关概念

- [[CUDA Kernel]]
- [[Megakernel]]
- [[算子融合]]
- [[Tail Effect]]

## 研究备注

- 后续可补 NVIDIA 官方文档中的 PDL 支持条件、API 约束和与 CUDA Graphs / streams 的关系。
