---
type: concept
topic: GPU 编程
sources: 1
updated: 2026-08-21
---

# Persistent Kernel

## 定义

`Persistent Kernel` 是让一组 CTA 或 thread block clusters 长时间驻留在 GPU 上，并在 kernel 内连续取得和处理多个逻辑 work tiles 的执行方式。它仍由 host 发起一次 kernel launch，但由已经驻留的 worker 迭代处理后续 tiles，而不是让每个 tile 都依赖一个新的 CTA 实例。它不是 Hopper 专属能力；Hopper 的 TMA、WGMMA、cluster 和异步 barrier 只是让 persistent pipeline 更易组织。

## 它解决什么问题

- 摊销多个短 work tiles 重复承担的 CTA 初始化、descriptor、pipeline warm-up 与调度成本。
- 让前一 tile 的 compute/epilogue 与后一 tile 的 input load 形成跨 tile overlap。
- 对不等长 tiles 使用静态步进、global atomic queue 或 [[Cluster Launch Control]] 等方式动态取活，减轻 [[Tail Effect]]。
- 允许 kernel 内保留长期状态、barrier、SMEM buffers 或 instruction schedule，避免每个 tile 重建。

## 核心机制

1. 静态/atomic-queue 实现常只 launch 接近一波可驻留 workers；CLC 实现也可以按完整逻辑 grid launch，再由首批活跃 clusters 取消并继承尚未启动的任务坐标。
2. 每个长期驻留 CTA/cluster 完成当前 tile 后，从静态映射、任务池或 CLC 响应取得下一个 tile。
3. TMA/load、MMA 和 epilogue warps 可保持职责分工，并跨 tile 复用多级 buffer、barrier 和 descriptor。
4. workers 在任务池耗尽或调度终止条件满足后退出。

## 架构演进

- [[../entities/NVIDIA Ampere]] 的 `cp.async` 主要强化单 CTA 内 load/compute overlap，为多 stage pipeline 提供基础。
- [[../entities/NVIDIA Hopper]] 的 TMA 与异步 WGMMA 更适合 warp-specialized persistent pipeline，让一个 CTA 连续处理多个 tiles。
- [[../entities/NVIDIA Blackwell]] 的 `tcgen05` 与 [[Tensor Memory|TMEM]] 把 accumulator 从普通 RF 中迁出，并可把 output consumption/write-back 更明确地纳入跨 tile pipeline。
- Blackwell 的 [[Cluster Launch Control]] 进一步提供硬件支持的 dynamic persistent tile scheduling：活跃 cluster 取消尚未启动的 cluster，并继承其 work-tile 坐标。

## Hopper 上的典型 Warp Specialization

```text
Load warp      : TMA 预取 tile N+1
MMA warpgroup  : WGMMA 计算 tile N
Epilogue warps : 处理并写回 tile N-1
Barrier        : 跟踪数据 ready 与 buffer 可复用状态
Resident CTA   : 下一轮继续处理新 tile，而不是退出
```

长期驻留本身不创造额外 Tensor Core 算力；收益来自复用 pipeline state、跨 tile overlap 和更灵活的任务取得。Hopper thread block cluster 还能让多个 CTA 协作一个更大 work item，但 cluster size 增大会降低可同时 active 的 blocks，必须用 cluster-aware occupancy 计算。

## 与普通 Kernel 和 Megakernel 的区别

| 方式 | 典型边界 | 主要目标 |
| --- | --- | --- |
| 普通 tile kernel | 一个 CTA 通常负责一个逻辑 tile | 简单、由硬件补充 waves |
| Persistent kernel | 一个 CTA/cluster 连续处理多个同类 tiles | 摊销初始化、跨 tile overlap、动态负载均衡 |
| [[Megakernel]] | 跨多个算子、层甚至整模型的指令序列 | 减少更大范围 kernel 边界和中间状态往返 |

Persistent 描述的是 worker 生命周期与 work acquisition；megakernel 描述的是融合范围。一个 megakernel 往往是 persistent 的，但 persistent GEMM/attention kernel 不一定跨算子，因此不应把两者等同。

## 调度方式

- Static persistent：worker 按固定 stride 遍历 tiles，开销低，但不规则 workload 容易出现 straggler。
- Global atomic queue：worker 从全局 counter 取下一个 tile，负载均衡更强，但有 atomic/global-memory 成本。
- CLC dynamic persistent：Blackwell cluster 取消并接手未启动 cluster 的坐标，避免集中 counter，但引入 CLC response、mbarrier 和缓存顺序权衡。

## 关键权衡

- 长期驻留 worker 可能占满 SM，影响并发 kernel 的资源获得和优先级调度。
- 为复用 pipeline 而保留的 SMEM/register/TMEM 会限制 occupancy；更深 stage 不一定更快。
- 静态 tile 顺序有利于可预测局部性，但成本不均时会产生 straggler；动态顺序更均衡，却可能降低 L2 hit rate。
- Persistent kernel 不是“没有 kernel launch”：它只把多次逻辑 tile 执行合并进一次 launch，并摊销后续边界。
- 当总 work tiles 很少、tile 很均匀或普通 grid 已能充分覆盖 SM 时，额外 scheduler/barrier 可能得不偿失。

## 相关实体

- [[../entities/NVIDIA Ampere]]
- [[../entities/NVIDIA Hopper]]
- [[../entities/NVIDIA Blackwell]]

## 相关来源

- [[../sources/译 NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell]]

## 外部一手资料

- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)：Hopper cluster、DSMEM、TMA 与 occupancy 约束；该文档没有把 persistent kernel 定义为 Hopper 专属 feature。
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)：Hopper 异步执行、cluster 与 transaction barrier 背景。

## 相关概念

- [[GPU执行模型]]
- [[CUDA内存层次]]
- [[Tiling]]
- [[Tail Effect]]
- [[Occupancy]]
- [[Cluster Launch Control]]
- [[Megakernel]]
- [[Tensor Memory]]

## 研究备注

- 当前翻译来源对 Hopper persistent pipeline 的图示较简化；具体 worker 数、任务取得方式、warp 分工和跨 tile overlap 需绑定实际 CUTLASS/FA kernel。
- 官方 Hopper 文档说明了构成 persistent pipeline 的硬件原语，但并未要求所有 Hopper kernels 使用 persistent scheduler；不能从 GPU 架构反推某个实现必然 persistent。
- 需要进一步区分 persistent CTA、persistent thread、persistent cooperative kernel 与整个模型 megakernel 的术语边界。
