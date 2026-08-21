---
type: entity
entity_type: 硬件
topic: GPU 编程
sources: 4
updated: 2026-08-21
---

# NVIDIA Blackwell

## 一句话说明

`NVIDIA Blackwell` 是 NVIDIA 在 Hopper 之后的 GPU 架构；本页聚焦 B200/SM100 路线的 `tcgen05`、[[Tensor Memory|TMEM]]、2-CTA 与 [[Cluster Launch Control]] 等 kernel 编程能力，而不是完整产品规格。

## 类型

- GPU 架构 / 硬件平台

## 核心信息

- Blackwell 开始提供硬件支持的 Cluster Launch Control，使活跃 cluster 可以取消尚未启动的 cluster，并取得其 CTA 坐标来接手对应 work tile。
- CLC 通过 `clusterlaunchcontrol.try_cancel`、`clusterlaunchcontrol.query_cancel`、shared-memory 响应与 transaction barrier 暴露给 PTX/CUDA kernel。
- 来源使用 B200 做实验，并称该设备有 148 个 SM，可组成 74 个 `(2, 1)` cluster；该数字是来源实验环境口径，不应直接推广到所有 Blackwell SKU。
- NVIDIA CUTLASS / [[CuTe DSL]] 提供 CLC dynamic persistent scheduler 示例，可用于 GEMM，并可作为 grouped GEMM、变长 attention 等不规则 workload 的调度基础。
- [[Tensor Memory]] 为 `tcgen05` MMA 提供专用片上 operand/accumulator 存储；PAI-FA 来源据此重构 `head_dim=256` FlashAttention-4 的 Forward/Backward pipeline。
- 与 [[NVIDIA Hopper]] WGMMA 主要把 accumulator 长期放在普通 register 不同，Blackwell `tcgen05` 可将 accumulator 放入 TMEM，减少 MMA consumers 的 RF 压力，并支持 load/compute/output consumption 更明确地分工。
- 翻译来源把 B200 流水概括为 tile N+1 load、tile N MMA、tile N-1 store；实际 write-back 通常仍需 TMEM→RF/SMEM→GMEM epilogue，不能把 TMEM 当作直接任意写 global memory 的通用缓存。
- 2-CTA/CTA Pair 可在 thread block cluster 内通过 DSMEM 共享操作数或交换中间分块，用更复杂的 layout 与同步换取更大 tile 和更低单 CTA SMEM 压力。
- Blackwell kernel 更常显式组合 TMA、TMEM、Tensor Core、CUDA Core、warp specialization 和异步 producer-consumer pipeline；这些能力不意味着任意 workload 自动加速，仍需针对 shape 与资源占用调优。
- [[CAKE KDA]] 在 B200/SM100a 上让 FP32 recurrent state 使用 256 个 TMEM columns 跨 chunks 常驻，并以五级 SMEM ring 连接并行 preparation 与有序 recurrence。
- CAKE M128 将共享左操作数的 state/output 更新拼成逻辑 `M128×N160×K32` GEMM，再用两个 `K=16` 的 `tcgen05.mma` 累加；小 `32×32` 矩阵仍走 warp-level MMA，体现不同 Tensor Core 路径按 shape 分工。
- 来源中的 FF-KDA 图示使用 raw `cp.async.bulk` S2G/G2S 搬运 swizzled byte image，说明 Blackwell 数据移动优化既可能通过片上融合消除 workspace，也可能通过保留物理 layout 降低 workspace transport 开销。

## 相关概念

- [[Cluster Launch Control]]
- [[GPU执行模型]]
- [[Tiling]]
- [[Tail Effect]]
- [[CuTe DSL]]
- [[Occupancy]]
- [[Tensor Memory]]
- [[FlashAttention]]
- [[KDA]]
- [[Persistent Kernel]]

## 相关实体

- [[NVIDIA Ampere]]
- [[NVIDIA Hopper]]

## 相关来源

- [[../sources/Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs]]
- [[../sources/PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化]]
- [[../sources/REMINDER FF-KDA & CAKE KDA Highlights]]
- [[../sources/译 NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell]]

## 冲突与备注

- 当前页面只依据一篇 Colfax Research 技术文章整理 CLC，且文章日期、代码 commit 与工具链版本必须绑定引用。
- CLC 改善不规则 workload 的负载均衡，但均衡 workload 上不保证优于静态 persistent scheduler；缓存局部性、pipeline 深度与 tile 顺序仍需实测。
- B200 的 SM 数、cluster shape 与 benchmark 配置属于具体实验环境，不代表整个 Blackwell 产品家族。
- PAI-FA 来源中的 L20A/L20C 名称、FA3 baseline 和多组峰值吞吐口径不一致；在核对硬件 SKU、代码 commit 与测量配置前，只能作为来源声称。
- CAKE KDA 的 222 KiB SMEM、256 TMEM columns、32 warps/CTA 与 `2.0512×` 均绑定 FlashInfer PR #4262 的特定生成 kernel/workloads，不代表所有 B200 KDA shape。
- 翻译来源列出的 148 SM、192MB L2、7.672TB/s、228KB L1/SMEM 和 256KB TMEM 接近特定 B200 口径，不代表 B100、GB200 或所有 Blackwell SKU；具体容量还需区分芯片总量、每 SM 与每 CTA 可用上限。
