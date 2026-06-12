# Roofline 模型

## 定义

用算子 `arithmetic intensity` 与硬件峰值带宽 / 峰值算力共同刻画性能上界的分析模型。

## 它解决什么问题

- 区分一个 workload 当前主要受限于内存带宽还是计算吞吐
- 帮助判断优化应该优先减少访存、提高复用，还是继续压榨 FLOPs

## 核心机制

- 横轴可理解为算子的计算密度，即每搬运一单位数据能做多少计算
- 低 intensity 区域通常由带宽主导，表现为 memory bound
- 高 intensity 区域更接近计算上限，表现为 compute bound
- 对 batch size 1 的小模型 decode，整个 forward 也可以做粗粒度 roofline 账本：如果权重读取主导时延，理论上限近似由 `GPU memory bandwidth / model weight bytes` 决定；实际性能再被 kernel launch、tail effect、activation load/store 和同步开销拉低。

## LLM Attention 粗估

- 对 decode 阶段的标准 `MHA`，单个新 token 需要读取历史 `K/V cache` 并做 `QK` 与 `PV`；忽略 softmax、metadata 与 cache miss 时，算术强度大约是 `4 * H * S * D / (2 * H * S * D * bytes)`，即 `FP16/BF16` 下约 `1 FLOP/byte`。
- `GQA/MQA` 通过减少 `KV heads` 提高同一份 `K/V` 被多个 query heads 复用的次数，粗略 intensity 会随 `H_q / H_kv` 增大，但 decode attention 通常仍偏 memory-bound。
- [[MLA]] 这类 latent KV 结构显著压缩每个历史 token 需要从 HBM 读取的缓存量；如果实现能让 latent cache 在多个 heads 间高效复用，FLOPs 不会按同等比例下降，算术强度可能上升到接近甚至跨过硬件 ridge point，因此有机会从 memory-bound 转向 compute-bound。

## 关键权衡

- 它能给出高层优化方向，但不能直接替代对具体 kernel 的底层 profiling
- 对复杂算子链路，单一 roofline 视角可能掩盖调度、同步和 cache 行为细节
- `Look Ma, No Bubbles!` 来源中的 Llama-1B megakernel 是一个典型例子：纯带宽上限提示 H100 可能达到约 `1350 forward/s`，但多 kernel 边界下的停顿会显著降低端到端上限，因此优化目标从单算子 FLOPs 转向减少 memory pipeline bubble。

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 5 - GPUs]]
- [[../sources/MLA与DP Attention面试整理]]
- [[../sources/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B]]

## 相关概念

- [[算子融合]]
- [[重计算]]
- [[Tiling]]
- [[MLA]]
- [[KV Cache]]
- [[Megakernel]]
- [[Tail Effect]]

## 研究备注

- 这些 attention intensity 估算只适合作为面试和 roofline 直觉：真实数值会受 dtype、KV layout、分页、batch size、kernel 是否使用 Tensor Core、是否能跨 head 复用 latent cache 等因素影响。
- Megakernel 来源中的 `3.35 TB/s / 2.48 GB ~= 1350 forward/s` 是来源作者对 Llama-1B BF16 权重读取上限的粗估；引用时应保留 H100、单序列、16-bit、Llama-1B 的限定。
