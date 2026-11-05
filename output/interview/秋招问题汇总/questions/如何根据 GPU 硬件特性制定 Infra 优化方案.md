# 如何根据 GPU 硬件特性制定 Infra 优化方案？

- 整理状态：已整理
- 题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，AI Infra 硬件与算子第 3 题；2026-11-05 收录，实际面试日期与具体轮次未提供。

## 原题与条件

在 Infra 过程中怎么根据硬件特性进行优化？

## 30 秒回答

先把业务瓶颈映射到硬件资源：算力/带宽比决定复用需求，Tensor Core 支持决定精度和布局，片上容量约束 tile，异步能力决定流水组织，网络拓扑决定通信域。以目标延迟和成本下的吞吐为结果，用 profiler 验证，而不是仅以使用新指令为目标。

## 深入解释

- 算力相对带宽高：增加权重和输入复用，减少每次 forward 的 HBM bytes。
- Tensor Core：确保 dtype、维度和 layout 适合目标指令路径，不混用稀疏与稠密峰值。
- SMEM/寄存器等片上容量：联合调 tile、线程映射和流水深度，避免 spill 和驻留并发过低。
- 异步搬运：把未来 tile 搬运与当前计算衔接，正确设置同步；分配双缓冲不等于已有重叠。
- 网络拓扑：高频同步尽量放在适合的快互联域，跨节点 EP/TP/状态传输需分别建模。
- 小 workload：若 launch/调度占比高，可评估融合、图执行或 persistent 路径。

FA3 是明确架构条件的例子：面向 Hopper，结合 TMA、异步 Tensor Core 与 warp specialization 重叠搬运、矩阵乘和 Softmax。不能把它的具体收益和指令条件直接套到其他代际。

执行步骤：固定 workload 和 SLA→建立算力/字节/通信账本→选一个主要瓶颈→改实现→测端到端与 kernel 指标→确认瓶颈是否转移。

## 关联知识

- [[../../../../wiki/concepts/Profiling|Profiling]]
- [[../../../../wiki/concepts/Roofline 模型|Roofline 模型]]
- [[../../../../wiki/concepts/通信-计算重叠|通信-计算重叠]]
- [[../../../../wiki/concepts/Tiling|Tiling]]

## 参考来源与待核实

- [FA3 作者说明](https://pytorch.org/blog/flashattention-3/)。
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。
- 架构能力和最佳策略需核对目标 GPU；本批无硬件实测。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
