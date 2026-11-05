# GPU 算子有哪些优化方法，如何按瓶颈选择？

- 整理状态：已整理
- 题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，AI Infra 硬件与算子第 2 题；2026-11-05 收录，实际面试日期与具体轮次未提供。

## 原题与条件

讲一下算子有哪些优化方法？

## 30 秒回答

先固定正确性、shape、dtype 和硬件，判断限制来自字节搬运、计算吞吐、并行度、同步还是启动开销。分别考虑融合、合并访问、tiling、Tensor Core、异步流水、局部归约或批处理。资源受限时优化会互相制约，不能把所有技巧无条件叠加。

## 深入解释

| 主要瓶颈 | 候选方法 | 验证什么 |
|---|---|---|
| HBM bytes | 减少中间张量、融合、量化、合并访问 | 实测 traffic、带宽与精度 |
| 复用不足 | Shared/register tiling、L2 局部性 | 复用距离与资源占用 |
| 计算吞吐 | Tensor Core、合适 dtype/layout、独立累加链 | 指令路径、管线利用率 |
| 延迟隐藏不足 | 异步搬运、多 stage、合理并发 | 是否真实重叠、stall 是否减少 |
| 小任务/launch | Batched/grouped 调度、融合、CUDA Graph | 端到端 launch 与同步时间 |
| 原子/同步竞争 | 局部聚合、分层归约、减少屏障 | 竞争、正确性与归约数值差异 |

GEMM 可用“合并加载→分块复用→Tensor Core→多级流水→epilogue 融合”举例，但 reduction、elementwise、稀疏算子的主矛盾可能不同。

更大的 tile、更深流水会消耗寄存器和 SMEM，降低驻留并发甚至引起 spill。只提高 occupancy 也不保证更快。先对照强基线，再用 profiler 确认修改解决了原瓶颈，并评估端到端收益。

具体 GEMM 手段见 [[GEMM 常用的 GPU 优化手段有哪些|GEMM 优化题]]；本页保留跨算子的选择框架。

## 关联知识

- [[../../../../wiki/concepts/Profiling|Profiling]]
- [[../../../../wiki/concepts/Roofline 模型|Roofline 模型]]
- [[../../../../wiki/concepts/Tiling|Tiling]]
- [[../../../../wiki/concepts/算子融合|算子融合]]
- [[../../../../wiki/concepts/Occupancy|Occupancy]]

## 参考来源与待核实

- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。
- 具体收益依赖目标 GPU、shape 和实现；未运行 GPU benchmark。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
