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

## 关键权衡

- 它能给出高层优化方向，但不能直接替代对具体 kernel 的底层 profiling
- 对复杂算子链路，单一 roofline 视角可能掩盖调度、同步和 cache 行为细节

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 5 - GPUs]]

## 相关概念

- [[算子融合]]
- [[重计算]]
- [[Tiling]]

## 研究备注

- 后续可加入和 LLM 常见算子如 matmul、attention、layernorm 的 intensity 估算例子
