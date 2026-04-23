# Benchmarking

## 定义

通过可重复的计时实验测量某段代码或某个算子的端到端执行时间，以比较实现差异和观察性能随规模变化的趋势。

## 它解决什么问题

- 判断两个实现到底谁更快
- 识别性能是否按预期随 batch、dim、layers 或 sequence length 扩展

## 核心机制

- 先 warmup，避免首次执行的编译、缓存和初始化噪声
- 对 GPU 代码在计时边界前后进行 `synchronize`
- 进行多次 trial，观察均值和波动，而不是只看单次结果

## 关键权衡

- Benchmarking 给你“快不快”，但不直接告诉你“为什么快”或“慢在哪”
- 如果 workload、硬件、库版本或输入形状变化，历史 benchmark 很可能失效

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]

## 相关概念

- [[Profiling]]
- [[Roofline 模型]]

## 研究备注

- 后续可补 inference benchmark 与 training benchmark 在预热、异步执行和吞吐/时延指标上的差异
