# Profiling

## 定义

通过时间、调用栈和 kernel 级别的观测数据来定位程序在何处消耗资源的分析方法。

## 它解决什么问题

- 找出时间到底花在哪些算子、kernel 或同步点上
- 判断端到端瓶颈来自 Python、PyTorch 调度、CUDA kernel 还是内存访问模式

## 核心机制
  
- 记录 CPU 与 CUDA 活动
- 聚合每类操作的耗时、调用次数和栈信息
- 将高层算子一路追到实际触发的底层 kernel

## 关键权衡

- Profiling 能解释性能来源，但会引入观测开销
- 采样结果受输入规模、环境和 profiler 配置影响，不应脱离 benchmark 单独解读

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]

## 相关概念

- [[Benchmarking]]
- [[CUDA Kernel]]
- [[Triton]]

## 研究备注

- 后续可补 `torch.profiler`、Nsight Systems、Nsight Compute 各自更适合回答什么问题
