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
- 实战里通常分两层：`Nsight Systems` 先看系统级时间线与 GPU/CPU 断流，`Nsight Compute` 再看热点 kernel 的微观瓶颈
- 为了避免采到无关阶段，往往会结合 benchmark、warmup 和 `NVTX` 标注只截取关键区间

## 关键权衡

- Profiling 能解释性能来源，但会引入观测开销
- 采样结果受输入规模、环境和 profiler 配置影响，不应脱离 benchmark 单独解读
- 如果还没有先通过 benchmark 稳定复现问题，直接上 profiler 很容易追错热点

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]
- [[../sources/CUDA优化维度框架]]

## 相关概念

- [[Benchmarking]]
- [[CUDA Kernel]]
- [[Occupancy]]
- [[Triton]]

## 研究备注

- 一个实用工作流是：先 `benchmark`，再用 `Nsight Systems` 回答“时间花在哪”，最后用 `Nsight Compute` 回答“这个 kernel 为什么慢”
- `Nsight Systems` 更适合先看 `GPU gap / memcpy / NCCL overlap / stream 并发 / SM Active / DRAM Bandwidth` 这类粗粒度信号
- `Nsight Compute` 更适合看 `achieved occupancy`、`warp stall`、`Tensor Core` 利用率、`L2/DRAM throughput`、coalescing 和 `bank conflict`
- 若问题明显落在访存和 block 配置层面，这份来源补了几个很实用的第一轮信号：`sectors / requests` 看 coalescing、Occupancy 面板看资源瓶颈、shared memory 访存模式看 bank conflict、grid 大小与尾部波次关系看 [[Tail Effect]]
