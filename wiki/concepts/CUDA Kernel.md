# CUDA Kernel

## 定义

运行在 GPU 上、由大量线程并行执行的函数，是 CUDA 编程模型中的基本计算单元。

## 它解决什么问题

- 让开发者能直接控制 GPU 上的数据并行计算
- 为融合、自定义访存模式和手工优化提供最低层的实现入口

## 核心机制

- 开发者编写单线程视角下的计算逻辑
- 通过 `grid / block / thread` 索引把逻辑映射到大规模并行执行
- 更复杂的 kernel 需要显式考虑 shared memory、同步和数据布局
- 真正影响性能的常见检查项往往集中在访存模式、shared memory 组织、occupancy、控制流分歧和 launch 配置
- 在工程面试和手写题语境里，很多 kernel 又会进一步收束成少量可复用模板，如 [[Warp Shuffle Reduce]]、[[Block Reduce]] 和 [[Grid-stride Loop]]

## 关键权衡

- 控制力最强，适合极致优化
- 实现复杂度高，调试、移植和维护成本也更高

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]
- [[../sources/你一定要知道：CUDA优化六要]]
- [[../sources/秋招CUDA手撕题复盘（附代码）]]

## 相关概念

- [[GPU执行模型]]
- [[Profiling]]
- [[Triton]]
- [[Bank Conflict]]
- [[Occupancy]]
- [[Warp Divergence]]
- [[Warp Shuffle Reduce]]
- [[Block Reduce]]
- [[Grid-stride Loop]]

## 研究备注

- 后续可补 CUDA C++、CUTLASS、PyTorch extension 与自定义 op 之间的关系，以及常见 kernel 优化 checklist 的 profiler 对应信号
- 现有来源已经覆盖“性能原则”和“面试模板”两条线；后续可以继续补 `compute-bound kernel`，把 `GEMM / Tensor Core / FlashAttention` 接进来
