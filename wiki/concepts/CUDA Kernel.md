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
- Kernel launch 第三个参数控制每个 block 的 [[动态共享内存]] 大小；不传时默认是 `0`，只影响 `extern __shared__`，不影响编译期大小固定的静态 shared memory
- 真正影响性能的常见检查项往往集中在访存模式、shared memory 组织、occupancy、控制流分歧和 launch 配置
- 对很多 `memory-bound` kernel，一个很实用的排障顺序是：先看 [[内存合并访问]]，再看 [[Bank Conflict]] / [[Tiling]]，随后检查 [[Occupancy]]、[[Warp Divergence]] 和 [[Tail Effect]]
- 如果 workload 长期只落在少数固定 shape 上，还可以把 tile、线程映射、数据布局和 epilogue 固化成 shape-specialized kernel，再结合 autotune 去逼近该 shape 簇的局部最优
- 在工程面试和手写题语境里，很多 kernel 又会进一步收束成少量可复用模板，如 [[Warp Shuffle Reduce]]、[[Block Reduce]] 和 [[Grid-stride Loop]]

## 关键权衡

- 控制力最强，适合极致优化
- 实现复杂度高，调试、移植和维护成本也更高
- 自定义 kernel 想稳定超越官方库，通常依赖更窄的 workload 假设；shape 一旦变化，历史最优配置很可能失效

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]
- [[../sources/你一定要知道：CUDA优化六要]]
- [[../sources/CUDA优化维度框架]]
- [[../sources/秋招CUDA手撕题复盘（附代码）]]
- [[../sources/CUDA内存层次与动态共享内存问答整理]]

## 相关概念

- [[GPU执行模型]]
- [[CUDA内存层次]]
- [[动态共享内存]]
- [[Profiling]]
- [[Triton]]
- [[Bank Conflict]]
- [[Occupancy]]
- [[Warp Divergence]]
- [[Warp Shuffle Reduce]]
- [[Block Reduce]]
- [[Grid-stride Loop]]

## 研究备注

- 常见的超越官方库路径不是“更底层”本身，而是利用固定 shape、固定 layout 和可融合算子链，把通用问题改造成特化问题
- 后续可补 CUDA C++、CUTLASS、PyTorch extension 与自定义 op 之间的关系，以及常见 kernel 优化 checklist 的 profiler 对应信号
- 现有来源已经覆盖“性能原则”和“面试模板”两条线；后续可以继续补 `compute-bound kernel`，把 `GEMM / Tensor Core / FlashAttention` 接进来
