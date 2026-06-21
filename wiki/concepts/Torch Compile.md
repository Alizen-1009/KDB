# Torch Compile

## 定义

PyTorch 2.x 提供的图捕获与编译入口，用于把 Python 层模型或函数自动转换成更高效的执行计划和底层 kernel。

## 它解决什么问题

- 自动完成部分算子融合与代码生成
- 减少开发者手工写 CUDA / Triton kernel 的次数

## 核心机制

- 捕获 Python / PyTorch 计算图
- 对图做优化、融合和代码生成
- 将高层实现降到更适合当前硬件和算子模式的执行路径
- 在 CODA 来源中，`cuBLAS + torch.compile` 被作为基线之一；这体现了通用图编译融合与专门的 `GEMM + epilogue` 重写之间的分工差异。

## 关键权衡

- 对很多常见模式能带来“低改动、高收益”的优化
- 但收益依赖动态图特征、输入形状、图可捕获性和后端成熟度
- 面对需要代数重排和 GEMM epilogue 特化的路径，通用图编译未必能自动发现所有优化机会；这类场景可能仍需要 [[CODA]]、[[Triton]]、CUTLASS/[[CuTe DSL]] 或手写 kernel。
- 在 `vLLM` 语境里，`torch.compile` / vLLM compile 与 CUDA Graphs 是两层优化：`cudagraph_mode=NONE` 只关闭 CUDA Graphs，`--enforce-eager` 才表示关闭 compile 集成并完全走 eager mode。

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/vLLM]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]
- [[../sources/还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速]]

## 相关概念

- [[Triton]]
- [[算子融合]]
- [[Profiling]]
- [[CODA]]

## 研究备注

- 后续可补 Dynamo / AOTAutograd / Inductor 在 `torch.compile` 栈中的分工
- 可继续补充 `torch.compile` 与 CODA 这类 domain-specific kernel abstraction 的边界：前者偏通用图优化，后者偏固定模式的数学重写和内核特化。
