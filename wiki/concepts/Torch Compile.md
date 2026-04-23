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

## 关键权衡

- 对很多常见模式能带来“低改动、高收益”的优化
- 但收益依赖动态图特征、输入形状、图可捕获性和后端成熟度

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]

## 相关概念

- [[Triton]]
- [[算子融合]]
- [[Profiling]]

## 研究备注

- 后续可补 Dynamo / AOTAutograd / Inductor 在 `torch.compile` 栈中的分工
