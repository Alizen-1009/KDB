# Triton

## 定义

一种面向 GPU kernel 编写的高层编程与编译框架，允许用 Python 风格代码描述块级计算，再编译为高性能 GPU kernel。

## 它解决什么问题

- 降低手写 CUDA kernel 的门槛
- 在保持较强控制力的同时，让一部分 memory coalescing、shared memory 和代码生成细节交给编译器处理

## 核心机制

- 以 block / program 为中心组织 kernel，而不是手工管理每个线程细节
- 用 JIT 编译把 Python 风格 kernel 转成底层 GPU 实现
- 适合实现融合算子、softmax、matmul、norm 等常见高性能 kernel

## 关键权衡

- 比原生 CUDA 更易写、更适合快速迭代
- 仍然要求开发者理解底层数据布局、tile 设计和性能瓶颈

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]

## 相关概念

- [[CUDA Kernel]]
- [[Torch Compile]]
- [[Tiling]]

## 研究备注

- 后续可补 Triton 在 softmax、attention、matmul 和 fused MLP kernel 中的典型使用方式
