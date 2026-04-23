# Block Reduce

## 定义

一种在整个 thread block 范围内完成归约的实现模式，通常先做 warp 内归约，再把各 warp 的部分结果写入 shared memory，最后做第二轮归约。

## 它解决什么问题

- 把单个 warp 无法覆盖的数据范围扩展到整个 block
- 为 `softmax`、[[RMSNorm]]、`LayerNorm` 等需要块级统计量的 kernel 提供通用骨架
- 在保证正确性的前提下，减少全局内存访存和跨 block 通信需求

## 核心机制

- 每个线程先计算自己的局部统计量
- 先用 [[Warp Shuffle Reduce]] 完成 warp 内归约
- 每个 warp 由一个线程把结果写入 shared memory
- 由前几个线程读取 shared memory 中的 warp 结果并再次归约
- 在关键读写点使用 `__syncthreads()` 保证共享内存可见性

## 关键权衡

- 相比纯 shared-memory 树形归约，warp-first 版本通常更省 shared memory 和同步
- 设计时要注意 `blockDim.x` 与 warp 数关系，以及 shared memory 大小计算
- 如果同步位置错误，容易引入数据竞争或读到未写完的值

## 相关来源

- [[../sources/秋招CUDA手撕题复盘（附代码）]]

## 相关概念

- [[CUDA Kernel]]
- [[Warp Shuffle Reduce]]
- [[Grid-stride Loop]]

## 研究备注

- 这是典型的“面试可背模板”，但真实工程里仍要结合寄存器压力、shared memory 占用和 occupancy 一起看
