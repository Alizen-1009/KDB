# Warp Shuffle Reduce

## 定义

一种在单个 warp 内使用 `__shfl_*_sync` 指令直接交换寄存器值并完成归约的实现模式，常用于求和、求最大值等小范围并行归约。

## 它解决什么问题

- 避免 warp 内归约频繁读写 shared memory
- 降低同步开销，加快 `sum / max` 这类基础归约操作
- 为 [[Block Reduce]]、[[RMSNorm]]、`softmax` 等 kernel 提供更轻量的底层积木

## 核心机制

- 每个线程先持有自己的局部值
- 通过 `offset = 16 -> 8 -> 4 -> 2 -> 1` 逐轮把高位 lane 的值拉到低位 lane
- 每轮执行一次结合操作，如 `+` 或 `fmaxf`
- 最终结果通常只在 `lane 0` 上有意义

## 关键权衡

- warp 内开销低，但天然只适用于 32 线程范围内的协作
- 使用 `_sync` 版本时需要显式给出参与 mask，避免活跃线程集合错误
- 如果问题规模超过一个 warp，仍需与 [[Block Reduce]] 或更高层级归约配合

## 相关来源

- [[../sources/秋招CUDA手撕题复盘（附代码）]]

## 相关概念

- [[CUDA Kernel]]
- [[Block Reduce]]
- [[RMSNorm]]

## 研究备注

- 这类写法在面试里常作为“默认归约模板”；需要能熟练从 `sum` 切换到 `max`、`argmax` 或 pairwise combine 版本
