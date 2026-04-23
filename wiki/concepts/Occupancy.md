# Occupancy

## 定义

`Occupancy` 指一个 SM 上实际活跃 warp 数占该 SM 理论最大活跃 warp 数的比例，用来衡量 GPU 是否拥有足够并发来隐藏延迟。

## 它解决什么问题

- 避免因为活跃 warp 太少而无法覆盖 global memory 或指令流水线延迟
- 帮助分析寄存器使用、shared memory 占用和 block 大小是否限制了并发度

## 核心机制

- 每个 SM 能同时驻留的 block / warp 数量受寄存器、shared memory 和线程数共同约束
- 如果单个 block 过“重”，即使计算逻辑正确，也可能让可并发 block 数下降
- 更高 occupancy 往往有助于隐藏延迟，但不是越高越好，仍要结合 ILP、访存模式和实际瓶颈判断

## 关键权衡

- 过低 occupancy 常见于寄存器压力过大或 shared memory 过多
- 盲目追求满 occupancy 可能反而破坏单线程效率、tile 设计或数据复用

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/你一定要知道：CUDA优化六要]]

## 相关概念

- [[GPU执行模型]]
- [[CUDA Kernel]]
- [[Tail Effect]]

## 研究备注

- 后续可补 occupancy calculator、寄存器 spilling 与 block size sweep 的具体分析方法
