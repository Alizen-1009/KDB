# Occupancy

## 定义

`Occupancy` 指一个 SM 上实际活跃 warp 数占该 SM 理论最大活跃 warp 数的比例，用来衡量 GPU 是否拥有足够并发来隐藏延迟。

## 它解决什么问题

- 避免因为活跃 warp 太少而无法覆盖 global memory 或指令流水线延迟
- 帮助分析寄存器使用、shared memory 占用和 block 大小是否限制了并发度

## 核心机制

- 每个 SM 能同时驻留的 block / warp 数量受寄存器、shared memory 和线程数共同约束
- 这里的“驻留”表示 warp 上下文已经在 SM 上、可被 scheduler 切换调度，不等于所有 warp 同一瞬间同时执行
- 如果单个 block 过“重”，即使计算逻辑正确，也可能让可并发 block 数下降
- 更高 occupancy 往往有助于隐藏延迟，但不是越高越好，仍要结合 ILP、访存模式和实际瓶颈判断
- 一个常见估算方式是分别从线程数、寄存器、shared memory 和每 SM block 上限四个约束计算可驻留 warp 数，再取最小值
- 例如 `256 threads/block` 对应 `8 warp/block`；若线程数约束允许每 SM 进 `8` 个 block，则恰好是 `64 warp`，接近 `100% occupancy`
- 若每线程寄存器过多，或每 block shared memory 占用过大，即使 block size 合适，也会把可驻留 warp 压低；寄存器溢出到 local memory 还会额外放大延迟

## 关键权衡

- 过低 occupancy 常见于寄存器压力过大或 shared memory 过多
- 盲目追求满 occupancy 可能反而破坏单线程效率、tile 设计或数据复用
- 一个经验判断是：`25% -> 50%` 往往比 `50% -> 100%` 更值得争取；对更偏 `compute-bound` 的 kernel，低 occupancy 也未必就是主要瓶颈
- `256 threads/block` 常被视作默认甜区，但它更适合作为 profiling 前的起点，而不是跨场景定律

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/你一定要知道：CUDA优化六要]]
- [[../sources/CUDA优化维度框架]]
- [[../sources/CUDA内存层次与动态共享内存问答整理]]
- [[../sources/多卡GPU监控与SM执行模型面试整理]]

## 相关概念

- [[GPU执行模型]]
- [[CUDA Kernel]]
- [[CUDA内存层次]]
- [[动态共享内存]]
- [[Tail Effect]]

## 研究备注

- 后续可补 occupancy calculator、寄存器 spilling 与 block size sweep 的具体分析方法
- `Nsight Compute` 的 Occupancy 面板通常能直接提示当前是线程数、寄存器还是 shared memory 在卡住并发
- `SM Active`、`SM Issue` 与 `Occupancy` 不等价：`SM Active` 关注 SM 是否有 warp 驻留，`SM Issue` 关注 warp scheduler 是否在发射指令，`Occupancy` 关注可驻留 warp 数占理论上限的比例。LLM 推理中，decode 小 batch 可能 SM Active/Occupancy 都不高；复杂 fused attention kernel 可能 occupancy 不满但数据复用好；KV cache memory-bound 场景可能 occupancy 足够但 SM Issue/Tensor Active 上不去。
