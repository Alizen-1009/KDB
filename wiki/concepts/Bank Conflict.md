# Bank Conflict

## 定义

当同一个 warp 中多个线程访问 shared memory 时，如果多个访问命中同一个 bank 且无法并行服务，就会形成 `bank conflict`，导致访问串行化。

## 它解决什么问题

- 帮助识别 shared memory 明明很快却没有发挥预期性能的原因
- 降低由于共享内存地址布局不合理带来的额外等待

## 核心机制

- shared memory 被划分为多个 bank，可以并行服务不同 bank 上的访问
- 若 warp 内多个线程落在同一 bank 上，就需要分多轮完成
- 通过 padding、转置布局或调整线程映射，可以减少 bank conflict

## 关键权衡

- shared memory 能显著提升局部复用，但错误布局会把收益吃掉
- 为减少 bank conflict 引入 padding 或更复杂布局时，也要权衡地址计算和额外存储开销

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/你一定要知道：CUDA优化六要]]

## 相关概念

- [[CUDA Kernel]]
- [[Tiling]]
- [[GPU执行模型]]

## 研究备注

- 后续可补典型 matmul tile 中 `transpose + padding` 的 bank conflict 对比示例
