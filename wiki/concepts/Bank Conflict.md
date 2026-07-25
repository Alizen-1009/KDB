---
type: concept
topic: GPU 编程
sources: 3
updated: 2026-05-06
---

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
- 一个常见教学模型是 `32` 个 bank、每个 bank 宽 `4B`，可用 `bank = (byte_address / 4) % 32` 粗略判断访问映射
- 需要把“同 bank 同地址”和“同 bank 不同地址”区分开：前者通常可视作广播，后者才会形成真正的 conflict 排队
- 在二维 tile 里，按列访问 `tile[32][32]` 往往容易把一列元素全部映到同一 bank；常见修复是把布局改成 `tile[32][33]`，用一列 padding 打散映射

## 关键权衡

- shared memory 能显著提升局部复用，但错误布局会把收益吃掉
- 为减少 bank conflict 引入 padding 或更复杂布局时，也要权衡地址计算和额外存储开销
- 它和 [[内存合并访问]] 解决的是不同层级的问题：前者省 HBM 事务，后者省 shared memory 内部排队，往往需要分别检查

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/你一定要知道：CUDA优化六要]]
- [[../sources/CUDA优化维度框架]]
- [[../sources/CUDA内存层次与动态共享内存问答整理]]

## 相关概念

- [[CUDA Kernel]]
- [[CUDA内存层次]]
- [[动态共享内存]]
- [[Tiling]]
- [[GPU执行模型]]

## 研究备注

- 后续可补典型 matmul tile 中 `transpose + padding` 的 bank conflict 对比示例
