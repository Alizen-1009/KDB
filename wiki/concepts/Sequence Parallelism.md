# Sequence Parallelism

## 定义

把部分按序列维度独立的激活与逐点操作沿 sequence 轴切分到多个设备上，以继续压缩 activation memory 的并行方式。

## 它解决什么问题

- 在张量并行已经分掉参数计算后，继续降低不会自动缩放的 activation memory
- 缓解 layer norm、dropout 和某些逐点操作在长序列训练中的显存压力

## 核心机制

- 将按序列维度可分的中间张量分散到多个设备
- 在需要完整视图时使用 all-gather
- 在适合回收时使用 reduce-scatter 把结果重新切回去

## 关键权衡

- 能把 activation memory 更接近线性地扩展到多设备
- 但会引入额外通信与更复杂的并行调度

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 7 - Parallelism basics]]

## 相关概念

- [[Tensor Parallelism]]
- [[流水线并行]]
- [[重计算]]

## 研究备注

- 后续可补 Sequence Parallel 在 Megatron-LM 中的具体实现位置，以及它和 context parallel 的边界
