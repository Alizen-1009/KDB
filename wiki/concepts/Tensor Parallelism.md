# Tensor Parallelism

## 定义

在单层内部按张量维度切分权重和计算，让多个 GPU 协同完成同一层前向与反向计算的并行方式。

## 它解决什么问题

- 降低超大模型在单卡上放不下或算不动的问题
- 在多 GPU 环境中压缩单请求时延
- 在训练场景中沿层宽方向拆分大矩阵计算，提升模型可扩展性

## 核心机制

- 把单层权重与矩阵乘法切分到多个 GPU
- 每个 GPU 计算部分结果
- 通过 all-reduce 或类似同步机制在层间汇总
- 通常优先用于节点内高速互联环境，因为每层都会发生较重通信
- 在 Lecture 8 的最小实现里，前向先做局部 matmul，再通过 `all_gather` 拼回完整激活

## 关键权衡

- 能有效降低单请求时延
- 高度依赖节点内高速互联，跨节点通信成本会迅速上升
- 相比流水线并行没有 bubble，但通信更频繁、对带宽要求也更高

## 相关实体

- [[../entities/TensorRT-LLM]]
- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/斯坦福CS336 Lecture 7 - Parallelism basics]]
- [[../sources/斯坦福CS336 Lecture 8 - Distributed communication and training code]]

## 相关概念

- [[PD分离]]
- [[KV Cache]]
- [[Torch Distributed]]
- [[Sequence Parallelism]]
- [[流水线并行]]

## 研究备注

- 后续可补 `Expert Parallelism` 和训练/推理下 Tensor Parallel 的不同瓶颈
