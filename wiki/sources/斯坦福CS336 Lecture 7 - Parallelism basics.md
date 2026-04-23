# 斯坦福CS336 Lecture 7 - Parallelism basics

## 来源信息

- 标题：斯坦福CS336 Lecture 7 - Parallelism basics
- 作者：[[../entities/Stanford CS336]]
- 日期：2025 Spring
- 类型：课程讲义
- 原始文件：[[../raw/articles/斯坦福CS336 Lecture 7 - Parallelism basics|斯坦福CS336 Lecture 7 - Parallelism basics]]

## 2-3 条核心摘要

- 这讲把大模型训练的并行问题拆成非常清楚的系统账本：内存怎么扩、计算怎么扩、通信代价怎么扩，以及这些目标为什么彼此冲突。
- Stanford 把并行训练组织成一条非常顺的路线：`collective communication -> data parallel / ZeRO / FSDP -> pipeline parallel -> tensor parallel -> sequence parallel -> 3D parallelism`。
- 这讲最重要的工程直觉不是某种并行“最好”，而是不同并行方式分别在消耗 `带宽 / batch size / 实现复杂度 / 利用率` 这几种资源。

## 值得关注的论断

- `ZeRO stage 1` 在带宽受限视角下几乎是“免费”的内存优化，因此很多训练系统会默认吃掉这部分收益。
- Pipeline parallel 通常更适合跨节点慢链路，Tensor parallel 更适合节点内高速互联。
- 真正的大规模训练往往不是 DP、TP、PP 三选一，而是它们的组合，再辅以 sequence parallel 和 recomputation。

## 关键概念

- [[集合通信]]
- [[数据并行]]
- [[ZeRO]]
- [[FSDP]]
- [[流水线并行]]
- [[Tensor Parallelism]]
- [[Sequence Parallelism]]

## 相关实体

- [[../entities/Stanford CS336]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`集合通信`、`数据并行`、`ZeRO`、`FSDP`、`流水线并行`、`Tensor Parallelism`、`Sequence Parallelism`
- 会更新哪些实体页：`Stanford CS336`
- 是否存在冲突：与现有 wiki 无直接冲突，但 `Tensor Parallelism` 需要从“推理降时延”视角扩展到“训练并行”视角

## 待确认

- 后续可继续补 Stanford 对 Expert Parallel、Context Parallel 以及现实训练案例的内容，把这讲和真实开源系统连接起来
