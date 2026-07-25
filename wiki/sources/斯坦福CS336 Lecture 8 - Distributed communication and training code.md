---
type: source
source_kind: 课程
topic: 并行与分布式
updated: 2026-04-23
---

# 斯坦福CS336 Lecture 8 - Distributed communication and training code

## 来源信息

- 标题：斯坦福CS336 Lecture 8 - Distributed communication and training code
- 作者：[[../entities/Stanford CS336]]
- 日期：2025 Spring
- 类型：可执行课程讲稿 / 代码
- 原始文件：[[../../raw/articles/斯坦福CS336 Lecture 8 - Distributed communication and training code|斯坦福CS336 Lecture 8 - Distributed communication and training code]]

## 2-3 条核心摘要

- 这讲把 Lecture 7 的并行训练原语直接翻译成最小可运行代码，核心价值是让人看到 `all_reduce / reduce_scatter / all_gather / send / recv` 分别在什么并行模式里出现。
- 课程把分布式训练拆成三层：`collective abstraction -> torch.distributed / NCCL backend -> parallel training skeletons`，把抽象、接口和实现位置区分得很清楚。
- 对 AI infra 来说，这讲最有价值的不是某段玩具 MLP 代码，而是它训练你把“切分什么、同步什么、什么时候通信”作为读并行系统的基本框架。

## 值得关注的论断

- 分布式训练和单 GPU kernel 优化一样，也必须 benchmark 真正的通信路径，而不能只看峰值带宽。
- `torch.distributed` 负责接口整合，NCCL 负责 GPU 通信执行，硬件拓扑则决定这些原语最后到底跑得有多快。
- Data / Tensor / Pipeline parallel 的本质区别，在代码层面往往比在口头定义上更容易看懂。

## 关键概念

- [[集合通信]]
- [[Torch Distributed]]
- [[数据并行]]
- [[Tensor Parallelism]]
- [[流水线并行]]

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/NCCL]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`集合通信`、`Torch Distributed`、`数据并行`、`Tensor Parallelism`、`流水线并行`
- 会更新哪些实体页：`Stanford CS336`、`NCCL`
- 是否存在冲突：与现有 wiki 无直接冲突，但会把并行概念页从“原理”推进到“代码实现骨架”层

## 待确认

- 后续可继续补 PyTorch FSDP、Megatron-LM 或 DeepSpeed 中这些通信骨架如何演化成生产实现
