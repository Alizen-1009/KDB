# 斯坦福CS336 Lecture 5 - GPUs

## 来源信息

- 标题：斯坦福CS336 Lecture 5 - GPUs
- 作者：[[../entities/Stanford CS336]]
- 日期：2025 Spring
- 类型：课程讲义 / 视频镜像
- 原始文件：[[../raw/articles/斯坦福CS336 Lecture 5 - GPUs|斯坦福CS336 Lecture 5 - GPUs]]

## 2-3 条核心摘要

- 这讲从硬件层面解释了为什么 LLM 时代的性能优化必须围绕 GPU 展开：GPU 的本质优势在于大规模并行吞吐，而不是单线程低延迟。
- 讲义把 GPU 性能优化组织成一条统一主线：`memory hierarchy -> arithmetic intensity -> coalescing / fusion / tiling / recomputation`，最终用这条主线解释 FlashAttention 为什么有效。
- 按官方课程边界看，这讲重点是 GPU 与 CUDA 性能基础，不是系统性的分布式训练；B 站镜像标题比官方讲义范围更宽。

## 值得关注的论断

- 现代 GPU 上，算力增长速度快于内存带宽增长速度，因此很多 ML workload 的真正瓶颈是数据搬运而不是 FLOPs 本身。
- `Tiling` 是理解高性能矩阵乘法和 FlashAttention 的核心抽象，因为它把 shared memory 复用、coalesced access 和分块计算统一起来。
- FlashAttention 的价值不只是一个更快的 attention kernel，而是“从硬件约束反推算法形态”的代表案例。

## 关键概念

- [[GPU执行模型]]
- [[Roofline 模型]]
- [[算子融合]]
- [[重计算]]
- [[内存合并访问]]
- [[Tiling]]
- [[FlashAttention]]

## 相关实体

- [[../entities/Stanford CS336]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`GPU执行模型`、`Roofline 模型`、`算子融合`、`重计算`、`内存合并访问`、`Tiling`、`FlashAttention`
- 会更新哪些实体页：`Stanford CS336`
- 是否存在冲突：与现有 wiki 无直接冲突，但需要标注“视频标题中的分布式训练表述比官方讲义更宽”

## 待确认

- 后续可补 Stanford CS336 `Lecture 7 - Parallelism basics`，与本讲形成“单 GPU 性能基础 -> 多 GPU 并行训练”连续知识链
