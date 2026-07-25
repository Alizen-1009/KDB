---
type: source
source_kind: 文章
topic: GPU 编程
updated: 2026-04-23
---

# 你一定要知道：CUDA优化六要

## 来源信息

- 标题：你一定要知道：CUDA优化六要
- 作者：简行AI
- 日期：2026-04-22（剪藏时间）
- 类型：短文 / 工程清单 / CUDA 优化笔记
- 原始文件：[[../../raw/articles/你一定要知道：CUDA优化六要|你一定要知道：CUDA优化六要]]

## 2-3 条核心摘要

- 这篇短文把 CUDA 性能优化压缩成一个非常实用的六项检查清单：`global memory 合并访问`、`shared memory bank conflict`、`occupancy`、`tiling / 数据复用`、`warp divergence`、以及 `launch 配置 / tail effect`。
- 它的价值不在于推导细节，而在于提供了一个很接近真实排障顺序的心智模型：先看访存，再看执行分歧与资源占用，最后看 launch 配置是否把 SM 喂饱。
- 从 AI infra 视角看，这六项正好覆盖了大多数手写 CUDA / Triton / 高性能 attention kernel 的常见瓶颈来源，因此适合作为 profiling 后的第一轮排查框架。

## 值得关注的论断

- CUDA 优化通常不是单点微调，而是访存模式、共享内存组织、并发度和 launch 配置的联动问题。
- `Occupancy` 的目标不是机械拉满，而是在寄存器、shared memory 和线程数之间取得足够好的平衡，以隐藏访存和执行延迟。
- `Block` 数量如果过少，即使单个 block 很快，也可能因为 `tail effect` 让最后一波只占用部分 SM，拖慢整体吞吐。

## 关键概念

- [[CUDA Kernel]]
- [[内存合并访问]]
- [[Bank Conflict]]
- [[Occupancy]]
- [[Tiling]]
- [[Warp Divergence]]
- [[Tail Effect]]

## 相关实体

- [[../entities/Stanford CS336]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`CUDA Kernel`、`GPU执行模型`、`内存合并访问`、`Tiling`
- 会创建哪些概念页：`Bank Conflict`、`Occupancy`、`Warp Divergence`、`Tail Effect`
- 是否存在冲突：与现有 wiki 无直接冲突，但会把现有 GPU / CUDA 基础页从“概念介绍”推进到“性能排障清单”视角

## 待确认

- 这篇文章是 checklist 风格，不提供定量 benchmark；后续若要形成更系统的 CUDA 性能专题，仍需补更正式的 profiling / kernel 资料
