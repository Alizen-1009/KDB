---
type: source
source_kind: 文章
topic: GPU 编程
updated: 2026-05-06
---

# CUDA优化维度框架

## 来源信息

- 标题：CUDA优化维度框架
- 作者：待确认（原始剪藏未提供明确署名）
- 日期：2026-04-28（剪藏时间）
- 类型：短文 / 图文清单 / CUDA 优化框架
- 原始文件：[[../../raw/articles/CUDA优化维度框架|CUDA优化维度框架]]

## 2-3 条核心摘要

- 这份材料把 CUDA kernel 的常见性能问题压缩成一套六维排障框架：[[内存合并访问]]、[[Bank Conflict]]、[[Occupancy]]、[[Tiling]]、[[Warp Divergence]]、[[Tail Effect]]。
- 它相对现有 `CUDA优化六要` 的增量价值，在于补进了更可执行的判断规则和数字例子，例如 `sectors / requests` 判断合并访问、`tile[32][33]` 规避 bank conflict、以及 `108 SM × 8 Block/SM` 这类 tail effect 估算。
- 这份资料的定位更像 profiling 后的第一轮排查清单，而不是严谨 benchmark；它适合把 wiki 里的 CUDA 基础概念推进到“怎么查、怎么估、怎么解释”的工程口径。

## 值得关注的论断

- `Global Memory` 合并优化减少的是 HBM 搬运量，而 `bank conflict` 优化减少的是 shared memory 内部排队，两者互不替代。
- `Occupancy` 的目标不是机械拉满，而是避免 scheduler 找不到就绪 warp；若 kernel 更偏 `compute-bound`，较低 occupancy 也可能接近跑满算力。
- `Warp divergence` 是 warp 内部问题，不同 warp 走不同分支通常没有额外代价；热点循环里的 per-thread 条件才是更危险的路径。
- `Tail effect` 是否严重，取决于最后一波 block 在总执行时间中的占比，而不只是尾部瞬时利用率低不低。

## 关键概念

- [[CUDA Kernel]]
- [[内存合并访问]]
- [[Bank Conflict]]
- [[Occupancy]]
- [[Tiling]]
- [[Warp Divergence]]
- [[Tail Effect]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`内存合并访问`、`Bank Conflict`、`Occupancy`、`Tiling`、`Warp Divergence`、`Tail Effect`、`CUDA Kernel`、`GPU执行模型`
- 会创建哪些概念页：无
- 是否存在冲突：与现有 wiki 无直接冲突；本次主要是把既有 CUDA 条目从“概念解释”补强到“性能排障 checklist”视角

## 待确认

- 原始 Markdown 正文非常短，核心信息主要集中在配图中；本次摘要基于图文合并阅读整理。
- 文中的 `~400 cycles`、`256 threads/block` 等更接近经验值或教学示例，不应直接视为跨架构恒定规律。
