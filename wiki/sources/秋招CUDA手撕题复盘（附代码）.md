# 秋招CUDA手撕题复盘（附代码）

## 来源信息

- 标题：秋招CUDA手撕题复盘（附代码）
- 作者：待确认（原始剪藏未提供明确署名）
- 日期：2026-04-23（剪藏时间）
- 类型：短文 / 面试复盘 / CUDA Kernel 题单
- 原始文件：[[../raw/articles/秋招CUDA手撕题复盘（附代码）|秋招CUDA手撕题复盘（附代码）]]

## 2-3 条核心摘要

- 这篇材料把训练/推理系统岗位里的高频 `memory-bound` CUDA coding 题，整理成一组可手写、可迁移的 kernel 模板，而不是零散题目堆砌。
- 文中反复强调，表面不同的 `Softmax`、`RMSNorm`、`Histogram`、`均值滤波`，核心都能收束到三类套路：[[Warp Shuffle Reduce]]、[[Block Reduce]]、[[Grid-stride Loop]]。
- 它的价值主要不在 benchmark，而在把 CUDA 性能概念推进到“面试时如何快速写出正确骨架”的层面，并给出 `Online Softmax`、shared-memory tiling、shared histogram 等典型实现切口。

## 值得关注的论断

- 在作者经历的秋招训练/推理系统面试中，`memory-bound kernel` 的考察频率明显高于 `compute-bound kernel`。
- `Softmax` 不只是高频题，还经常从三遍扫描版本继续追问到 [[Online Softmax]] 递推式，说明面试会把 kernel 手写能力和 attention 机制理解连起来考。
- `Histogram` 和 `均值滤波` 这类题的重点不在“把功能写出来”，而在是否能主动说出 shared memory、halo、原子冲突消减等优化动机。

## 关键概念

- [[CUDA Kernel]]
- [[Warp Shuffle Reduce]]
- [[Block Reduce]]
- [[Grid-stride Loop]]
- [[Online Softmax]]
- [[RMSNorm]]
- [[Histogram]]
- [[Tiling]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`CUDA Kernel`、`Online Softmax`、`Tiling`
- 会创建哪些概念页：`Warp Shuffle Reduce`、`Block Reduce`、`Grid-stride Loop`、`RMSNorm`、`Histogram`
- 是否存在冲突：与现有 wiki 无直接冲突；本次主要把 CUDA 条目从“性能原则”进一步推进到“面试题型模板”视角

## 待确认

- 原始 Markdown 主要保留了导语，具体题目和代码骨架集中在配图中；本次摘要基于图文合并阅读整理。
- 这篇材料属于经验复盘，不提供 profiler 数据、定量 benchmark 或严格的最优实现证明，更适合作为题型地图和术语锚点，而不是性能结论来源。
