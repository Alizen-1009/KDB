---
type: source
source_kind: 论文
topic: 模型架构
updated: 2026-04-23
---

# mHC: Manifold-Constrained Hyper-Connections

## 来源信息

- 标题：mHC: Manifold-Constrained Hyper-Connections
- 作者：Zhenda Xie 等；DeepSeek-AI
- 日期：2026-01-05（arXiv v2；初版提交于 2025-12-31）
- 类型：论文 / arXiv
- 原始文件：`raw/papers/2512.24880v2.pdf`

## 2-3 条核心摘要

- 这篇论文把 `Hyper-Connections (HC)` 重新解释为一种宏观拓扑扩展：它通过扩宽 residual stream、增加跨流混合来提升表达力，但无约束混合会破坏 residual 的 identity mapping，从而在大规模训练里引入数值不稳定。
- `mHC` 的核心改动不是换 attention 或 FFN，而是把 `HC` 的 residual mixing 矩阵投影到 `doubly stochastic manifold` 上；论文用 `Sinkhorn-Knopp` 迭代近似实现这个约束，以恢复“平均信号守恒”的传播性质。
- 这篇工作同时是架构与系统共设计：为了让 widened residual stream 在训练里可落地，作者补了 fused kernel、mixed-precision kernel、stage-aligned recomputing，以及在 `DualPipe` 调度中的通信重叠。

## 值得关注的论断

- 论文认为 `HC` 的主要瓶颈不是 FLOPs，而是组合映射失去 identity mapping 后带来的传播不稳定，以及 widened residual stream 带来的额外内存访问开销。
- 在 27B 实验里，`mHC` 相比 `HC` 明显缓和了梯度和 loss 的异常波动；论文给出的稳定性分析里，组合映射的最大 gain 从 `HC` 的接近 `3000` 降到了 `mHC` 的约 `1.6`。
- 按论文报告，`mHC` 在扩展率 `n=4` 的大规模训练里只引入约 `6.7%` 的额外时间开销，同时在多数下游 benchmark 上优于 baseline，并在多数任务上超过 `HC`。

## 关键概念

- [[Hyper-Connections]]
- [[mHC]]
- [[算子融合]]
- [[重计算]]
- [[流水线并行]]

## 相关实体

- [[../entities/DeepSeek-AI]]

## 与现有 wiki 的关系

- 会创建哪些概念页：`Hyper-Connections`、`mHC`
- 会创建哪些实体页：`DeepSeek-AI`
- 会更新哪些概念页：`算子融合`、`重计算`、`流水线并行`
- 是否存在冲突：与现有 wiki 无直接冲突，但需要明确区分 `mHC` 这类宏观连接拓扑改造，和 `MLA` / `MoE` / `Engram` 这类模块级机制

## 待确认

- 公开资料里是否存在官方完整训练实现与 kernel 代码，当前仍待补充；这次 ingest 以 arXiv 论文为主
