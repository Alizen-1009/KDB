---
type: source
source_kind: 论文
topic: 模型架构
updated: 2026-04-23
---

# Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

## 来源信息

- 标题：Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
- 作者：Xin Cheng 等；Peking University / DeepSeek-AI
- 日期：2026-01-12（arXiv 提交时间）
- 类型：论文 / 官方实现
- 原始文件：`raw/papers/2601.07372v1.pdf`

## 2-3 条核心摘要

- 这篇论文提出 `conditional memory` 作为与 MoE 条件计算互补的另一条稀疏轴，并用 `Engram` 模块把经典 `N-gram embedding` 现代化为一种 `O(1)` 查找的静态记忆原语。
- 论文把“给定总参数与 FLOPs 预算，稀疏容量应如何在 MoE 专家与记忆表之间分配”形式化为 `Sparsity Allocation` 问题，并观察到稳定的 U 型规律：纯 MoE 不是最优点，混合分配更好。
- 这篇工作的 AI infra 含量很高：训练阶段通过表分片 + `All-to-All` 获取激活行，推理阶段依靠确定性地址做 host-memory prefetch，强调“存储与计算解耦”的系统可扩展性。

## 值得关注的论断

- `Engram-27B` 在严格 `iso-parameter` 与 `iso-FLOPs` 条件下优于 `MoE-27B`，而且收益不只体现在知识任务，也明显体现在推理、代码和数学任务上。
- 论文的机制解释是：Engram 把早期层原本用于“静态模式重建”的工作外包给查表，等价于给主干保留了更多有效深度去处理复杂推理。
- 由于检索地址只依赖输入 token 序列而不依赖运行时 hidden state，Engram 比动态路由更适合做推理时的异步预取和大表下沉。

## 关键概念

- [[Conditional Memory]]
- [[Sparsity Allocation]]

## 相关实体

- [[../entities/Engram]]

## 与现有 wiki 的关系

- 会创建哪些概念页：`Conditional Memory`、`Sparsity Allocation`
- 会创建哪些实体页：`Engram`
- 是否存在冲突：与现有 wiki 无直接冲突，但需要明确区分它和 `RAG`、`KV Cache`、`Prefix Caching` 这类运行时检索或缓存机制

## 待确认

- 当前仓库中的原始资料只有 arXiv PDF；后续若补官方技术报告、训练配置或更完整源码，可继续细化其系统实现细节
