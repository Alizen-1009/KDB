---
type: source
source_kind: 面试整理
topic: KV Cache
updated: 2026-04-23
---

# 美团一面：请介绍 vLLM PageAttention

## 来源信息

- 标题：美团一面：请介绍 vLLM PageAttention
- 作者：待确认（原始剪藏未提供明确署名）
- 日期：2026-04-23（剪藏时间）
- 类型：短文 / 面试复盘 / 推理系统概念讲解
- 原始文件：[[../../raw/articles/美团一面：请介绍 vLLM PageAttention|美团一面：请介绍 vLLM PageAttention]]

## 2-3 条核心摘要

- 这篇材料把 `PagedAttention` 讲成了一个很适合面试回答的操作系统类比：在 token 序列和 GPU 物理显存之间插入一层按块管理的“虚拟地址层”，让 KV cache 的逻辑连续性和物理存放位置解耦。
- 它最有价值的地方是把 `PagedAttention` 拆成了三个明确对象：`Logical KV blocks`、`Physical KV blocks` 和 `Block Table`，并用 `prefill -> decode` 的流程示意它们如何协作。
- 相比只说“PagedAttention 能减少碎片”，这篇材料进一步强调了它减少的是传统按 `(batch_size, max_seq_len)` 预分配策略导致的显存浪费，包括预留槽位浪费、内部碎片和外部碎片。

## 值得关注的论断

- `PagedAttention` 的关键不只是分页本身，而是“逻辑上连续、物理上分散”的映射能力，这让 decode 阶段即使面对零散显存，也能继续把新 token 的 KV 接到已有前缀后面。
- `Block Table` 不只是页表类比，它还是运行时管理的核心结构：谁映射到哪、块是否填满、下一个 token 应该写到哪里，都依赖它维护。
- 这类面试题通常不会只停留在“虚拟内存类比”，更希望候选人能讲清楚：prefill 先构建前缀 KV，decode 再按 block table 取历史 KV 并把新 token 逐步填入物理块。

## 关键概念

- [[PagedAttention]]
- [[KV Cache]]
- [[Continuous Batching]]
- [[Prefix Caching]]

## 相关实体

- [[../entities/vLLM]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`PagedAttention`、`KV Cache`
- 会更新哪些实体页：`vLLM`
- 是否存在冲突：与现有 wiki 无直接冲突；本次主要把 `PagedAttention` 从“分页式内存管理”推进到“block table 驱动的运行流程”视角

## 待确认

- 原始 Markdown 正文主要保留了标题与评论区，具体讲解信息集中在配图中；本次摘要基于图文合并阅读整理。
- 文中对碎片的分类和说明明显偏面试表达，不等同于正式论文中的术语定义，后续如需更严格表述，仍建议补 vLLM 论文或官方文档。
