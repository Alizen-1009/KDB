---
type: source
source_kind: 文章
topic: 模型架构
updated: 2026-04-23
---

# Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构

## 来源信息

- 标题：Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构
- 作者：[[特里斯丹井底之娃 往上爬]]
- 日期：2026-04-03
- 类型：文章
- 原始文件：[[../../raw/articles/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构|Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构]]

## 2-3 条核心摘要

- 文章认为 Gemma 4 的参数效率和长上下文能力，主要建立在 `Per-Layer Embeddings (PLE)`、`Shared KV Cache`、以及 `混合注意力 + Dual RoPE` 这三条技术主线上。
- `PLE` 通过为每一层提供轻量级的 layer-specific 输入通道，挑战了“所有层共享同一个 upfront embedding”这一标准 Transformer 假设，是 Gemma 4 架构上的关键创新。
- `Shared KV Cache` 不是服务层的缓存复用，而是模型内部最后若干层共享 K/V 表示，用更低的缓存占用和更少的投影计算换取长上下文推理效率。

## 值得关注的论断

- `PLE` 是 Gemma 4 intelligence-per-parameter 提升的重要来源之一。
- `Shared KV Cache` 和 `Prefix Caching / Cache-aware routing / PD分离` 处在不同优化层级，不能混为一谈。
- `Double-Wide MLP` 可以理解为对共享 KV 层表达能力下降的一种补偿。
- Gemma 4 的长上下文能力来自混合注意力和分层 RoPE 设计，而不只是单纯扩大 context window。

## 关键概念

- [[Per-Layer Embeddings]]
- [[Shared KV Cache]]
- [[混合注意力]]
- [[Dual RoPE]]
- [[Double-Wide MLP]]

## 相关实体

- [[../entities/Gemma 4]]
- [[../entities/Google DeepMind]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`KV Cache`
- 会创建哪些概念页：`Per-Layer Embeddings`、`Shared KV Cache`、`混合注意力`、`Dual RoPE`、`Double-Wide MLP`
- 会创建哪些实体页：`Gemma 4`、`Google DeepMind`
- 是否存在冲突：暂无直接冲突，但 `Double-Wide MLP` 作为补偿机制的解释仍需结合官方实现进一步核实

## 待确认

- 文中对部分源码逻辑做了解释性推断，后续应尽量与官方论文、代码注释或 Hugging Face 实现交叉验证
