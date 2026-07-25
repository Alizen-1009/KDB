---
type: concept
topic: 模型架构
sources: 2
updated: 2026-05-07
---

# Per-Layer Embeddings

## 定义

为每个 decoder layer 提供额外轻量输入通道的 embedding 机制，使不同层能够接收不同类型的 token-level 条件信号。

## 它解决什么问题

- 缓解“单一输入 embedding 需要服务所有层”的信息压缩问题
- 让浅层和深层能更有针对性地接收不同抽象层次的信息

## 核心机制

- 在主 embedding 之外增加一条独立的 per-layer input 通道
- 实现上不是“每层各有一个独立 embedding 模块”，而是一个共享的 embedding lookup 一次产出所有层要用的切片，再 reshape 成 `[..., num_hidden_layers, hidden_size_per_layer_input]`
- 同时再把主 `inputs_embeds` 通过一个 projection 投到同样的 per-layer 形状，和上面的 token lookup 相加
- 进入具体层时，每层只取自己的那一片 `per_layer_input[:, :, i, :]`
- 每个 decoder layer 再通过自己独立的 `per_layer_input_gate` / `per_layer_projection` 参数，把这份 side input 注入主残差流

## 关键权衡

- 增加了参数量，但重计算成本相对有限
- 能提高表示灵活性，但也引入更复杂的输入注入路径
- 它更像“按层条件输入 + 层内调制”，而不是简单在每层前额外加一个 embedding

## 相关实体

- [[../entities/Gemma 4]]

## 相关来源

- [[../sources/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构]]
- [[../sources/Gemma 4：Drafter 详解]]

## 相关概念

- [[Double-Wide MLP]]
- [[混合注意力]]
- [[MTP Drafter]]

## 研究备注

- Gemma 4 的源码里，PLE 注入位置在 attention 残差和 feedforward 残差之后，作为额外一段层内残差路径，而不是直接替代输入 embedding
- `MTP Drafter` 使用 target activations 与 drafter token embedding 的组合输入，但这和目标模型本体的 PLE 不是同一个机制；两者都体现了 Gemma 4 系列对“额外 embedding/activation 通道”的依赖
