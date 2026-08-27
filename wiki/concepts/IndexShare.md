---
type: concept
topic: 注意力机制
sources: 1
updated: 2026-08-27
---

# IndexShare

## 定义

`IndexShare` 是 [[DeepSeek Sparse Attention|DSA]] Indexer 的复用机制：让多个计算单元共享已经选出的 top-k 历史 token indices，以避免重复执行长上下文检索；它本身不是新的 Attention 类型。

## 它解决什么问题

- DSA 虽把核心 attention 限制到 top-k 历史 token，但每层独立扫描长上下文的 Indexer 仍可能成为 `1M` context 下的重要成本。
- MTP 推测解码的多个 iteration 若反复选择近似历史位置，也会产生重复 Indexer 开销。

## 核心机制

### 跨 Transformer 层

GLM-5.2 配置为：

```text
index_topk_freq        = 4
index_skip_topk_offset = 3
index_topk_pattern     = null
```

`78` 层中有 `21` 个 Full Indexer 和 `57` 个 Shared Indexer：前三层均独立计算，此后近似“一层计算、三层共享”。

共享对象仅是 **top-k token indices**：

```text
Full Indexer layer  -> 计算 Top-K indices -> 用本层 Q/K/V 执行本层 MLA
Shared layers       -> 复用这些 indices   -> 各自用自己的 Q/K/V 执行自己的 MLA
```

因此不共享 Attention 输出、Q/K/V 参数或 [[KV Cache]]。

### 跨 MTP iterations

`index_share_for_mtp_iteration=true` 表示第一次 MTP iteration 计算索引后，后续 iterations 可以复用。这与跨 Transformer 层的 IndexShare 是两个独立维度：

| 维度 | 共享范围 |
|---|---|
| `index_topk_freq=4` | 不同 Transformer 层 |
| `index_share_for_mtp_iteration=true` | 不同 [[Multi-Token Prediction|MTP]] decoding iterations |

## GLM 版本边界

- [[../entities/GLM-5 系列|GLM-5.2]] 使用跨层 IndexShare，并保留 MTP iteration sharing。
- GLM-5 没有 IndexShare；GLM-5.1 与 GLM-5 公开 Base 配置相同，不能把 5.1 的后训练升级归因于 IndexShare。
- GLM-5.3 官方说明使用与 5.2 相同 Base，但没有独立公开文本 checkpoint config；这一结论来自官方说明，不是直接核对 5.3 config。
- [[../entities/GLM-5.3-Flash]] 的 DSA `indexer_types` 全为 `full`，没有 5.2 式跨层 IndexShare；它仍启用 MTP iteration sharing。

## 与 K-pool / QSA 的边界

- IndexShare 复用已经计算出的 top-k indices。
- K-pool / [[Qwen Sparse Attention|QSA]] 减少 Indexer 的候选表示或打分长度。
- QSA 明确使用 `r=4` average pooling 和 block 公式；GLM-5.3-Flash 只能从配置确认 K-pool 压缩与 tail 保留，不能套用 QSA 公式，也不能断言一定是 average pooling。

## 关键权衡

- 跨层共享降低 Indexer FLOPs，但复用前层选择可能降低每层独立检索的灵活性。
- 官方称 GLM-5.2 在 `1M` context 下每 token FLOPs 约降低 `2.9×`；这是官方声称，当前来源未做本地复测。
- 共享索引不减少各层自身 MLA 的 Q/K/V 投影与 sparse attention 计算。

## 相关实体

- [[../entities/GLM-5 系列]]
- [[../entities/GLM-5.3-Flash]]

## 相关来源

- [[../sources/glm-5-architecture-evolution]]

## 相关概念

- [[DeepSeek Sparse Attention]]
- [[MLA]]
- [[Multi-Token Prediction]]
- [[Qwen Sparse Attention]]

## 研究备注

- 后续可结合 GLM-5.2 / vLLM 的具体实现核对 shared index 生命周期、buffer layout、并行与回退路径。
- `2.9×` FLOPs 结论引用时必须保留 `1M context` 与“官方声称、非本地复测”的边界。
