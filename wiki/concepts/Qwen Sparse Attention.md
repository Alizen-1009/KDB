---
type: concept
topic: 注意力机制
sources: 3
updated: 2026-08-27
---

# Qwen Sparse Attention

## 定义

`Qwen Sparse Attention (QSA)` 是 Qwen3.8-Flash-Next 在长上下文 CPT 阶段引入的块级稀疏注意力：轻量 MQA indexer 先在压缩后的 micro-block 上评分，再将选中的块展开给 sparse core attention。

## 它解决什么问题

- 降低长上下文 full attention 的二次复杂度与推理成本。
- 通过先压缩 key 序列，降低 token-level sparse indexer 自身随序列增长的开销。
- 在 GDN/全局注意力混合主干中避免依赖跨层 index 共享。

## 核心机制

- QSA 只在 `256K` continued pretraining（CPT）阶段引入，替换 backbone 与 MTP 中所有 full-attention 层。
- indexer 为 MQA：`4` 个 query heads、`1` 个共享 key head，head dimension `128`，其中 `64` 维使用 partial RoPE。
- key 先按压缩比 `r=4` 切成不重叠块并做平均池化，再对压缩 key 赋该块的起点位置；这避免平均不同 rotary phase。query 仍使用自身 token position。
- 每个 query 的 token budget 为 `K=2048`，最多选择 `512` 个完整块；最后一个未完成块的尾部 tokens 始终加入 sparse attention 集合。
- Stage 1 只训练 indexer：`1000` 步、LR `1e-3`、每步 `8×256K`，约 `2B` tokens。Stage 2 联合训练 backbone 与 indexer：`8000` 步、LR `2.5e-5`、每步 `96×256K`，约 `200B` tokens。
- 多步 MTP 可跨 prediction steps 复用 top-k indices；论文的四步实验中平均接受长度为 `4.06→4.07`。

## 评测与效率边界

- 短上下文八项任务中 QSA 在 `7/8` 项不降，均分 `75.9→76.8`。
- RULER `512K–1M` 为 `90.08→93.00`；MRCR 在 `512K` 为 `30.66→40.53`、在 `1M` 为 `20.71→26.44`。
- `1M` kernel 实验中，prefill 条件是 chunked prefill 最后 `16K`、`BS=1`；decode 条件是 `BS=4`、`next_n=4`，包括 `3` 步 MTP。相对 FlashInfer paged GQA attention，包含 indexer 与 sparse core 的 QSA attention module 在该设置下 prefill/decode 为 `7.6×/4.9×`。
- 上述数字是指定 kernel-level module 实验，不代表任意硬件、batch、context 或端到端 serving 的固定加速。

## 与 DeepSeek Sparse Attention 的区别

[[DeepSeek Sparse Attention|DSA]] 直接对历史 token / MLA latent entries 做细粒度打分，lightning indexer 仍为 `O(L²)`，再选择 top-`2048` entries 执行 `O(Lk)` core attention。QSA 则先按 `r=4` 压缩成 micro-block，把 indexer 降为论文口径 `O(L²/r)`，再展开选中块；其 `K=2048` 也是 token budget，而不是 `2048` 个块。

两者都先用 dense attention distribution 蒸馏 indexer，再联合训练主模型适应 sparse pattern，但训练语境不同：DSA 基于 `128K` MLA MQA mode，Sparse Training 约 `943.7B` tokens；QSA 在 `256K` GDN hybrid CPT 中使用，联合阶段约 `200B` tokens。不能仅凭相同的 top-k 数字或两阶段流程把二者视为同一实现。

## 与 GLM K-pool / IndexShare 的边界

- QSA 明确在 Indexer 打分前按 `r=4` 对 keys 做 average pooling，并给出 micro-block 选择与展开公式。
- [[../entities/GLM-5.3-Flash]] 只能从配置确认 `index_kpool=4`、`index_kpool_compress=true` 与 tail always-select；其来源没有完整 pooling 公式，不能宣称它与 QSA 完全相同或一定使用 average pooling。
- [[IndexShare]] 复用已经选出的 top-k token indices；QSA / K-pool 则减少 Indexer 候选或打分长度。二者都不是新的 Attention 类型，也不能因数值同为 `4` 就视为同一机制。

## 关键权衡

- 块压缩将 indexer 复杂度从论文所述 `O(n²)` 降至 `O(n²/r)`，但仍需额外 indexer、top-k 与 sparse kernel。
- 稀疏模式不是拿来即用：dense distillation 后直接启用稀疏 attention 会掉点，需要 Stage 2 联合训练让 backbone 适应。
- top-k 选择保留固定 token budget，直接 token-level retrieval 能力与计算成本受 `r`、`K` 和尾块规则共同影响。

## 相关实体

- [[../entities/Qwen3.8-Flash-Next]]
- [[../entities/FlashInfer]]
- [[../entities/GLM-5.3-Flash]]

## 相关来源

- [[../sources/On the Design of Qwen3.8-Next Architecture：Evaluation, Efficiency, and Training Stability]]
- [[../sources/DeepSeek-V3.2-Exp：Boosting Long-Context Efficiency with DeepSeek Sparse Attention]]
- [[../sources/glm-5-architecture-evolution]]

## 相关概念

- [[混合注意力]]
- [[RoPE]]
- [[Chunked Prefill]]
- [[Multi-Token Prediction]]
- [[DeepSeek Sparse Attention]]
- [[IndexShare]]

## 研究备注

- 论文没有给出可脱离指定实验条件泛化的端到端吞吐或延迟结论；引用 `7.6×/4.9×` 时必须同时保留上下文、batch、chunk、MTP 与 FlashInfer baseline。
