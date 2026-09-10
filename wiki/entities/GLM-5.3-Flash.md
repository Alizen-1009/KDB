---
type: entity
entity_type: 模型
topic: 模型架构
sources: 1
updated: 2026-08-27
---

# GLM-5.3-Flash

## 一句话说明

GLM-5.3-Flash 是 [[Z.ai]] 重新训练的 `320B-A18B` 原生多模态 Base，以 `34` 个 KDA 层、`11` 个 DSA 层、mHC 和 Vision Tower 组成；它不是 GLM-5.3 的量化或蒸馏版本。

## 类型

- 模型

## 核心信息

- 架构入口：`Glm5NextForConditionalGeneration`；`model_type=glm5_next`，子配置类型为 `glm5_next_text` / `glm5_next_vision`。
- 文本主干：`45` 层、hidden `4096`、`288` routed experts、top-`8`、`1` shared expert；前三层使用 Dense MLP。
- 层分布：`34` 个 [[../concepts/KDA|KDA]] 层与 `11` 个 [[../concepts/DeepSeek Sparse Attention|DSA]] 层，约 `3:1`。DSA 位于 `3/7/.../43`，最后的 layer `44` 是 KDA。
- 配置字段 `full_attn_layers` 是混合线性模型的通用字段名；实际 `layer_types` 为 `deepseek_sparse_attention`，不是普通 dense full attention。
- 上下文配置为 `1M`；采用 [[../concepts/mHC|mHC]]，并包含原生 Vision Tower。

## KDA 与 DSA

- KDA：`num_heads=64`、`head_dim=128`、`short_conv_kernel_size=4`、`gate_lower_bound=-5.0`。它用固定大小 recurrent state 聚合历史，DSA 则保留显式 top-k token 检索，两者互补。
- DSA：`index_topk=2048`、`index_n_heads=32`、`index_head_dim=128`。
- Flash MLA/DSA 的 Q/K 采用 `qk_nope_head_dim=256`、`qk_rope_head_dim=0`、`mla_use_nope=true`，没有 GLM-5/5.2 的 `192 NoPE + 64 RoPE` 划分。
- K-pool 配置为 `index_kpool=4`、`index_kpool_compress=true`、`index_kpool_always_select_tail=true`。只能确认 Indexer key 候选压缩和尾部保留；没有完整 pooling 公式，不能宣称与 [[../concepts/Qwen Sparse Attention|QSA]] 完全相同或一定是 average pooling。
- 所有 DSA `indexer_types` 均为 `full`，因此没有 GLM-5.2 的跨层 [[../concepts/IndexShare|IndexShare]]；`index_share_for_mtp_iteration=true` 仅表示 MTP iterations 间复用索引。

## mHC 配置观察

```text
mhc               = true
hc_mult           = 4
hc_eps            = 1e-6
hc_sinkhorn_iters = 20
```

这些字段确认 checkpoint 启用了 mHC，但只是配置观察，不是对 mHC 机制或新收益的独立证据。

## Vision Tower

- depth `24`，hidden `1024`，intermediate `4096`，heads `16`。
- image size `448`，patch size `14`，temporal patch `2`，spatial merge `2`。
- projection intermediate `10240`，输出维度 `4096`，与文本 hidden space 对齐。

## 相关概念

- [[../concepts/混合注意力]]
- [[../concepts/KDA]]
- [[../concepts/DeepSeek Sparse Attention]]
- [[../concepts/MLA]]
- [[../concepts/IndexShare]]
- [[../concepts/mHC]]
- [[../concepts/Multi-Token Prediction]]
- [[../concepts/RoPE]]
- [[../concepts/Qwen Sparse Attention]]

## 相关来源

- [[../sources/glm-5-architecture-evolution]]

## 冲突与备注

- 应与 [[GLM-5 系列|GLM-5.3 文本旗舰]]区分：GLM-5.3 官方说明沿用 5.2 Base，而 Flash 是架构完全不同的新 Base。
- 来源核对的 vLLM commit `94d96e2446d6` 有通用 KDA 实现，但未找到 `Glm5NextForConditionalGeneration` / `glm5_next` 原生注册；不能据此声称该 commit 支持本模型。
