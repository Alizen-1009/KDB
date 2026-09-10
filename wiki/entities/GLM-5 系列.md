---
type: entity
entity_type: 模型
topic: 模型架构
sources: 1
updated: 2026-08-27
---

# GLM-5 系列

## 一句话说明

GLM-5 系列是 [[Z.ai]] 的模型家族：GLM-5 / 5.1 / 5.2 / 5.3 沿用 `744B-A40B` 文本旗舰主线，[[GLM-5.3-Flash]] 则是采用 KDA + DSA、mHC 与 Vision Tower 的全新 `320B-A18B` 多模态 Base。

## 类型

- 模型

## 版本导航

| 版本 | 基础配置 | 注意力与索引 | 上下文 | 变化边界 |
|---|---|---|---|---|
| GLM-5 | `744B-A40B`；78 层；hidden `6144` | 每层 DSA；无 IndexShare | `202752`，约 `200K` | 系列 DSA 起点 |
| GLM-5.1 | 与 GLM-5 公开 Base 配置相同 | DSA；无 IndexShare / KDA | 约 `200K` | 主要为后训练变化 |
| GLM-5.2 | `744B-A40B`；78 层；hidden `6144` | MLA + DSA；跨层 IndexShare；MTP iteration sharing | `1,048,576` | 新增 IndexShare；`rope_theta=8,000,000` |
| GLM-5.3 | 官方说明使用与 5.2 相同 Base | 继承 5.2 路线 | `1M` | 无独立公开文本 checkpoint config；改进来自 post-training |
| GLM-5.3-Flash | `320B-A18B`；45 文本层；hidden `4096` | `34 KDA + 11 DSA`；无跨层 IndexShare | `1M` | 全新原生多模态 Base，见 [[GLM-5.3-Flash]] |

## GLM-5 / GLM-5.2 主线配置

- 架构为 `GlmMoeDsaForCausalLM`，`model_type=glm_moe_dsa`。
- GLM-5 的 DSA Indexer：`index_topk=2048`、`index_n_heads=32`、`index_head_dim=128`。
- [[../concepts/MLA|MLA]]：`q_lora_rank=2048`、`kv_lora_rank=512`、`qk_nope_head_dim=192`、`qk_rope_head_dim=64`、`v_head_dim=256`。
- MoE：`256` routed experts、top-`8`、`1` shared expert；layers `0–2` 为 Dense MLP，`3–77` 为 MoE MLP。Dense MLP 描述 FFN，不表示 Dense Attention。
- [[../concepts/DeepSeek Sparse Attention|DSA]]、MLA、MoE 分别解决历史 token 选择、KV 压缩与 FFN 容量扩展，不是互斥方案。

## GLM-5.2 / 5.3 边界

- GLM-5.2 的 [[../concepts/IndexShare]] 配置产生 `21` 个 Full 与 `57` 个 Shared Indexer，跨层共享的是 top-k token indices；共享层仍使用自己的 Q/K/V 执行自己的 MLA。
- `index_share_for_mtp_iteration=true` 是 MTP iterations 间复用，与跨 Transformer 层 IndexShare 是两个维度。
- 官方称 `1M` context 下每 token FLOPs 约降低 `2.9×`；该数字未经本地复测。
- GLM-5.3 仅能依据官方说明视为与 GLM-5.2 相同 Base；不能写成已经直接核对独立 5.3 config。

## 相关概念

- [[../concepts/DeepSeek Sparse Attention]]
- [[../concepts/IndexShare]]
- [[../concepts/MLA]]
- [[../concepts/Multi-Token Prediction]]
- [[../concepts/RoPE]]
- [[../concepts/混合注意力]]

## 相关来源

- [[../sources/glm-5-architecture-evolution]]

## 冲突与备注

- “GLM-5.2 才开始使用 DSA”不准确：GLM-5 与 GLM-5.1 已使用 DSA，5.2 新增的是 IndexShare。
- “GLM-5.3 使用 KDA”不准确：KDA + DSA 混合架构属于 GLM-5.3-Flash。
- GLM-5.3-Flash 不是 GLM-5.3 的量化或蒸馏版本，应作为独立 Base 阅读。
