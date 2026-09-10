---
type: source
source_kind: 文章
topic: 模型架构
updated: 2026-08-27
---

# GLM-5 系列模型架构演进：DSA、IndexShare、KDA 与 mHC

## 来源信息

- 标题：GLM-5 系列模型架构演进：DSA、IndexShare、KDA 与 mHC
- 作者：本地技术整理；事实依据见下方参考资料
- 日期：2026-08-27
- 类型：文章
- 原始文件：`raw/articles/glm-5-architecture-evolution.md`
- 资料性质：基于官方 GLM 仓库与文档、官方 Hugging Face checkpoint config、GLM-5 / IndexShare / Kimi Linear 资料及本地 vLLM 源码的综合整理，不是单篇官方技术报告。

## 2-3 条核心摘要

- [[../entities/GLM-5 系列|GLM-5 系列]]存在两条不同的基础架构路线：GLM-5 / 5.1 / 5.2 / 5.3 是 `744B-A40B`、`78` 层主线，其中 5.2 加入跨层 [[../concepts/IndexShare|IndexShare]] 并把配置上限扩展到 `1M`；GLM-5.1 与 5.3 的主要变化来自后训练，不能分别误写成引入 IndexShare / KDA。
- [[../entities/GLM-5.3-Flash]] 是重新训练的 `320B-A18B` 原生多模态 Base，不是 GLM-5.3 的量化或蒸馏版本。其 `45` 个文本层由 `34` 个 [[../concepts/KDA|KDA]] 层与 `11` 个 [[../concepts/DeepSeek Sparse Attention|DSA]] 层按约 `3:1` 组成，并启用 [[../concepts/mHC|mHC]]。
- DSA、[[../concepts/MLA|MLA]] 与 MoE 解决不同问题：DSA 选择要访问的历史 token，MLA 压缩 KV 表示，MoE 扩展 FFN 容量。IndexShare 与 K-pool / [[../concepts/Qwen Sparse Attention|QSA]] 也不能混同：前者复用已有 top-k indices，后两者减少 Indexer 候选或打分长度。

## 值得关注的论断

- GLM-5.2 的 IndexShare 共享对象只是 DSA Indexer 产生的 top-k token indices；共享层仍以自己的 Q/K/V 执行自己的 MLA，不共享 Attention 输出、Q/K/V 参数或 KV Cache。官方称该机制在 `1M` context 下将每 token FLOPs 降低约 `2.9×`，这是官方声称，本文没有本地复测。
- GLM-5.3 官方说明其使用与 GLM-5.2 相同的 Base、改进来自 post-training；由于没有独立公开的文本 checkpoint config，本文没有把 GLM-5.3 写成“已直接核对 config”。
- 本文核对的 vLLM commit `94d96e2446d6` 可确认 GLM-5 / 5.1 / 5.2 的 `GlmMoeDsaForCausalLM` 路径，但没有找到 `Glm5NextForConditionalGeneration` / `glm5_next` 原生注册，因此不能据此声称该 commit 支持 GLM-5.3-Flash。

## 版本矩阵

| 版本 | Base 规模与层数 | 注意力 / Indexer | 上下文与位置 | 事实边界 |
|---|---|---|---|---|
| GLM-5 | `744B-A40B`；`78×6144` | 每层 DSA；无 IndexShare | `max_position_embeddings=202752`，约 `200K` | `GlmMoeDsaForCausalLM` / `glm_moe_dsa` |
| GLM-5.1 | 与 GLM-5 公开 Base 配置相同 | DSA；无 IndexShare / KDA | 约 `200K` | 主要变化来自后训练 |
| GLM-5.2 | `744B-A40B`；`78×6144` | MLA + DSA；跨层 IndexShare；MTP iteration sharing | `1,048,576`；`rope_theta=8,000,000` | 21 个 Full、57 个 Shared Indexer |
| GLM-5.3 | 官方说明与 GLM-5.2 相同 Base | 继承 5.2 路线 | `1M` | 无独立公开文本 checkpoint config；主要改进来自 post-training |
| GLM-5.3-Flash | `320B-A18B`；`45×4096` | `34 KDA + 11 DSA`；无跨层 IndexShare；仍有 MTP iteration sharing | `1M`；DSA 的 Q/K 全部 NoPE | 全新原生多模态 Base，不是 5.3 的量化 / 蒸馏 |

## 实现与配置边界

### GLM-5 / GLM-5.2 的 DSA、MLA 与 MoE

- 架构入口为 `GlmMoeDsaForCausalLM`，`model_type=glm_moe_dsa`。GLM-5 的 Indexer 为 `index_topk=2048`、`index_n_heads=32`、`index_head_dim=128`。
- MLA 参数为 `q_lora_rank=2048`、`kv_lora_rank=512`、`qk_nope_head_dim=192`、`qk_rope_head_dim=64`、`v_head_dim=256`。
- MoE 有 `256` 个 routed experts、top-`8`、`1` 个 shared expert；layers `0–2` 是 Dense MLP，`3–77` 是 MoE MLP。这里的 Dense 描述 FFN，不表示前三层使用 Dense Attention。

### GLM-5.2 IndexShare 的两个维度

- 配置字段为 `index_topk_freq=4`、`index_skip_topk_offset=3`、`index_topk_pattern=null`。`78` 层中前三层为 Full Indexer，此后近似“一层计算、三层共享”，合计 `21` Full、`57` Shared。
- 跨 Transformer 层的 IndexShare 复用 top-k token indices；`index_share_for_mtp_iteration=true` 则在 [[../concepts/Multi-Token Prediction|MTP]] decoding iterations 间复用索引。这是两个独立维度。

### GLM-5.3-Flash

- 模型入口为 `Glm5NextForConditionalGeneration`，`model_type=glm5_next`，并有 `glm5_next_text` / `glm5_next_vision` 子配置。
- 文本侧有 `288` routed experts、top-`8`、`1` shared expert，前三层为 Dense MLP。DSA 位于 layers `3/7/.../43`，layer `44` 为 KDA；配置字段 `full_attn_layers` 只是通用名，实际 `layer_types` 为 `deepseek_sparse_attention`，不是普通 dense full attention。
- KDA 配置为 `num_heads=64`、`head_dim=128`、`short_conv_kernel_size=4`、`gate_lower_bound=-5.0`；固定 recurrent state 与 DSA 的显式 top-k token 检索互补。
- Flash DSA 仍为 `index_topk=2048`、`index_n_heads=32`、`index_head_dim=128`，但 `qk_nope_head_dim=256`、`qk_rope_head_dim=0`、`mla_use_nope=true`。
- `index_kpool=4`、`index_kpool_compress=true`、`index_kpool_always_select_tail=true` 只能确认 Indexer key 候选压缩与尾部保留；原文没有完整 pooling 公式，不能宣称其与 QSA 完全相同或一定使用 average pooling。
- 所有 DSA 的 `indexer_types` 均为 `full`，所以没有 GLM-5.2 的跨层 IndexShare；`index_share_for_mtp_iteration=true` 仍表示 MTP iterations 间复用。
- `mhc=true`、`hc_mult=4`、`hc_eps=1e-6`、`hc_sinkhorn_iters=20` 是 checkpoint 配置观察，不构成新的 mHC 机制证据。
- Vision Tower 配置：depth `24`、hidden `1024`、intermediate `4096`、heads `16`、image `448`、patch `14`、temporal patch `2`、spatial merge `2`、projection intermediate `10240`、输出 `4096`。

### QSA、K-pool 与 IndexShare

- QSA 明确在打分前将 Indexer keys 以 `r=4` 做 average pool，并给出 block 公式；Flash 只能从 config 确认 K-pool 压缩与 tail 保留，不能套用 QSA 公式。
- IndexShare 是复用已经产生的 top-k indices；K-pool / QSA 是压缩 Indexer 候选或打分长度。它们都是 Indexer 优化，不是新的 Attention 类型。

### vLLM commit `94d96e2446d6`

- GLM-5 / 5.1 / 5.2 注册 `GlmMoeDsaForCausalLM`；CUDA 复用 `vllm/models/deepseek_v32/` 的 DSA 路径。
- 通用兼容类位于 `vllm/model_executor/models/deepseek_v2.py`，GLM 类是 `DeepseekV2ForCausalLM` 子类；IndexShare 字段由原文列出的 `deepseek_v32/attention.py` 与 `deepseek_v2.py` 读取。
- 该 commit 已有面向 Kimi K3 的 NVIDIA / AMD 通用 KDA 路径及 third-party FLA 实现，但没有找到 GLM-5.3-Flash 的原生模型注册。未来版本是否支持需重新核实。

## 关键概念

- [[../concepts/IndexShare]]
- [[../concepts/DeepSeek Sparse Attention]]
- [[../concepts/KDA]]
- [[../concepts/MLA]]
- [[../concepts/混合注意力]]
- [[../concepts/mHC]]
- [[../concepts/Multi-Token Prediction]]
- [[../concepts/RoPE]]
- [[../concepts/Qwen Sparse Attention]]

## 相关实体

- [[../entities/GLM-5 系列]]
- [[../entities/GLM-5.3-Flash]]
- [[../entities/Z.ai]]
- [[../entities/vLLM]]

## 与现有 wiki 的关系

- 将 GLM 的两种 DSA 组合接入既有 DSA / MLA 页面：GLM-5 主线使用每层 DSA，5.2 通过 IndexShare 降低 Indexer 重复成本；Flash 则以 KDA 为主、周期性插入 DSA，并使用 K-pool 配置。
- 为混合注意力、MTP、RoPE 与 mHC 增加 GLM-5.3-Flash 这个具体配置案例，同时保留“配置观察不等于新机制证据”的边界。
- 与既有 QSA 页面没有事实冲突；新增的是 K-pool 与 QSA average-pool 公式不能互相套用的实现边界。

## 参考资料

- [GLM-5 官方仓库 README（中文）](https://github.com/zai-org/GLM-5/blob/main/README_zh.md)
- [GLM-5.3 官方说明](https://docs.z.ai/guides/llm/glm-5.3)
- [GLM-5.2 Blog](https://z.ai/blog/glm-5.2)
- [GLM-5.3-Flash Blog](https://z.ai/blog/glm-5.3-flash)
- [GLM-5 config.json](https://huggingface.co/zai-org/GLM-5/blob/main/config.json)、[GLM-5.1 config.json](https://huggingface.co/zai-org/GLM-5.1/blob/main/config.json)、[GLM-5.2 config.json](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
- [GLM-5.3-Flash config.json](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)、[BF16 config.json](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/blob/main/config.json)
- [GLM-5: From Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)
- [IndexShare](https://arxiv.org/abs/2603.12201)
- [Kimi Linear / Kimi Delta Attention](https://arxiv.org/abs/2510.26692)
- [FlashKDA](https://github.com/moonshotai/FlashKDA)

## 待确认

- GLM-5.3 尚无独立公开文本 checkpoint config；后续若发布，应重新核对其是否仍与 GLM-5.2 Base 完全一致。
- GLM-5.3-Flash 的 K-pool 完整 pooling / score 公式，以及未来 vLLM release 的原生支持矩阵，均需结合正式实现继续核实。
- 官方 `1M` context 每 token FLOPs 约降 `2.9×` 的 IndexShare 结论尚未本地复测。
