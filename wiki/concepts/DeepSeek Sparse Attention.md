---
type: concept
topic: 注意力机制
sources: 2
updated: 2026-08-27
---

# DeepSeek Sparse Attention

## 定义

`DeepSeek Sparse Attention (DSA)` 是 DeepSeek-V3.2-Exp 的细粒度 token-level sparse attention：轻量 lightning indexer 为当前 query 与全部历史 token 打分，只让 top-k latent KV entries 进入基于 MLA 的核心 attention。

## 它解决什么问题

- 降低长上下文中核心 attention 的二次计算成本。
- 在保留 [[MLA]] latent KV cache 压缩收益的基础上，进一步减少每个 query 实际读取和参与 softmax 的历史条目。
- 允许从 dense DeepSeek-V3.1-Terminus checkpoint 通过 continued training 迁移到 sparse pattern。

## 核心机制

### Lightning Indexer

对 query token `h_t` 与历史 token `h_s`，index score 简化写作：

```text
I_t,s = Σ_j w_t,j · ReLU(q_t,j · k_s)
```

- `q_t,j` 与动态权重 `w_t,j` 来自 query token；`k_s` 来自历史 token。
- Indexer head 数较少，论文称可使用 FP8；但未披露具体 head 数和维度。
- 对每个 query 选出 index score 最高的 `k=2048` 个历史条目。

### Sparse Core Attention

```text
selected = TopK({I_t,s}, k=2048)
u_t = Attention(h_t, {c_s | s in selected})
```

这里 `c_s` 是 MLA latent KV entry。DSA 基于 MLA 的 MQA mode：每个 latent entry 由当前 token 的所有 query heads 共享，方便 kernel 对同一 KV entry 做跨 query 复用。

## 训练流程

### Dense Warm-up

- 起点是上下文已扩到 `128K` 的 DeepSeek-V3.1-Terminus。
- 保持 dense attention，冻结除 lightning indexer 外的所有模型参数。
- 将主 attention 的各 heads 分布求和、沿序列做 L1 normalization，作为 indexer 的 KL distillation target。
- `1000` steps，LR `1e-3`，每步 `16×128K`，约 `2.1B` tokens。

### Sparse Training

- 启用 top-k token selection，并训练主模型适应 sparse pattern。
- Indexer input 从主模型计算图 detach：indexer 只由 top-k 范围内的 KL loss 优化，主模型只由语言模型 loss 优化。
- `15000` steps，LR `7.3e-6`，每步 `480×128K`，约 `943.7B` tokens。

因此 DSA 不是推理阶段临时给 dense model 加一个 top-k filter，而是需要接近 `1T` tokens 的 sparse continued training。

## 与 Qwen Sparse Attention 的区别

| 维度 | DSA | [[Qwen Sparse Attention|QSA]] |
|---|---|---|
| Index 粒度 | 单个历史 token / latent entry | 先压缩成 micro-block，再选块并展开 |
| Indexer 复杂度 | `O(L²)`，但比核心 MLA 轻 | 论文口径 `O(L²/r)` |
| Selection budget | top-`2048` token entries | `K=2048` token budget，对应最多 `512` 个 `r=4` 完整块加尾块 |
| 引入阶段 | `128K` continued training | `256K` CPT |
| Warm-up | `1000` steps、约 `2.1B` tokens | `1000` steps、约 `2B` tokens |
| Sparse joint training | `15000` steps、约 `943.7B` tokens | `8000` steps、约 `200B` tokens |
| 主干语境 | MLA MQA mode | GDN/QSA hybrid 中的 QSA 层 |

两者都使用“dense teacher 初始化 indexer，再联合训练主干适应 sparse pattern”的路线，但粒度、主干、训练预算与 kernel 不能互换。QSA 的 micro-block compression 直接针对 DSA 类 token-level indexer 在长序列上的二次开销。

## GLM 系列中的两种 DSA 组合

- [[../entities/GLM-5 系列|GLM-5 / 5.1 / 5.2]] 使用 `GlmMoeDsaForCausalLM`（`model_type=glm_moe_dsa`）。GLM-5 的 Indexer 为 `index_topk=2048`、`index_n_heads=32`、`index_head_dim=128`；5.2 保留 DSA/MLA 主干，并用 [[IndexShare]] 在 Transformer 层间复用 top-k token indices。
- [[../entities/GLM-5.3-Flash]] 只在 `45` 个文本层中的 `11` 层使用 DSA，其余 `34` 层使用 [[KDA]]。Flash DSA 仍为 top-`2048`、`32×128` Indexer，但 Q/K 采用 `256` 维 NoPE、`0` 维 RoPE，并配置 K-pool 候选压缩与尾部保留。
- Flash 的 DSA `indexer_types` 全为 `full`，所以没有 GLM-5.2 式跨层 IndexShare；它只保留 MTP iterations 间的 index sharing。
- Flash 配置只能确认 K-pool 压缩与 tail 保留，原始整理没有完整 pooling 公式；不能将其直接等同于 [[Qwen Sparse Attention|QSA]] 的 `r=4` average-pool block 公式。
- DSA、[[MLA]] 与 MoE 分别作用于历史 token 选择、KV 表示压缩与 FFN 容量；GLM 前三层的 Dense MLP 不表示 Dense Attention。

## 复杂度与 Serving 边界

- 核心 attention 从 `O(L²)` 变为 `O(Lk)`，其中 `k=2048 ≪ L`。
- Lightning indexer 仍是 `O(L²)`；“整体线性”并不成立，只是 indexer 每次打分比完整 MLA attention 轻得多。
- 论文 Figure 3 使用 H800 GPU 集群实际服务 benchmark，并按 `$2/GPU-hour` 换算；短上下文 prefill 使用 masked MHA mode 模拟 DSA。
- Figure 3 未给精确数字表，不能从图中估读固定美元成本或速度倍数。

## 关键权衡

- 更小的 selected set 可降低 core attention 成本，但 selection recall 决定被遗漏历史信息的风险。
- Indexer 本身仍随 `L²` 增长，极长上下文下可能重新成为瓶颈。
- Dense-to-sparse 迁移训练成本很高；效率收益需要与约 `943.7B` sparse training tokens 一起理解。
- 表 1 未显示整体显著能力退化，但部分任务下降与 reasoning output length 混杂，不能只看单次分数判定 DSA 质量损失。

## 相关实体

- [[../entities/DeepSeek-V3.2-Exp]]
- [[../entities/DeepSeek-AI]]
- [[../entities/GLM-5 系列]]
- [[../entities/GLM-5.3-Flash]]

## 相关来源

- [[../sources/DeepSeek-V3.2-Exp：Boosting Long-Context Efficiency with DeepSeek Sparse Attention]]
- [[../sources/glm-5-architecture-evolution]]

## 相关概念

- [[Qwen Sparse Attention]]
- [[IndexShare]]
- [[KDA]]
- [[MLA]]
- [[KV Cache]]
- [[RoPE]]
- [[Benchmarking]]

## 研究备注

- 待补公开 inference implementation 的 tensor shape、indexer head 配置、FP8 format、prefill/decode kernel 与 memory layout。
- 作者把模型明确标为 experimental，并表示正在做更大规模真实场景验证；当前资料不能代表成熟生产结论。
