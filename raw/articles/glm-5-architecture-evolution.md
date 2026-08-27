---
title: GLM-5 系列模型架构演进：DSA、IndexShare、KDA 与 mHC
date: 2026-08-27
tags:
  - GLM
  - LLM Architecture
  - Sparse Attention
  - Linear Attention
  - MoE
aliases:
  - GLM-5 Architecture
  - GLM-5 架构对比
---

# GLM-5 系列模型架构演进：DSA、IndexShare、KDA 与 mHC

> [!summary] 核心结论
> GLM-5 系列的架构演进可以概括为：
>
> **GLM-5 / GLM-5.1：DSA**  
> **GLM-5.2 / GLM-5.3：DSA + IndexShare**  
> **GLM-5.3-Flash：KDA + DSA 混合注意力 + mHC + 原生多模态**
>
> 其中，GLM-5.1 和 GLM-5.3 的主要变化来自后训练而非基础架构；真正发生明显架构变化的是 GLM-5.2 的 IndexShare，以及 GLM-5.3-Flash 的全新混合注意力架构。

## 1. 版本总览

| 模型 | 参数规模 | 层数 / Hidden Size | 注意力结构 | IndexShare | 上下文 | 主要变化 |
|---|---:|---:|---|---|---:|---|
| GLM-5 | 744B-A40B | 78 / 6144 | 每层 DSA | 无 | 约 200K | 首次在 GLM-5 系列中引入 DSA |
| GLM-5.1 | 744B-A40B | 78 / 6144 | 与 GLM-5 相同 | 无 | 约 200K | 主要升级后训练、代码与长程 Agent 能力 |
| GLM-5.2 | 744B-A40B | 78 / 6144 | 每层 DSA | 有 | 1M | 跨层 IndexShare、MTP Index Sharing |
| GLM-5.3 | 744B-A40B | 与 5.2 相同 | 与 GLM-5.2 相同 | 继承 5.2 | 1M | 相同 Base，主要升级后训练 |
| GLM-5.3-Flash | 320B-A18B | 45 / 4096 | 34 KDA + 11 DSA | 无跨层 IndexShare | 1M | 全新 Base、mHC、原生多模态 |

符号说明：

- `744B-A40B`：总参数约 744B，每个 token 激活约 40B 参数。
- `320B-A18B`：总参数约 320B，每个 token 激活约 18B 参数。
- DSA：DeepSeek Sparse Attention。
- KDA：Kimi Delta Attention。
- mHC：Manifold-Constrained Hyper-Connections。
- MTP：Multi-Token Prediction，用于推测解码等场景。

---

## 2. GLM-5：以 MLA、DSA 和 MoE 为主体

GLM-5 是 GLM-5 系列基础架构的起点。其模型配置中的核心字段如下：

```text
architecture             GlmMoeDsaForCausalLM
model_type               glm_moe_dsa
hidden_size              6144
num_hidden_layers        78
num_attention_heads      64
n_routed_experts         256
n_shared_experts         1
num_experts_per_tok      8
first_k_dense_replace    3
moe_intermediate_size    2048
index_topk               2048
index_n_heads            32
index_head_dim           128
q_lora_rank              2048
kv_lora_rank             512
qk_nope_head_dim         192
qk_rope_head_dim         64
v_head_dim               256
max_position_embeddings  202752
num_nextn_predict_layers 1
```

### 2.1 MoE 结构

GLM-5 共有 256 个 routed experts，每个 token 选择 8 个专家，并额外包含一个 shared expert。前三层使用 Dense MLP，后续层使用 MoE：

```text
Layers 0–2   Dense MLP
Layers 3–77  MoE MLP
```

这里的 “Dense” 描述的是 MLP 类型，不代表前三层采用 Dense Attention。GLM-5 的注意力主干仍然是 DSA。

### 2.2 MLA

GLM-5 的注意力投影延续了 MLA（Multi-head Latent Attention）路线，通过低秩表示减少 KV Cache：

- Query LoRA rank：2048
- KV LoRA rank：512
- Q/K 非 RoPE 维度：192
- Q/K RoPE 维度：64
- Value head 维度：256

因此，一个注意力 head 的 Q/K 逻辑维度为：

```text
192 NoPE + 64 RoPE = 256
```

MLA 负责压缩 KV 表示，而 DSA 负责进一步减少真正参与注意力计算的历史 token 数。两者解决的是不同层面的问题。

---

## 3. DSA：DeepSeek Sparse Attention

### 3.1 基本机制

DSA 的核心是在执行 MLA 稀疏注意力之前，先通过一个轻量级 Indexer 从历史上下文中筛选最相关的 token。

一次 DSA 注意力可以简化为：

```text
当前 Query
   │
   ▼
Lightning Indexer 对历史 token 打分
   │
   ▼
选出 Top-K 历史位置
   │
   ▼
仅对这些位置执行 MLA Attention
```

GLM-5 系列的关键 Indexer 配置为：

```text
index_topk      = 2048
index_n_heads   = 32
index_head_dim  = 128
```

也就是说，无论上下文实际有多长，最终稀疏 MLA 主要关注 Indexer 选出的 2048 个历史位置。

### 3.2 DSA 与标准 Attention 的区别

标准全注意力需要让每个 Query 与全部历史 token 计算注意力，而 DSA 把过程拆为：

1. 通过较轻量的 Indexer 扫描历史上下文；
2. 找出 Top-K 候选位置；
3. 只对候选位置执行较重的 MLA 注意力。

这会显著减少真正进入注意力主计算的 token 数。但是，Indexer 本身仍需要处理长上下文；当每一层都独立运行 Indexer 时，Indexer 可能成为 1M 上下文推理中的重要开销。GLM-5.2 的 IndexShare 正是为了解决这一问题。

### 3.3 DSA、MLA 与 MoE 的关系

三者作用在不同模块：

| 技术 | 作用对象 | 主要目标 |
|---|---|---|
| MLA | KV 表示与 Attention 投影 | 压缩 KV Cache |
| DSA | 历史 token 选择 | 减少实际注意力计算量 |
| MoE | FFN/MLP | 用较低激活参数换取更大模型容量 |

因此，GLM-5 不是在 MLA、DSA、MoE 中三选一，而是同时组合使用三者。

---

## 4. GLM-5.1：基础架构不变，主要升级后训练

GLM-5.1 和 GLM-5 的公开 `config.json` 基本相同。两者都使用：

```text
GlmMoeDsaForCausalLM
744B-A40B
78 layers
hidden_size = 6144
256 routed experts
Top-8 routing
DSA Top-K = 2048
```

公开配置中可观察到的主要差别只是记录的 Transformers 版本：

```text
GLM-5    transformers_version = 5.0.2.dev0
GLM-5.1  transformers_version = 5.4.0
```

因此，不能把 GLM-5.1 的能力提升归因于新的基础注意力结构。其主要变化在后训练阶段，包括代码、工具调用、长程 Agent 任务和持续迭代能力的强化。

> [!important]
> GLM-5.1 不是 IndexShare 模型，也没有切换到 KDA。它与 GLM-5 使用同一类 DSA Base Architecture。

---

## 5. GLM-5.2：通过 IndexShare 降低 DSA Indexer 成本

GLM-5.2 保留了 GLM-5/5.1 的整体规模和主体架构：

- 744B 总参数、40B 激活参数；
- 78 层，Hidden Size 6144；
- 256 个 routed experts，每个 token 激活 8 个；
- MLA + DSA；
- DSA Top-K 仍为 2048。

真正重要的架构变化是 IndexShare。

### 5.1 IndexShare 配置

GLM-5.2 新增以下关键字段：

```json
{
  "index_topk_freq": 4,
  "index_skip_topk_offset": 3,
  "index_topk_pattern": null,
  "index_share_for_mtp_iteration": true
}
```

其 `indexer_types` pattern 为：

```text
Layer 0   Full
Layer 1   Full
Layer 2   Full
Layer 3   Shared
Layer 4   Shared
Layer 5   Shared
Layer 6   Full
Layer 7   Shared
Layer 8   Shared
Layer 9   Shared
Layer 10  Full
...
```

在 78 层中共有：

- 21 个 Full Indexer 层；
- 57 个 Shared Indexer 层。

前三层之后，基本呈现“一层计算、三层共享”的规律。

### 5.2 IndexShare 共享的是什么

IndexShare 共享的是 Indexer 输出的 Top-K token indices，而不是 Attention 输出、Q/K/V 权重或 KV Cache。

```text
DSA Layer A
  ├─ 运行 Indexer
  ├─ 得到 Top-K token indices
  └─ 使用这些 indices 执行本层 MLA

DSA Layer B/C/D
  ├─ 复用 Layer A 的 Top-K token indices
  └─ 使用各自的 Q/K/V 执行各自的 MLA
```

因此，各层仍然保留独立的注意力参数和输出，只是避免重复执行相似的长上下文检索过程。

官方说明称，IndexShare 在 1M context 下可将每 token FLOPs 降低约 2.9 倍。

### 5.3 MTP iteration 之间的 Index Sharing

GLM-5.2 还设置了：

```text
index_share_for_mtp_iteration = true
```

这表示在 MTP 推测解码过程中：

1. 第一个 MTP iteration 计算 Top-K indices；
2. 后续 iteration 可复用这组 indices；
3. 避免每个 speculative step 都重复运行 Indexer。

这与跨 Transformer 层的 IndexShare 是两个不同维度的共享：

| 机制 | 共享范围 |
|---|---|
| `index_topk_freq=4` | 不同 Transformer 层之间 |
| `index_share_for_mtp_iteration=true` | 不同 MTP decoding iteration 之间 |

### 5.4 1M 上下文

GLM-5.2 同时修改了长上下文相关配置：

```text
max_position_embeddings = 1,048,576
rope_theta               = 8,000,000
```

相比 GLM-5/5.1 的约 200K 上下文，GLM-5.2 将配置上限扩展到 1M。IndexShare 的价值也主要体现在这种超长上下文场景。

---

## 6. GLM-5.3：沿用 GLM-5.2 Base

GLM-5.3 当前没有公开独立的文本模型 checkpoint config。官方文档明确说明：

> GLM-5.3 uses the same base model as GLM-5.2, with all improvements driven by post-training.

因此，GLM-5.3 的基础架构可视为：

```text
GLM-5.3 Base
  = GLM-5.2 Base
  = 744B-A40B
  + 78 layers
  + MLA
  + DSA
  + IndexShare
  + 1M context
```

GLM-5.3 的主要改进来自后训练，包括更大规模的长程 Agent 环境、代码任务、强化学习、SAO 与 compaction 等，而不是把基础注意力切换成 KDA。

> [!warning]
> “GLM-5.3 使用 KDA”这一说法不准确。使用 KDA + DSA 混合架构的是 **GLM-5.3-Flash**，不是文本旗舰 GLM-5.3。

---

## 7. GLM-5.3-Flash：全新的 KDA + DSA 混合模型

GLM-5.3-Flash 并不是 GLM-5.3 的简单量化或蒸馏版本，而是重新训练的新 Base。其模型入口也从 `GlmMoeDsaForCausalLM` 变为：

```text
architecture  = Glm5NextForConditionalGeneration
model_type    = glm5_next
text type     = glm5_next_text
vision type   = glm5_next_vision
```

### 7.1 模型规模变化

| 配置 | GLM-5.2/5.3 | GLM-5.3-Flash |
|---|---:|---:|
| 总参数 | 744B | 320B |
| 激活参数 | 40B | 18B |
| Transformer 层数 | 78 | 45 |
| Hidden Size | 6144 | 4096 |
| Routed Experts | 256 | 288 |
| 每 token 激活专家 | 8 | 8 |
| Shared Experts | 1 | 1 |
| 前置 Dense MLP 层 | 3 | 3 |

Flash 虽然专家总数从 256 增加到 288，但层数、Hidden Size 和单 token 激活规模显著降低，目标是获得更好的部署效率。

### 7.2 混合注意力层分布

GLM-5.3-Flash 共 45 个文本层，其中：

- 34 个 `linear_attention` 层，即 KDA；
- 11 个 `deepseek_sparse_attention` 层，即 DSA。

层模式基本为：

```text
KDA → KDA → KDA → DSA
KDA → KDA → KDA → DSA
KDA → KDA → KDA → DSA
...
KDA
```

对应层号：

```text
KDA layers:
0, 1, 2,
4, 5, 6,
8, 9, 10,
...
40, 41, 42,
44

DSA layers:
3, 7, 11, 15, 19, 23,
27, 31, 35, 39, 43
```

即大约每三个 KDA 层插入一个 DSA 层，形成 3:1 的线性注意力与稀疏注意力混合架构。

> [!note]
> Flash 配置中的 `full_attn_layers` 是混合线性模型使用的通用字段名。在实际 `layer_types` 中，这些层被标记为 `deepseek_sparse_attention`，因此这里不是普通 Dense Full Attention，而是 DSA。

---

## 8. KDA：Kimi Delta Attention

KDA 是一种线性/循环注意力机制。它不会像标准 Attention 那样在每一步都显式访问完整历史 KV，而是维护固定大小的 recurrent state。

### 8.1 基本思路

可以把 KDA 简化理解为：

```text
旧状态 S(t-1)
   │
   ├─ 根据当前 K/V 和门控进行增量更新
   ▼
新状态 S(t)
   │
   └─ 当前 Query 从状态中读取输出
```

与普通 Attention 相比：

| 项目 | 标准/稀疏 Attention | KDA |
|---|---|---|
| 历史信息载体 | KV Cache | 固定大小 recurrent state |
| 状态大小与上下文长度关系 | 通常随上下文增长 | 基本固定 |
| Decode 单步成本 | 与历史长度相关 | 更接近常数成本 |
| 精确位置检索 | 较强 | 通常弱于显式 Attention |
| 长上下文效率 | 受 KV 与检索成本影响 | 更高 |

KDA 基于 delta-rule 式状态更新，并使用 channel-wise gating，即不同 key/channel 可以拥有不同的衰减和保留行为。这比只使用单个标量 forget gate 的线性注意力具有更细粒度的记忆控制能力。

### 8.2 Flash 中的 KDA 配置

```text
num_heads              = 64
head_dim               = 128
short_conv_kernel_size = 4
gate_lower_bound       = -5.0
```

短卷积用于补充局部建模能力，循环状态负责长期信息聚合。

### 8.3 为什么仍需要 DSA

纯线性注意力擅长低成本压缩和传播历史状态，但精确回忆某个遥远 token、代码符号或文档片段通常更困难。

Flash 的混合设计让两种注意力互补：

- KDA 层：低成本地持续聚合上下文；
- DSA 层：显式检索 Top-K 历史 token，恢复精确长程访问能力。

可以把它理解为：

```text
KDA：持续压缩和更新“工作记忆”
DSA：按需回查“原始历史记录”
```

---

## 9. Flash 中的 DSA 变化

GLM-5.3-Flash 的 DSA 仍然使用 2048 个 Top-K token：

```text
index_topk      = 2048
index_n_heads   = 32
index_head_dim  = 128
```

但它与 GLM-5.2 的 DSA 存在几个区别。

### 9.1 不使用 RoPE 子空间

Flash 设置：

```text
qk_nope_head_dim = 256
qk_rope_head_dim = 0
mla_use_nope      = true
```

而 GLM-5/5.2 使用：

```text
qk_nope_head_dim = 192
qk_rope_head_dim = 64
```

也就是说，Flash 的 DSA/MLA 将整个 256 维 Q/K head 用作 NoPE 部分，不再保留 64 维 RoPE 子空间。

### 9.2 K-pool 压缩

Flash 新增：

```text
index_kpool                    = 4
index_kpool_compress           = true
index_kpool_always_select_tail = true
```

这是针对 DSA Indexer 的另一类效率优化，通过 K-pool 压缩减少索引阶段需要处理的候选表示。

### 9.3 没有跨层 IndexShare

Flash 的 `indexer_types` 全部标记为 `full`。KDA 层本身不使用 DSA Indexer，而每个实际 DSA 层拥有自己的 Full Indexer。

因此 Flash 没有 GLM-5.2 那种“一层 Indexer、后面三层共享”的跨层 IndexShare。不过它仍然设置：

```text
index_share_for_mtp_iteration = true
```

这只表示 MTP iteration 之间可以共享索引，不能等同于 GLM-5.2 的跨 Transformer 层 IndexShare。

---

## 10. mHC：Manifold-Constrained Hyper-Connections

GLM-5.3-Flash 还首次引入 mHC：

```text
mhc               = true
hc_mult           = 4
hc_eps            = 1e-6
hc_sinkhorn_iters = 20
```

传统 Transformer 通常维护单一 residual stream：

```text
x → Attention → Add → MLP → Add
```

Hyper-Connections 将 residual stream 扩展成多条并行连接流，使网络能够以更灵活的方式在不同层之间传递和组合信息。Flash 中的 `hc_mult=4` 表示使用四倍连接流。

mHC 在此基础上加入流形约束，并使用 Sinkhorn 迭代对连接变换进行规范化，主要目标是：

- 提升深层网络的信息流容量；
- 改善规模扩展效率；
- 避免普通 Hyper-Connections 在大模型训练中出现不稳定连接；
- 让 Attention 和 FFN 输出以受约束的方式混合回多条 residual stream。

---

## 11. 原生多模态架构

GLM-5.3-Flash 是 GLM-5 系列首个原生多模态模型。除了 45 层文本模型，它还包含独立 Vision Tower：

```text
vision depth                  = 24
vision hidden_size            = 1024
vision intermediate_size      = 4096
vision num_heads              = 16
image_size                    = 448
patch_size                    = 14
temporal_patch_size           = 2
spatial_merge_size            = 2
projection_intermediate_size  = 10240
out_hidden_size               = 4096
```

Vision Tower 的输出被投影到文本模型的 4096 维 Hidden Space，并通过专用 image/video token 接入统一序列。

因此，Flash 与文本旗舰 GLM-5.3 的关系不是“小号文本模型”，而是：

```text
GLM-5.3
  └─ 744B-A40B 文本旗舰
     沿用 GLM-5.2 Base

GLM-5.3-Flash
  └─ 320B-A18B 全新多模态 Base
     KDA + DSA + mHC
```

---

## 12. 架构演进总结

### 12.1 GLM-5 → GLM-5.1

基础模型配置基本不变：

```text
MLA + DSA + MoE
```

主要变化来自后训练和 Agent 能力强化。

### 12.2 GLM-5.1 → GLM-5.2

主体规模不变，但为 1M 长上下文加入：

```text
DSA + IndexShare
```

通过跨层共享 Top-K indices，降低每一层重复运行 Indexer 的成本；同时增加 MTP iteration 之间的索引共享。

### 12.3 GLM-5.2 → GLM-5.3

Base Architecture 不变：

```text
GLM-5.3 Base = GLM-5.2 Base
```

主要通过后训练获得能力提升。

### 12.4 GLM-5.3 → GLM-5.3-Flash

Flash 是全新训练的 Base，架构发生根本变化：

```text
纯 DSA Stack
   ↓
KDA 为主、DSA 为辅的混合 Stack
   + mHC
   + 原生多模态
   + 更小激活参数
```

---

## 13. 常见误区

### 误区一：GLM-5.2 才开始使用 DSA

不正确。GLM-5 和 GLM-5.1 已经使用 DSA。GLM-5.2 新增的是 IndexShare。

### 误区二：IndexShare 是一种新 Attention

不正确。IndexShare 是 DSA Indexer 的复用机制。真正的 Attention 仍然是每层独立的稀疏 MLA。

### 误区三：GLM-5.3 使用 KDA

不正确。文本旗舰 GLM-5.3 沿用 GLM-5.2 Base。KDA 出现在 GLM-5.3-Flash。

### 误区四：Flash 只有 KDA，没有 Attention 检索能力

不正确。Flash 每三个 KDA 层后插入一个 DSA 层，共有 11 个 DSA 层负责精确 Top-K 长程检索。

### 误区五：Flash 也使用 GLM-5.2 的 IndexShare

不准确。Flash 保留 MTP iteration 之间的 index sharing，但没有 GLM-5.2 那种跨 Transformer 层共享 DSA Top-K indices 的模式。

---

## 14. vLLM 实现对应关系

本文核对时使用的本地 vLLM 版本：

```text
commit 94d96e2446d6
```

### 14.1 GLM-5/5.1/5.2

模型注册入口：

```text
vllm/model_executor/models/registry.py
```

注册名称：

```text
GlmMoeDsaForCausalLM
```

CUDA 实现实际复用 DeepSeek-V3.2 DSA 路径：

```text
vllm/models/deepseek_v32/
├── attention.py
├── nvidia/model.py
├── nvidia/mtp.py
└── nvidia/glm52_low_latency_gemm.py
```

通用兼容实现位于：

```text
vllm/model_executor/models/deepseek_v2.py
```

其中 `GlmMoeDsaForCausalLM` 是 `DeepseekV2ForCausalLM` 的子类；CUDA 路径则将其映射到 DeepSeek-V3.2 DSA 专用实现。

IndexShare 关键字段由以下位置读取：

```text
vllm/models/deepseek_v32/attention.py
vllm/model_executor/models/deepseek_v2.py
```

包括：

```text
index_topk_freq
index_topk_pattern
index_skip_topk_offset
```

### 14.2 KDA

当前仓库已有通用 KDA 实现，主要用于 Kimi-K3/Kimi Linear 等模型：

```text
vllm/models/kimi_k3/nvidia/kda.py
vllm/models/kimi_k3/amd/kda.py
vllm/third_party/flash_linear_attention/ops/kda.py
```

在上述本地 commit 中，还没有找到以下 GLM-5.3-Flash 原生注册：

```text
Glm5NextForConditionalGeneration
glm5_next
glm5_next_text
glm5_next_vision
```

因此，本文对 Flash 的架构判断来自官方 checkpoint 的 `config.json`，而不是该本地 vLLM commit 的模型注册实现。

---

## 15. 一句话记忆

```text
GLM-5       = DSA 起点
GLM-5.1     = 同架构，强化后训练
GLM-5.2     = DSA + IndexShare + 1M
GLM-5.3     = 同 5.2 Base，继续强化后训练
GLM-5.3-F   = KDA × 3 + DSA × 1 + mHC + Vision
```

或者进一步压缩成：

> **DSA 解决“从超长历史中看哪些 token”，IndexShare 解决“不要每层重复找”，KDA 解决“多数层不再保存和扫描完整历史”，mHC 解决“层间信息如何通过更多受约束的 residual streams 流动”。**

---

## 参考资料

### 官方 GLM 仓库与文档

1. Z.ai, **GLM-5 系列官方仓库 README（中文）**  
   <https://github.com/zai-org/GLM-5/blob/main/README_zh.md>

2. Z.ai, **GLM-5.3 官方说明**  
   <https://docs.z.ai/guides/llm/glm-5.3>

3. Z.ai, **GLM-5.2 Blog**  
   <https://z.ai/blog/glm-5.2>

4. Z.ai, **GLM-5.3-Flash Blog**  
   <https://z.ai/blog/glm-5.3-flash>

### 官方模型配置

5. Z.ai, **GLM-5 config.json**  
   <https://huggingface.co/zai-org/GLM-5/blob/main/config.json>

6. Z.ai, **GLM-5.1 config.json**  
   <https://huggingface.co/zai-org/GLM-5.1/blob/main/config.json>

7. Z.ai, **GLM-5.2 config.json**  
   <https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json>

8. Z.ai, **GLM-5.3-Flash config.json**  
   <https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json>

9. Z.ai, **GLM-5.3-Flash-BF16 config.json**  
   <https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/blob/main/config.json>

### 相关架构资料

10. GLM-5 Team, **GLM-5: From Vibe Coding to Agentic Engineering**  
    <https://arxiv.org/abs/2602.15763>

11. Z.ai, **IndexShare**  
    <https://arxiv.org/abs/2603.12201>

12. Moonshot AI, **Kimi Linear / Kimi Delta Attention**  
    <https://arxiv.org/abs/2510.26692>

13. Moonshot AI, **FlashKDA**  
    <https://github.com/moonshotai/FlashKDA>
