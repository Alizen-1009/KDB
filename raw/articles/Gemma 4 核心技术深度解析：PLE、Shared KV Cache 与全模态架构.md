---
title: "Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构"
source: "https://zhuanlan.zhihu.com/p/2023393285226465048"
author:
  - "[[特里斯丹井底之娃 往上爬]]"
published:
created: 2026-04-13
description: "Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构 发布于 2026-04-03 • 公众号具身之道Per-Layer Embeddings (PLE)：动机、原理、伪代码实现、参数效率分析Shared KV Cache：复用机制、内存收益、源…"
tags:
  - "clippings"
---
*发布于 2026-04-03 • 公众号具身之道*

1. **[Per-Layer Embeddings](https://zhida.zhihu.com/search?content_id=272503982&content_type=Article&match_order=1&q=Per-Layer+Embeddings&zhida_source=entity) (PLE)** ：动机、原理、伪代码实现、参数效率分析
2. **Shared KV Cache** ：复用机制、内存收益、源码级实现解析
3. **[混合注意力](https://zhida.zhihu.com/search?content_id=272503982&content_type=Article&match_order=1&q=%E6%B7%B7%E5%90%88%E6%B3%A8%E6%84%8F%E5%8A%9B&zhida_source=entity) + 双 [RoPE](https://zhida.zhihu.com/search?content_id=272503982&content_type=Article&match_order=1&q=RoPE&zhida_source=entity)** ：交替层设计、不同频率配置、long context 支持
4. **[视觉编码器](https://zhida.zhihu.com/search?content_id=272503982&content_type=Article&match_order=1&q=%E8%A7%86%E8%A7%89%E7%BC%96%E7%A0%81%E5%99%A8&zhida_source=entity)** ：可变分辨率、二维 RoPE、池化策略
5. **音频编码器** ： [Conformer 架构](https://zhida.zhihu.com/search?content_id=272503982&content_type=Article&match_order=1&q=Conformer+%E6%9E%B6%E6%9E%84&zhida_source=entity) 、chunked attention、相对位置偏置
6. **[MoE](https://zhida.zhihu.com/search?content_id=272503982&content_type=Article&match_order=1&q=MoE&zhida_source=entity) 与 [Double-Wide MLP](https://zhida.zhihu.com/search?content_id=272503982&content_type=Article&match_order=1&q=Double-Wide+MLP&zhida_source=entity)** ：稀疏专家、补偿机制
7. **性能对比** ：与 Gemma 3、其他尺寸的基准数据
8. **部署生态** ：Transformers、llama.cpp、MLX 等支持

---

## 一、引言：开源模型的新里程碑

2026 年 4 月，Google DeepMind 发布了 **Gemma 4** 系列模型，这是迄今为止最强大的开源模型家族之一。Gemma 4 提供四种尺寸（E2B、E4B、26B MoE、31B Dense），在 Apache 2.0 许可证下全面开源，支持文本、图像、音频（小模型）和视频理解，context window 高达 256K tokens。

更令人震撼的是其性能：31B 模型在 Arena AI 排行榜上位列开源模型第 3 名，26B MoE（仅激活 4B 参数）也冲到第 6 名——这意味着 Gemma 4 以 **20 倍更小的激活参数量** ，挑战了更大模型的性能边界。

这种「 intelligence-per-parameter 」的突破从何而来？答案隐藏在三个核心技术中：

1. **Per-Layer Embeddings (PLE)** ：重新思考 token 表示的注入方式
2. **Shared KV Cache** ：让最后一层 KV 高效复用
3. **混合注意力 + 双 RoPE 配置** ：兼顾局部高效与全局长文本

本文基于 Hugging Face Transformers 源码和官方文档，深入解剖这三项技术，包含 **动机、原理、伪代码和实验效果** 。

---

## 二、Per-Layer Embeddings (PLE)：突破单层 Embedding 的范式

### 2.1 出发动机：标准 Transformer 的瓶颈

在标准 Transformer 中，每个 token 在输入层获得一个 embedding vector，然后这个相同的向量通过所有 decoder 层进行变换。这相当于让单层的 embedding「一次说完所有信息」，迫使它必须包含：

- 词汇语义信息
- 位置信息（虽然位置编码会叠加）
- 语法角色信息
- 领域信息
- 任务特定信息

然而，不同 decoder layer 在不同阶段需要不同的信息：

- 浅层：需要更底层的词汇/句法特征
- 深层：需要更抽象的语义/推理特征

如果所有层都依赖同一个 upfront embedding，会导致 **信息压缩的次优解** 。

### 2.2 技术原理：并行的小型条件通道

PLE 的核心思想是： **为每个 layer 添加一个专属的小型 embedding，与主残差流并行注入信息** 。

具体实现：

```
# 伪代码：标准 vs PLE

# Standard Transformer
input_emb = embed_tokens[token_ids]  # [batch, seq_len, hidden_size]
x = input_emb + position_emb
for layer in layers:
    x = layer(x)  # 所有层都从同一个 embedding 开始

# Gemma 4 with PLE
input_emb = embed_tokens[token_ids]  # 主 embedding
ple_emb = ple_embed_tokens[token_ids]  # 第二 embedding table，更小的 dim（通常 256）
x = input_emb + position_emb
for i, layer in enumerate(layers):
    layer_specific_ple = ple_emb  # 每层有自己的 ple embedding
    # 也可以加 projection 引入语境信息
    context_aware_ple = ple_proj(input_emb)  # 从主 embedding 投影
    ple_signal = layer_specific_ple + context_aware_ple
    x = layer(x, ple_signal)  # 在 layer 内部，ple_signal 作为额外条件
```

根据 Hugging Face 源码（ `modular_gemma4.py` 和 `configuration_gemma4.py` ）：

```
# Gemma4TextConfig 中的 PLE 配置
vocab_size_per_layer_input: int = 262_144  # 较小的 embedding table
hidden_size_per_layer_input: int = 256      # PLE 的维度（主 hidden_size 通常 2304）

# 在 Gemma4TextDecoderLayer 中
if self.hidden_size_per_layer_input:
    self.act_fn = ACT2FN[config.hidden_activation]
    self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
    self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
    self.post_per_layer_input_norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

# 在 forward 中（伪代码）
def forward(self, hidden_states, per_layer_input=None, ...):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Attention 块
    attn_output, _ = self.self_attn(hidden_states, ...)
    hidden_states = self.post_attention_layernorm(attn_output)
    hidden_states = residual + hidden_states

    # PLE 信号注入（在 MLP 之前或同时）
    if per_layer_input is not None:
        ple_signal = self.per_layer_projection(self.per_layer_input_gate(hidden_states))
        ple_signal = self.post_per_layer_input_norm(ple_signal)
        hidden_states = hidden_states + ple_signal  # 轻量级残差

    # MLP
    residual = hidden_states
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states
```

**关键特性：**

1. **双 embedding table** ：
- `embed_tokens`: 主 embedding，维度 = `hidden_size` （如 2304）
- `ple_embed_tokens` （或 `vocab_size_per_layer_input` 对应的 embedding）：第二 embedding，维度 = `hidden_size_per_layer_input` （如 256）
1. **每层独立** ：每个 decoder layer 都有自己的 per-layer embedding vector（从第二 embedding table 查找），这意味着每个 token 在每个 layer 都有专属的通道接收 token-specific 信号。
2. ***context-aware* 投影** ：通过 `per_layer_input_gate` 从主 hidden states 投影，使得 PLE 信号能携带语境信息，而非纯静态 embedding。
3. **参数成本低** ：PLE table 虽然大（26 万词 × 256 ≈ 66M 参数），但只用于快速 lookup，不参与 heavy 矩阵运算，因此「有效参数」不计入 FLOPs 计算。这也是为什么 E2B/E4B 标注为 Effective 2B/4B 而非总参数 5.1B/8B 的原因。

### 2.3 效果与影响

- **参数效率提升** ：E2B 仅 2.3B 有效参数，但总参数（含 PLE）达 5.1B，在保持计算成本低的前提下，让每层都有专属信号通道。
- **多模态场景的巧妙处理** ：对于图像/音频 token，PLE 使用 pad token ID，接收「中性」信号，因为 multimodal token 没有原始 token ID。这避免了多模态嵌入与文本 PLE 的冲突。

---

## 三、Shared KV Cache：大幅削减长上下文推理成本

### 3.1 出发动机：KV Cache 的重复计算

在自回归生成中，KV Cache 存储了所有 previous token 的 key/value 投影，避免重复计算。然而，标准 Transformer 每层都有独立的 K/V 投影，这意味着：

- 每层存储一套 KV → 长上下文下内存占用巨大（layers × KV）
- 相邻层的 KV 通常高度相关（都来自同一 hidden states）
- 最后几层往往只需要关注全局信息，无需每层独立投影

**是否可以让最后 N 层共享同一套 KV？**

### 3.2 技术原理：复用最后非共享层的 KV

Gemma 4 引入 `num_kv_shared_layers` 参数（默认 0），从后往前指定多少层共享 KV：

- **Shared layers** （最后 N 层）：不计算自己的 `k_proj` / `v_proj` ，直接复用 **同类型（sliding/full）的最后一个非共享层** 的 KV
- **Non-shared layers** ：正常计算 KV，同时如果是最后非共享层，则存储 KV 供后续共享层使用

源码解析（ `Gemma4TextAttention` ）：

```
class Gemma4TextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        # ... 其他初始化 ...
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0

        if self.is_kv_shared_layer:
            # 找到同类型最后一个非共享层
            prev_layers = config.layer_types[:first_kv_shared_layer_idx]
            self.kv_shared_layer_index = len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])
            self.store_full_length_kv = False
        else:
            self.kv_shared_layer_index = None
            # 如果是该类型最后一个非共享层，则需要存储 KV 供共享层复用
            self.store_full_length_kv = layer_idx == len(prev_layers) - 1 - prev_layers[::-1].index(config.layer_types[layer_idx])

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        # K/V 计算——复用逻辑
        if self.is_kv_shared_layer and past_key_values is not None:
            # 直接从 shared_layers 字典获取复用 KV
            key_states, value_states = past_key_values.shared_layers[self.kv_shared_layer_index]
            key_states = key_states.to(query_states.device)  # 设备迁移
            value_states = value_states.to(query_states.device)
        else:
            # 正常计算
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        # 更新 KV cache（仅非共享层写，共享层只读）
        if past_key_values is not None:
            if not self.is_kv_shared_layer:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
            if self.store_full_length_kv:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[self.layer_idx] = (key_states, value_states)

        # Attention 计算（省略）
        ...
```

### 3.3 内存与性能收益

假设：

- 32 层模型
- 最后 8 层启用共享
- Hidden size = 2304，num\_attention\_heads = 8，head\_dim = 256

**标准 KV Cache 大小：**

```
每层 KV 大小 = 2 × (batch × seq_len × num_kv_heads × head_dim) × bytes_per_element
32 层总占用 = 32 × 单层 KV
```

**Shared KV Cache（前 24 层独立，后 8 层共享）：**

- 24 层独立 KV
- 8 层共享同一套 KV（同类型：sliding 与 full 分开）
- 总占用 ≈ (24 + 2) / 32 × 原始大小 = 81% 存储

**实际收益** ：内存占用减少约 20%，同时避免后向投影的矩阵乘法（每个共享层省去 k\_proj/v\_proj 的计算），对长上下文生成效率提升明显。

---

## 四、混合注意力机制 + 双 RoPE 配置

### 4.1 交替的局部与全局注意力

Gemma 4 使用 **交替层类型** ：

- `sliding_attention` ：局部窗口注意力（512 或 1024 tokens）
- `full_attention` ：全局注意力，无窗口限制

默认模式：5:1 比例（每 5 个 sliding 层后 1 个 full 层），最后一层强制为 full。

```
# configuration_gemma4.py
sliding_window_pattern = 6
self.layer_types = [
    "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
    for i in range(self.num_hidden_layers)
]
```

**动机：**

- Sliding window：计算 O(n·w) 而非 O(n²)，适合长文本且保证局部信息处理高效
- Full attention：周期性全局视野，建立长距离依赖
- 交替设计在 GPU memory bandwidth 利用率上更优

### 4.2 Dual RoPE：不同频率的旋转位置编码

RoPE（Rotary Position Embedding）将位置信息注入 Q/K 向量。Gemma 4 的不同层类型使用不同的 RoPE 频率：

```
# configuration_gemma4.py —— 默认配置
default_rope_params = {
    "sliding_attention": {
        "rope_type": "default",
        "rope_theta": 10_000.0,  # 标准 RoPE，适合局部
    },
    "full_attention": {
        "rope_type": "proportional",
        "partial_rotary_factor": 0.25,  # 仅 25% 的 head_dim 应用 RoPE
        "rope_theta": 1_000_000.0,      # 更长周期，支持 256K 上下文
    },
}
```

**Proportional RoPE** 的核心：

- 只对 head\_dim 的一部分维度应用旋转（如 25%），其余保持线性
- 更大的 `rope_theta` 意味着更长的有效周期（更平缓的正弦曲线）
- 这允许模型在 long context 下保持位置敏感度，而不会因为过大的位置索引导致 RoPE 值重复

**实现细节** （ `Gemma4TextRotaryEmbedding` ）：

```
class Gemma4TextRotaryEmbedding(Gemma3RotaryEmbedding):
    def __init__(self, config, device=None, layer_type=None):
        # 遍历所有 layer_types，各自 register_buffer
        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params["rope_type"] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            else:
                rope_init_fn = self.compute_default_rope_parameters
            inv_freq, attention_scaling = rope_init_fn(self.config, **kwargs)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
```

在 forward 时，根据 `self.layer_type` 选择对应的 `inv_freq` 进行 cos/sin 计算。

---

## 五、视觉编码器：可变分辨率与二维 RoPE

### 5.1 Patch Embedding + 2D 位置编码

Gemma 4 的视觉编码器继承自 Gemma 3，但增强了可变分辨率支持。

```
class Gemma4VisionPatchEmbedder(nn.Module):
    def __init__(self, config):
        self.input_proj = nn.Linear(3 * patch_size**2, hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, position_embedding_size, hidden_size)  # 2D positions
        )
```

**无需 patch 归一化** ：Gemma 4 直接对像素值做 `2 * (pixel - 0.5)` 缩放（在代码中手动），而非 layer norm。

**二维位置嵌入** ： `position_embedding_table` 的维度 `[2, pos_emb_size, hidden_size]` 分别对应 x 和 y 方向。每个 patch 根据其 `(x, y)` 坐标，通过 one-hot 查找叠加 x 和 y 的嵌入。

### 5.2 可变图像 token 预算

Gemma 4 支持多种图像 token 数量：70, 140, 280, 560, 1120。通过配置 `num_soft_tokens` 控制：

- 较低 token 数（70-140）：适合分类、视频理解（帧数多、需速度）
- 较高 token 数（560-1120）：适合 OCR、文档解析、精细细节

**实现原理** ：视觉编码器输出固定数量的 patch tokens（取决于输入分辨率），然后通过 **空间池化** （ `Gemma4VisionPooler` ）压缩或保持到目标 token 数。

```
class Gemma4VisionPooler(nn.Module):
    def forward(self, hidden_states, pixel_position_ids, padding_positions, output_length):
        if hidden_states.shape[1] != output_length:
            # 通过 2D 空间池化调整 token 数
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length
            )
        hidden_states *= self.root_hidden_size  # 缩放
        return hidden_states, padding_positions
```

**效果** ：高分辨率图像（如 4K）下，如果 patch 太多（比如 56×56 = 3136 patches），会被池化到目标 token 数，大幅减少计算量。

### 5.3 多维 RoPE

视觉 token 同样应用 RoPE，但位置是二维的（x, y）。 `Gemma4VisionRotaryEmbedding` 对每个维度分别计算 cos/sin，然后拼接：

```
def forward(self, x, position_ids):
    all_cos, all_sin = [], []
    for i in range(2):  # x 和 y 两个维度
        dim_position_ids = position_ids[:, :, i]
        freqs = (inv_freq_expanded @ dim_position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        all_cos.append(cos)
        all_sin.append(sin)
    cos = torch.cat(all_cos, dim=-1)
    sin = torch.cat(all_sin, dim=-1)
    return cos, sin
```

这样，视觉 attention 能融入二维空间关系，优于传统 1D RoPE（仅序列位置）。

---

## 六、音频编码器：USM-Style Conformer

音频是 E2B 和 E4B 的专属能力（31B 和 26B 无音频输入）。

### 6.1 架构概览

音频编码器由以下组件构成：

1. **Subsampling Convolutional Projection** ：两层 stride=2 的 Conv2D，将时间维度下采样 4 倍
2. **Conformer Layer** （12 层）：
- Feed-Forward 1
- Chunked Local Attention with Relative Position Bias
- Light Conv1d（深度可分离卷积）
- Feed-Forward 2
1. **RMSNorm** ：每个模块前后均有归一化

**编码器参数** ：

```
hidden_size = 1024
num_layers = 12
num_attention_heads = 8
attention_chunk_size = 12  # 局部 attention 的块大小
attention_context_left = 13  # 左视野
attention_context_right = 0  # 右视野
```

### 6.2 Chunked Attention + 相对位置偏置

音频序列可以很长（30 秒，采样率 16kHz 时 48000 tokens），全注意力不现实。Gemma 4 音频使用 **分块局部 attention** ：

```
class Gemma4AudioAttention(nn.Module):
    def __init__(self, config, layer_idx):
        self.chunk_size = config.attention_chunk_size  # 12
        self.max_past_horizon = config.attention_context_left - 1  # 12
        self.max_future_horizon = config.attention_context_right  # 0
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon  # 24
```

**Forward 流程** ：

1. 将 `[batch, seq_len, hidden]` 的 hidden states 拆成 non-overlapping 的块（ `_convert_to_block` ）
2. 为每个块提取 overlapping context（ `_extract_block_context` ），窗口大小 = `context_size`
3. 计算 attention: `Q @ K^T + 相对位置偏置`

相对位置偏置通过 `relative_k_proj` 将位置编码投影后与 Q 相加：

```
relative_key_states = self.relative_k_proj(position_embeddings)  # [context, num_heads, head_dim]
matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)  # 相对位置项
matrix_bd = self._rel_shift(matrix_bd)  # 移位对齐块内位置
attn_weights = matrix_ac + matrix_bd
```

### 6.3 信号处理细节

- **Clipped Linears** ： `Gemma4ClippableLinear` 在线性层前后对激活进行 clamp（从 checkpoint 读取边界），提升数值稳定性。
- **Gradient Clipping** ：每个 audio layer 内部对激活进行 `torch.clamp(-gradient_clipping, gradient_clipping)` ，防止梯度爆炸。
- **Conv1d Causal Padding** ： `Gemma4AudioCausalConv1d` 计算 left padding = `(kernel_size-1)*dilation + 1 - stride` ，确保因果卷积。

### 6.4 效果

根据官方基准：

| 模型 | CoVoST（翻译任务，↑越高越好） | FLEURS（ASR，↓越低越好） |
| --- | --- | --- |
| E2B | 33.47 | 0.09 |
| E4B | 35.54 | 0.08 |

E4B 在语音识别和翻译上均优于 E2B，且接近更强的通用模型。音频质量在同类开源模型中属于领先水平。

---

## 七、MoE 与 Double-Wide MLP

### 7.1 MoE 架构（26B A4B）

Gemma 4 26B A4B 采用 Sparse MoE：

```
总参数: 25.2B
激活参数: 3.8B  (top_k=4 或 4+?)
专家数: 128
共享专家: 1个（gate 网络共享）
```

**Router 设计** （ `Gemma4TextRouter` ）：

```
class Gemma4TextRouter(nn.Module):
    def forward(self, hidden_states):
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        expert_scores = self.proj(hidden_states)  # linear → [B*S, num_experts]
        router_probabilities = F.softmax(expert_scores, dim=-1)
        # Top-K 选择
        top_k_weights, top_k_index = torch.topk(router_probabilities, k=self.config.top_k_experts, dim=-1)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return router_probabilities, top_k_weights, top_k_index
```

**Double-Wide MLP** ： 在 KV sharing 的层中，MLP 的中间维度会扩大 2 倍：

```
class Gemma4TextMLP(nn.Module):
    def __init__(self, config, layer_idx):
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
        super().__init__()
        self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
```

动机：当 KV projection 被共享后，该层的表达能力可能下降，double-wide MLP 补偿计算能力。

---

## 八、基准性能对比

来自官方和技术社区评测（Arena AI、Hugging Face LLM Leaderboard）：

| Benchmark | Gemma 4 31B | Gemma 4 26B A4B | Gemma 4 E4B | Gemma 3 27B |
| --- | --- | --- | --- | --- |
| MMLU-Pro | 85.2% | 82.6% | 69.4% | 67.6% |
| AIME 2026 (no tools) | 89.2% | 88.3% | 42.5% | 20.8% |
| GPQA Diamond | 84.3% | 82.3% | 58.6% | 42.4% |
| LiveCodeBench v6 | 80.0% | 77.1% | 52.0% | 29.1% |
| Codeforces ELO | 2150 | 1718 | 940 | 110 |
| MMMU Pro (Vision) | 76.9% | 73.8% | 52.6% | 49.7% |
| MATH-Vision | 85.6% | 82.4% | 59.5% | 46.0% |
| MRCR v2 128k (long ctx) | 66.4% | 44.1% | 25.4% | 13.5% |

**观察：**

- 31B Agent 级表现，全面超越 Gemma 3
- 26B A4B 以 4B 激活参数达到 31B 的 95% 以上能力，效率极佳
- E4B 在推理、编码、视觉上均显著优于 Gemma 3，证明 PLE 和架构优化的有效性

---

## 九、部署与生态

### 9.1 支持的后端

Gemma 4 发布即支持：

- Transformers（英伟达 GPU、CPU）
- Llama.cpp（GGUF 量化，支持 Metal、CUDA、CPU）
- MLX（Apple Silicon，支持 TurboQuant 4-bit）
- Transformers.js（WebGPU）
- Mistral.rs（Rust，UQFF 量化）
- SGLang（高效 serving）
- ONNX（部署到边缘设备）

### 9.2 量化与优化

| 方案 | 目标设备 | 精度损失 |
| --- | --- | --- |
| GGUF Q4\_K\_M | 消费级 GPU | <1% |
| GPTQ 4-bit | 英伟达 GPU | ~0.5% |
| AWQ 4-bit | 生产部署 | ~1% |
| MLX TurboQuant 3.5-bit | MacBook | 几乎无损 |
| UQFF 8-bit (mistral.rs) | 推理 API | <1% |

### 9.3 推理代码示例

```
from transformers import AutoProcessor, AutoModelForMultimodalLM
model_id = "google/gemma-4-E4B-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto", dtype="auto")

# 多模态推理
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "image", "url": "https://example.com/image.jpg"},
        {"type": "text", "text": "Describe what you see in detail."}
    ]}
]
inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True,
    enable_thinking=True  # 启用思考模式
).to(model.device)
output = model.generate(**inputs, max_new_tokens=512)
result = processor.decode(output[0], skip_special_tokens=False)
parsed = processor.parse_response(result)  # 分离思考内容与最终答案
print(parsed["content"])
```

---

## 十、总结：开源模型的新高度

Gemma 4 的核心创新可以归纳为：

1. **PLE** ：打破「一个 embedding 走天下」的范式，让每层都有专属的低维条件信号，提升参数效率。这对边缘设备模型尤其重要。
2. **Shared KV Cache** ：让最后 N 层复用 KV，显著降低长上下文的内存占用和计算成本，使 256K 上下文在消费级 GPU 上可行。
3. **混合注意力 + 双 RoPE** ：sliding + full 交替，结合局部高效与全局视野；不同层配不同 RoPE 频率，兼顾短程精度与长程泛化。

此外，可变分辨率视觉、二维 RoPE、Audio Conformer 使 Gemma 4 成为真正的 **全模态统一模型** 。

从规模上看：

- E2B/E4B： **设备端 AI** 的新标准，2-4B 有效参数 + PLE，在手机/树莓派上跑多模态推理
- 26B A4B： **效率之王** ，4B 激活参数 ≈ 13B 模型的推理速度，性能接近 31B
- 31B： **研究/部署首选** ，最佳性能，适合 fine-tuning 和 agent 应用

Gemma 4 于 2026 年 4 月已在 Hugging Face 全面开源（Apache 2.0），配合 Transformers、Llama.cpp、MLX 等生态，可无缝集成到现有工作流。对于具身智能研究，其多模态能力（特别是物体检测、指向、GUI 理解）和本地部署友好性，使其成为 Embodied AI agent 的理想基础模型。

---

## 参考文献

1. Hugging Face Blog: **[Welcome Gemma 4: Frontier multimodal intelligence on device](https://link.zhihu.com/?target=https%3A//huggingface.co/blog/gemma4)**
2. Google Blog: **[Gemma 4: Byte for byte, the most capable open models](https://link.zhihu.com/?target=https%3A//blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)**
3. Transformers 源码: `modeling_gemma4.py`, `configuration_gemma4.py`, `modular_gemma4.py`
4. Gemma 4 Technical Report: **[ai.google.dev/gemma/docs/core](https://link.zhihu.com/?target=https%3A//ai.google.dev/gemma/docs/core)**

---

**致谢** ：感谢 Google DeepMind 的开源贡献，以及 Hugging Face 团队的快速集成。Gemma 4 标志着开源 AI 进入了「每参数效率竞争」的新时代。

编辑于 2026-04-03 13:36・上海