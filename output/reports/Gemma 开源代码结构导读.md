# Gemma 开源代码结构导读

## 结论先看

如果你是想看清楚 Gemma 的模型代码结构，建议分成两条线：

- 看 Google 官方的简洁 PyTorch 参考实现：`google/gemma_pytorch`
- 看 Gemma 4 的当前可读实现：Hugging Face Transformers 的 `gemma4/`

原因很直接：

- `google/gemma_pytorch` 更适合快速建立“Gemma 模型骨架”直觉
- Gemma 4 的一些关键设计，例如 `Per-Layer Embeddings`、`Shared KV Cache`、混合注意力层型、双倍宽度 MLP，在 Transformers 的 `gemma4` 目录里暴露得更直接

## 官方源码入口

### 1. Google 官方 PyTorch 仓库

- 仓库：[google/gemma_pytorch](https://github.com/google/gemma_pytorch)
- 推荐先看：
  - [`gemma/config.py`](https://github.com/google/gemma_pytorch/blob/main/gemma/config.py)
  - [`gemma/model.py`](https://github.com/google/gemma_pytorch/blob/main/gemma/model.py)
  - [`gemma/gemma3_model.py`](https://github.com/google/gemma_pytorch/blob/main/gemma/gemma3_model.py)

这个仓库里，`gemma/model.py` 很适合用来建立最基础的阅读地图。核心类基本都在一个文件里，结构比较直白：

- `Sampler`
- `Linear`
- `Embedding`
- `RMSNorm`
- `GemmaMLP`
- `GemmaAttention`
- `GemmaDecoderLayer`
- `GemmaModel`
- `GemmaForCausalLM`

如果你想先把“标准 decoder-only 模型是怎么搭起来的”看透，这个入口很顺。

### 2. Gemma 4 的当前实现入口

- 目录：[transformers/models/gemma4](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4)
- 推荐先看：
  - [`configuration_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/configuration_gemma4.py)
  - [`modular_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modular_gemma4.py)
  - [`modeling_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)
  - [`processing_gemma4.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/processing_gemma4.py)

其中最重要的一点是：

- `modeling_gemma4.py` 是自动生成文件，更适合查 API 和最终展开后的实现
- `modular_gemma4.py` 才是更值得读的结构源文件

## 你最该看的几个代码位置

### 1. 先看配置，别一上来扎进 forward

`configuration_gemma4.py` 是理解 Gemma 4 的最佳入口，因为很多二手文章里的术语，在源码里都落成了明确字段。

最值得注意的字段：

- `vocab_size_per_layer_input`
- `hidden_size_per_layer_input`
- `num_kv_shared_layers`
- `use_double_wide_mlp`
- `enable_moe_block`
- `layer_types`
- `rope_parameters`

这几个字段直接告诉你：Gemma 4 不是“普通 decoder 堆很多层”这么简单，而是在文本主干里显式加入了逐层附加输入、KV 共享、不同注意力类型混排、不同 RoPE 策略等机制。

另外，默认 `layer_types` 会生成一种滑窗注意力和全局注意力混排的模式，最后一层会被强制设为 `full_attention`。这比很多解读文章都更接近真实实现。

### 2. Shared KV Cache 在哪里

`modular_gemma4.py` 里的 `Gemma4TextAttention` 是 Shared KV Cache 的核心入口。

重点看这些逻辑：

- `num_kv_shared_layers`
- `is_kv_shared_layer`
- `kv_shared_layer_index`
- `store_full_length_kv`

这里能直接看到它不是“所有层公用一套 KV”，而是：

- 先确定从哪一层开始进入共享区
- 再按层类型找到前面最后一个同类型的非共享层
- 共享层复用对应的 KV 状态，而不是自己再持有一套完整 K/V 投影参数和缓存

这部分源码比文章里的口头解释清楚得多。

### 3. PLE 在哪里

如果你前面看到有文章把它写成别的缩写，建议以后以源码命名为准。Gemma 4 代码里最直接的落点是：

- `hidden_size_per_layer_input`
- `vocab_size_per_layer_input`
- `get_per_layer_inputs()`
- `project_per_layer_inputs()`
- `per_layer_input_gate`
- `per_layer_projection`

这些实现集中出现在：

- `Gemma4TextDecoderLayer`
- `Gemma4TextModel`

这里最容易被讲错的一点是：

- 它不是“每层各自定义一个独立的 PLE embedding 模块”
- 更准确地说，是一个共享的 `embed_tokens_per_layer` 一次 lookup 出所有层的 per-layer token 向量，然后 reshape 成 `[..., num_hidden_layers, hidden_size_per_layer_input]`
- 随后在 layer loop 里，第 `i` 层只拿自己的切片 `per_layer_inputs[:, :, i, :]`

也就是说，“每层有自己的 PLE embedding”这句话只能算概念上成立，工程实现上并不是每层单独挂一个 embedding table。

从代码上看，它的意思不是“给整个模型再加一套普通 embedding”，而是：

- 为每一层准备一份逐层输入向量
- 这份逐层输入由两部分组成：
- 一部分来自 token id 的 per-layer embedding lookup
- 一部分来自主 `inputs_embeds` 的 per-layer projection
- 两者相加后，再把第 `i` 层自己的那一片送进该层
- 在层内通过 gate、逐元素乘法和 projection 把这份逐层输入注入主干 hidden states

更具体地说，层内不是简单做 `hidden_states + ple_signal`，而是：

- 先把 `hidden_states` 过一层 `per_layer_input_gate`
- 再过激活函数
- 然后和 `per_layer_input` 做逐元素乘法
- 再投影回主 hidden size
- 最后作为一条额外残差支路加回去

所以看源码时，最好把它理解成“per-layer side input / layer-conditioned modulation path”，这样比单纯看术语更不容易误解。

### 4. Double-Wide MLP 和 MoE 在哪里

还是在 `modular_gemma4.py` 里看文本部分：

- `Gemma4TextMLP`
- `Gemma4TextExperts`
- `Gemma4TextRouter`
- `Gemma4TextDecoderLayer`

你会看到 `enable_moe_block`、router、experts 和主 MLP 是怎么在 decoder layer 里接起来的。也就是说，Gemma 4 的层内结构不是“Attention + 单一路径 MLP”这么朴素。

### 5. 全模态结构怎么读

Gemma 4 在 Transformers 目录里已经拆成了几块：

- 文本：`Gemma4TextModel`
- 视觉：`Gemma4VisionModel`
- 音频：`Gemma4AudioModel`
- 多模态拼接：`Gemma4MultimodalEmbedder`
- 总模型：`Gemma4Model`
- 条件生成入口：`Gemma4ForConditionalGeneration`

如果你关心的是“整机结构”，建议从 `Gemma4Model` 和 `Gemma4ForConditionalGeneration` 往回跳。
如果你关心的是“文本主干到底怎么改的”，就直接盯 `Gemma4TextModel`、`Gemma4TextDecoderLayer`、`Gemma4TextAttention`。

## 建议阅读顺序

推荐按这个顺序看，最省力：

1. `google/gemma_pytorch/gemma/model.py`
2. `transformers/models/gemma4/configuration_gemma4.py`
3. `transformers/models/gemma4/modular_gemma4.py` 里的 `Gemma4TextAttention`
4. `transformers/models/gemma4/modular_gemma4.py` 里的 `Gemma4TextDecoderLayer`
5. `transformers/models/gemma4/modular_gemma4.py` 里的 `Gemma4TextModel`
6. 再回头看 `Gemma4Model`、`Gemma4ForConditionalGeneration`

这个顺序的好处是：

- 先建立“标准 Gemma 骨架”
- 再读 Gemma 4 的配置差异
- 最后再看它是如何把新机制一层层接进模型里的

## 对你这次问题的直接判断

如果你的目标是“验证公众号文章对 Gemma 4 的解释靠不靠谱”，我的建议是：

- 架构命名以源码字段为准
- 机制理解以 `configuration_gemma4.py` 和 `modular_gemma4.py` 为准
- 自动生成文件 `modeling_gemma4.py` 只作为补充，不作为第一入口

二手文章适合快速建立印象，但只要涉及 PLE、Shared KV、层型混排、RoPE 变体这些细节，最后还是得回到源码。
