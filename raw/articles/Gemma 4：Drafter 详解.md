# Gemma 4：Drafter 详解

## 来源信息

- 来源：小红书笔记截图
- 标题：Gemma 4：Drafter 详解
- 作者：小蚁AIGC
- 原文指向：用户提供的小红书链接，网页端只可读取首段与标题
- 截图数量：19 张
- 整理日期：2026-05-07
- 外部核对：Google 官方文章《Accelerating Gemma 4: faster inference with multi-token prediction drafters》

## 原始截图

原始截图已复制到 `raw/images/Gemma 4 Drafter 详解/`：

- `Gemma 4：Drafter 详解_1_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_2_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_3_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_4_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_5_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_6_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_7_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_8_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_9_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_10_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_11_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_12_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_13_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_14_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_15_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_16_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_17_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_18_小蚁AIGC_来自小红书网页版.jpg`
- `Gemma 4：Drafter 详解_19_小蚁AIGC_来自小红书网页版.jpg`

## 整理摘要

这组截图解释了 Google Gemma 4 的 `Multi-Token Prediction (MTP) drafter`。它把大模型逐 token 自回归生成的一部分工作拆成两段：先由更小的草稿模型快速生成多个候选 token，再由目标模型一次 forward 并行验证这些候选。若候选连续通过验证，就可以一次接受多个 token；若遇到第一个被拒绝的 token，后续候选会被丢弃，目标模型会用自身 forward pass 产出的 token 替代。

笔记强调 Gemma 4 drafter 不是普通外接小模型，而是为 Gemma 4 做了架构级适配：

- 使用目标模型最后层的 activation 作为 drafter 的输入信号之一。
- drafter 可以通过 cross-attention 复用目标模型已经计算好的 KV cache。
- E2B/E4B drafter 使用更小 hidden size，示例中从目标模型的 `1536` 维压缩到 drafter 内部 `256` 维。
- E2B/E4B 的 LM Head 使用 efficient embedder / clustering 思路：先预测 token cluster，再只在候选 cluster 内计算 token logits，降低完整 vocab logits 的开销。

## 关键机制

### 推测解码

标准自回归 LLM 每次 forward 主要产出一个新 token。对于明显的续写片段，这会浪费目标模型的显存带宽和计算时间。MTP drafter 用小模型提前生成多个草稿 token，再让目标模型并行验证，从而把一部分顺序生成开销转化为并行验证。

### 多 token 验证

目标模型验证草稿 token 时，不是逐个运行多次 forward，而是在一次 forward 中检查一串候选。接受规则是从前往后连续接受：第一个被拒绝 token 之前的候选可以保留，被拒绝 token 及其后的候选丢弃；目标模型会给出替代 token，并从这个新位置继续下一轮 drafter 生成。

### 目标模型 activation 注入

Gemma 4 drafter 会复用目标模型最后层 activation。笔记中的 E2B 示例把目标模型最后 activation 与 drafter token embedding 拼接，两者各有 `1536` 个数值，再下投影到 drafter 内部的 `256` 维。这让 drafter 的第一轮生成可以利用目标模型已经完成的大量计算，而不是从零构建上下文表示。

第一轮之后，drafter 以自回归方式使用自己上一轮生成的激活值继续预测后续 token。由于 drafter 内部维度更小，这部分计算速度显著快于目标模型。

### KV Cache 共享

笔记区分了 local/global attention 层。drafter 不需要完整处理 prompt 来建立自己的 KV cache，而是通过 cross-attention 复用目标模型已经算好的 KV cache。对于 local attention 层，drafter 可以复用目标模型最后计算出的 local KV cache；如果目标模型最后一层是 global 层，则 global KV cache 也可以被 drafter 的 global attention 层复用。

这与 Gemma 4 原有的 `Shared KV Cache` 有关，但这里关注的是 target model 与 drafter 之间的复用，而不是单个目标模型内部层间共享。

### Efficient Embedder / Clustering

E2B/E4B drafter 的最后 LM Head 可能成为瓶颈，因为完整词表大小可达数十万 token。笔记给出的做法是对 token embedding 做聚类，得到多个语义相近 token 组成的 cluster，并为每个 cluster 生成 embedding 表示。

推理时，模型先计算 cluster logits，选择最可能包含正确 token 的 cluster；随后只在被选中的 cluster 内计算 token logits。这样避免对整个词表都做完整 logits 计算，使 LM Head 更轻量。

## 与官方信息的核对

Google 官方文章确认了以下要点：

- Gemma 4 family 发布了 MTP drafters，用 specialized speculative decoding architecture 加速推理。
- 官方称最高可达 `3x` speedup，并强调不降低 output quality 或 reasoning logic。
- drafter 会使用目标模型 activations，并共享目标模型 KV cache。
- E2B/E4B 的 final logit calculation 是瓶颈，因此使用 efficient clustering technique in the embedder。
- 权重可在 Hugging Face / Kaggle 获取，支持 LiteRT-LM、MLX、Hugging Face Transformers、vLLM、SGLang、Ollama 等路径。

## 待核实与边界

- `目标模型最终可能接受任意数量草稿 token` 这类说法需要加上实现边界：实际会有 draft length / max drafted tokens 上限。
- `零质量损失` 依赖严格的 speculative decoding 验证规则。贪心解码下较直观；采样场景需要检查实现是否保持目标分布。
- `drafter = MTP Head` 是 Gemma 4 这篇笔记里的可用说法，但不能泛化到所有 speculative decoding；很多系统的 drafter 是独立小模型。
- Efficient Embedder / clustering 主要指 E2B/E4B edge models 的优化，不应直接套到所有 Gemma 4 尺寸。

## 关联概念

- [[Speculative Decoding]]
- [[MTP Drafter]]
- [[KV Cache]]
- [[Shared KV Cache]]
- [[Per-Layer Embeddings]]

## 关联实体

- [[Gemma 4]]
- [[Google DeepMind]]
