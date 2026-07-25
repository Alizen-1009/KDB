---
type: concept
topic: 投机解码
sources: 3
updated: 2026-07-08
---

# Multi-Token Prediction

## 定义

`Multi-Token Prediction`，简称 `MTP`，是让模型在同一个上下文位置预测多个未来 token 的方法。它相对于标准 `next-token prediction` 的变化是：不只预测 `t+1`，还额外预测 `t+2 / t+3 / ...`。

## 两种常见语境

- `训练目标`：在共享 transformer trunk 之上增加多个未来 token 的预测头，作为辅助 loss 提供更密集的训练信号。DeepSeek-V3 把 MTP 写成训练目标之一，用于增强模型能力。
- `推理 drafter`：把 MTP head / MTP module 当作草稿生成器，先快速预测多个候选 token，再交给 target model 验证。Gemma 4 的 [[MTP Drafter]] 就是这一类工程实现。

## 常见层结构

### 并行多头式

最简单的 MTP 是共享主干 transformer，只在顶部接 `n` 个独立输出头：

```text
tokens -> shared transformer trunk -> h_t
                                   -> head_1 -> logits for x_{t+1}
                                   -> head_2 -> logits for x_{t+2}
                                   -> head_3 -> logits for x_{t+3}
```

这种结构主要服务训练辅助目标，多个未来 token 的 loss 共同训练同一个 hidden state。

### 顺序模块式

DeepSeek-V3 更接近顺序 MTP module。每个 MTP module 不是单纯一个 linear head，而是包含共享 embedding、共享 output head、一个 transformer block 和 projection，把上一层 hidden state 与未来 token embedding 融合后再预测下一个未来 token：

```text
main model hidden h_t
  + Emb(x_{t+1}) -> projection -> MTP block 1 -> OutHead -> predict x_{t+2}
  + Emb(x_{t+2}) -> projection -> MTP block 2 -> OutHead -> predict x_{t+3}
```

这种结构保留了因果链：预测更远 token 时，会条件化在前一个未来 token 上。

DeepSeek-V3 开源权重中的具体口径：

- `num_nextn_predict_layers = 1`，即开源 V3 权重包含 `1` 个 MTP Module。
- 主模型包含 `model.layers.0` 到 `model.layers.60` 共 `61` 个 Transformer hidden layers；MTP module 作为追加层编号为 `model.layers.61`。
- MTP module 与主模型共享 `model.embed_tokens` 和 `lm_head / shared_head`。
- MTP module 自身包含 `enorm`、`eh_proj`、追加的 `model.layers.61.self_attn & mlp`、`hnorm`；其中 `enorm / hnorm` 是 RMSNorm，`eh_proj` 用于把归一化后的 hidden state 与未来 token embedding 融合/投影。
- 官方权重说明给出的 MTP 规模是 `11.5B` unique parameters，不含共享的 `0.9B` embedding 和 `0.9B` output head；若把共享部分也算入 activation parameter 口径，则 MTP activation parameters 为 `2.4B`。

### 独立 drafter 式

Gemma 4 的 [[MTP Drafter]] 更像专门训练的小型草稿模型。它会利用 target model activation、自己的 token embedding 和共享 KV cache，自回归地产生多个 draft token，再由 target model 验证。

```text
target model activation / KV cache
        + drafter token embedding
        -> small drafter layers
        -> lightweight LM head
        -> draft token sequence
```

这类结构已经明显偏推理系统设计，而不只是训练 loss。

## 与 Speculative Decoding 的关系

MTP 不是 [[Speculative Decoding]] 本身，而是 speculative decoding 里“谁来猜 token”的一种方案。

- [[Speculative Decoding]] 是执行框架：便宜机制先提出候选 token，target model 并行验证，按接受规则输出或回退。
- MTP 是候选生成机制之一：利用多 token 预测头、模块或轻量 drafter 生成候选。
- speculative decoding 还可以不用 MTP，例如小 draft model、`Medusa`、`EAGLE`、`ngram / suffix decoding` 等。
- MTP 也可以只作为训练辅助目标存在；如果推理时丢掉额外预测头，就不构成 speculative decoding。

## 推理流程直觉

1. 主模型按正常自回归路径生成当前 token。
2. MTP 模块基于当前 hidden state / activation 预测后续多个 token，形成 draft。
3. target model 对 draft token 做一次并行验证。
4. 从前往后接受连续通过验证的 token。
5. 遇到第一个未通过的 token 后，丢弃后续 draft，并由 target model 给出替代 token。

## 关键权衡

- 接受率决定收益上限：MTP 猜得越准，一次 target forward 能接受的 token 越多。
- draft 长度不是越长越好：更长 draft 提高理论并行度，也增加错误候选、KV cache 预留和回退管理成本。
- 训练耦合更强：相比外接独立小 draft model，MTP 往往需要模型训练或结构上专门支持。
- 采样质量依赖验证规则：贪心解码下较直观；采样场景需要严格的 speculative sampling / rejection sampling 才能保持 target 分布。

## 面试解释

> MTP 可以理解成“让模型顺手多预测几个未来 token”。如果这些预测头只用于训练，它是一个辅助训练目标；如果推理时拿这些预测结果当草稿，再让大模型并行验证，它就成了 speculative decoding 的 drafter。也就是说，speculative decoding 是“猜测-验证”的解码框架，MTP 是其中一种高耦合的猜测器。

## 相关来源

- [[../sources/LLM提速利器：投机推理的原理与常见方案]]
- [[../sources/Gemma 4：Drafter 详解]]
- [[../sources/RTP-LLM]]

## 相关概念

- [[Speculative Decoding]]
- [[MTP Drafter]]
- [[KV Cache]]
