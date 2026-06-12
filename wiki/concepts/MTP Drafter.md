# MTP Drafter

## 定义

`MTP Drafter` 是 Gemma 4 中用于 [[Multi-Token Prediction]] 的草稿模型机制。它通过轻量 drafter 先生成多个候选 token，再由目标模型并行验证候选，从而加速自回归推理。

## 它解决什么问题

- 标准 decode 每次 forward 通常只生成一个 token，延迟受 memory bandwidth 和参数搬运限制。
- 对常见、容易预测的续写，目标模型逐 token 生成会浪费大量重复开销。
- 小型 drafter 可以利用目标模型空闲计算资源先猜多个 token，而目标模型更适合一次并行验证这些候选。

## 核心流程

1. 目标模型生成当前 token，并产生中间 hidden states / activations。
2. drafter 使用目标模型 activation、自己的 token embedding 和共享 KV cache，快速自回归生成若干草稿 token。
3. 目标模型在一次 forward 中并行验证草稿 token。
4. 从前往后连续接受被目标模型认可的 token。
5. 遇到第一个拒绝 token 后，后续草稿 token 被忽略，目标模型用自身 forward pass 选择替代 token。

## Gemma 4 的实现要点

- `Target activations`：drafter 可以利用目标模型最后层 activation，提高草稿 token 质量。
- `Project down / up`：E2B 示例中，目标模型 activation 与 token embedding 各为 `1536` 维，drafter 内部压缩到 `256` 维计算，再上投影回目标维度附近。
- `KV cache sharing`：drafter 不必完整重跑 prompt，而是通过 cross-attention 复用目标模型已经计算好的 KV cache。
- `Efficient embedder`：E2B/E4B 对 LM Head 使用 clustering/sparse decoding 思路，先算 cluster logits，再在选中 cluster 内算 token logits。
- `Small drafter`：例如截图提到 E2B 对应 drafter 约 `76M` 参数、`4` 层；这些规格需按官方配置文件核实。

## 与 Speculative Decoding 的关系

`MTP Drafter` 是 [[Multi-Token Prediction]] 进入 [[Speculative Decoding]] 的一种具体实现。它和传统 draft model 路线相同点在于都采用“候选生成 + 目标模型验证”；不同点在于 Gemma 4 drafter 与目标模型结合更紧，会复用目标模型 activation 和 KV cache，并针对端侧模型优化 LM Head。

## 关键权衡

- 接受率越高，收益越明显；如果候选经常早早被拒绝，drafter 的额外计算可能抵消收益。
- draft length 越长，理论并行验证收益越大，但错误候选和 KV cache 回滚/管理成本也更高。
- 对小模型/端侧模型，LM Head 可能成为瓶颈，因此 clustering/sparse logits 有实际价值。
- 对 MoE 或大模型，收益还会受 batch size、路由开销、硬件内存带宽和框架支持影响。

## 面试解释

可以这样讲：

> Gemma 4 的 MTP drafter 本质上是 speculative decoding。目标模型先正常生成一个 token，同时留下 activation；drafter 利用这些 activation、共享的 KV cache 和自己的轻量网络，快速预测多个后续 token。目标模型随后一次 forward 并行验证这些 token，连续验证通过的 token 直接接受，第一个失败之后的候选丢弃，并由目标模型给出替代 token。Gemma 4 的特殊点是 drafter 不是完全独立的小模型，而是和 target model 深度耦合：复用 activation、共享 KV cache，并在 E2B/E4B 上用 clustered LM Head 降低 logits 计算。

## 相关实体

- [[../entities/Gemma 4]]
- [[../entities/Google DeepMind]]

## 相关来源

- [[../sources/Gemma 4：Drafter 详解]]
- [[../sources/LLM提速利器：投机推理的原理与常见方案]]

## 相关概念

- [[Speculative Decoding]]
- [[Multi-Token Prediction]]
- [[KV Cache]]
- [[Shared KV Cache]]
- [[Per-Layer Embeddings]]

## 研究备注

- 后续应补官方 documentation / Hugging Face config，确认不同 Gemma 4 尺寸对应 drafter 的层数、hidden size、draft length 和 sparse LM Head 实现细节。
- `MTP Drafter` 与 `Medusa / EAGLE / ngram speculative decoding` 的差异值得另写横向对比。
