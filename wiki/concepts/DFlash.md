---
type: concept
topic: 投机解码
sources: 1
updated: 2026-08-17
---

# DFlash

## 定义

`DFlash` 是一种基于 **single-step block diffusion drafter** 的 [[并行投机解码]]方法：drafter 以 target 给出的 anchor token 和多个 mask positions 为输入，用一次 forward 并行预测一段候选 token；它不是常规扩散模型那样反复执行多轮 denoising。为弥补小 drafter 的能力不足，它从 target model 的中间 hidden states 提取额外上下文并注入 draft Attention。

## 它解决什么问题

- 避免传统小 draft model 为产生多个候选而执行较长的自回归串行链。
- 利用 target 已计算的中间表示提高并行候选质量和接受长度。
- 让 target model 用一次小 chunk forward 验证多个候选，减少逐 token target steps。

## 核心机制

按当前二手来源的描述：

1. Target model 对当前 prefix forward，产生一个新 token 作为 drafter anchor。
2. 从 target 若干中间层抽取 hidden states，并融合成 `H_ctx`。
3. Drafter 输入 anchor 与若干 mask positions，形成 draft hidden `H_d`；同一 block 内 mask positions 可做双向 Attention，但不会根据本轮已经预测出的 token 再迭代去噪。
4. 每层 Attention 中，Query 只来自 `H_d`；投影后的 target context 与 draft hidden 拼成 `[H_ctx || H_d]`，供 K/V 读取。Target context 的 K/V 可缓存并跨 drafting round 复用。
5. Drafter 在多个 mask 位置一次并行输出 logits，并通过与 target 共享的 LM head 产生候选。
6. Target model 按普通 speculative decoding 规则验证候选前缀；drafter 只负责提案，不直接决定最终输出。

## 为什么 target hidden 有帮助

纯并行填空 drafter 既小于 target，又缺少完整的未来 token 自回归链，长后缀容易失去一致性。Target hidden states 已编码 prefix 的多层语义，作为额外 K/V context 可以提高 drafter 对当前上下文的理解。

这会让 drafter 与 target 深度耦合：需要确定抽取哪些层、怎样融合、如何投影，以及 runtime 如何把这些 activation 交给 drafter。

## 关键权衡

- 并行 proposal 减少 draft 串行延迟，但 target hidden extraction 和 drafter context Attention 会增加计算与显存流量。
- 固定较长草稿块可能产生大量低接受率后缀，draft 和 target verification 都会浪费。
- 块太短则无法充分摊薄 target model 的逐步 decode 成本。
- 最优长度依赖模型、数据、并发、acceptance rate 和 target verify kernel 的 shape 效率。

## 来源 Benchmark

来源在 Qwen3-4B、A800 和其所称 vLLM `0.26.0` 环境中报告：

- `num_speculative_tokens=4`：`448.92 tok/s`，约为 Baseline 的 `1.96x`。
- `num_speculative_tokens=7`：`479.56 tok/s`，约为 Baseline 的 `2.09x`。

这些数字没有完整并发、长度分布、TP 配置和稳定版本信息，只能作为来源实验观察。

## 相关实体

- [[../entities/vLLM]]

## 相关来源

- [[../sources/并行投机解码(DFlashDSpark)的快速理解与vLLM实测]]

## 相关概念

- [[并行投机解码]]
- [[DSpark]]
- [[Speculative Decoding]]
- [[MTP Drafter]]
- [[KV Cache]]

## 研究备注

- 已按 [DFlash 原论文](https://arxiv.org/abs/2602.06036) 核对 single-step block diffusion、block 内双向 Attention、target hidden 持久 K/V 注入与位置加权训练损失；具体 checkpoint / runtime 配置仍应绑定版本。
- 来源称 vLLM 可直接以 `method=dflash` 部署，但正式版本、CLI 与并行支持范围待源码验证。
