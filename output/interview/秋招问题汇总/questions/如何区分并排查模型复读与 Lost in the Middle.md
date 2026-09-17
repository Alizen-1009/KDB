# 如何区分并排查模型复读与 Lost in the Middle？

- 整理状态：已整理
- 题目出处：用户在本次对话中提供的面试题截图

## 30 秒回答

复读是自回归生成进入 token/句式的正反馈循环；Lost in the Middle 是信息虽然仍在窗口内，但放在长上下文中间时利用率下降。排查时先固定 seed 和 greedy，记录逐 token logits、entropy 与 EOS；再检查最终 tokenized prompt、截断、重复 RAG 文档、chat template、position IDs、RoPE、mask 和 KV Cache；最后把同一证据放在开头/中间/结尾做位置扫描。前者可通过采样、重复惩罚和停止条件缓解，后者更依赖检索去重、重排、压缩与分层处理上下文。

## 常见误区

- 服务端 retry 或把累计流式文本反复 append 也会造成“假复读”。
- repetition penalty 只能压制表象，不能修复中间证据利用失败，并可能破坏代码、JSON 和公式。

## 参考来源与待核实

- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- kernel/cache 排查项需要用同输入比较 eager 与优化 backend、cache on/off 的 logits，不能只看最终文本。

## 所属题单

- [[../sets/近期对话面试问题回顾|近期对话面试问题回顾]]
- [[../sets/模型架构|模型架构]]
- [[../README|秋招问题汇总]]
