---
type: concept
topic: 推理服务
sources: 1
updated: 2026-05-07
---

# Constrained Decoding

## 定义

`Constrained Decoding` 是在生成过程中限制模型只能输出满足特定格式或语法的 token 序列，例如 JSON、正则表达式、枚举值或结构化字段。

## 它解决什么问题

- 提高结构化输出的可解析性，减少 JSON 破损、字段缺失和格式漂移。
- 让 LLM 输出能更稳定地接入现有软件系统、工具调用或评测流程。
- 在 `LLM Programs` 中，把“输出必须符合某种 schema”从 prompt 约束变成 runtime 约束。

## 核心机制

- 常见做法是把正则表达式或 grammar 转成有限状态机 `FSM`。
- 解码时维护当前 FSM 状态，只允许能转移到合法状态的 token，其他 token 的概率被屏蔽。
- `SGLang` 的优化点是压缩 FSM 中相邻的单一转换边，使固定常量片段可以在一次前向中处理多个 token，而不是逐 token 解码。
- 这种压缩 FSM runtime 可以在输出模板中存在长固定片段时减少解码轮次。

## 关键权衡

- 可以显著提升结构化输出稳定性，但会增加 tokenizer、FSM 和模型 runner 的集成复杂度。
- 如果约束过窄，模型可能被迫输出低质量或语义不自然的内容。
- 多 token 快速路径依赖输出模板中是否存在可压缩的确定性片段；自由文本部分仍需要正常解码。

## 相关实体

- [[../entities/SGLang]]

## 相关来源

- [[../sources/SGLang：LLM推理引擎发展新方向]]

## 相关概念

- [[LLM Programs]]
- [[Speculative Decoding]]

## 研究备注

- 文章提到 Microsoft `Guidance` 是较早相关工作之一；后续可补 Guidance、Outlines、LMQL、llama.cpp grammar 等实现来比较 FSM/grammar 约束的工程差异。
- 需要进一步核实 SGLang 压缩 FSM 对不同 tokenizer、正则表达式复杂度和 JSON schema 的支持边界。
