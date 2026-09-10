---
type: entity
entity_type: 模型
topic: 模型架构
sources: 1
updated: 2026-08-27
---

# DeepSeek-V3.2-Exp

## 一句话说明

DeepSeek-V3.2-Exp 是从 DeepSeek-V3.1-Terminus 通过 continued training 加入 DeepSeek Sparse Attention 的实验模型，目标是在尽量保持能力的同时降低长上下文训练与推理成本。

## 类型

- 模型

## 核心信息

- 基础 checkpoint：上下文已经扩展到 `128K` 的 DeepSeek-V3.1-Terminus。
- 论文称相对该基础模型唯一的架构变化是 [[../concepts/DeepSeek Sparse Attention|DSA]]。
- Dense warm-up 使用约 `2.1B` tokens 初始化 lightning indexer；Sparse Training 使用约 `943.7B` tokens 让主模型适应 top-`2048` sparse pattern。
- Post-training 继续使用 specialist distillation 与单阶段 mixed GRPO，并保持与 DeepSeek-V3.1-Terminus 相同的 pipeline、算法和数据。
- 表 1 的 `14` 项评测有升有降，论文判断没有整体显著退化；GPQA、HLE、HMMT 2025 的部分差距与更少 reasoning tokens 有关。
- Checkpoint：[Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)

## 相关概念

- [[../concepts/DeepSeek Sparse Attention]]
- [[../concepts/Qwen Sparse Attention]]
- [[../concepts/MLA]]
- [[../concepts/KV Cache]]
- [[../concepts/RoPE]]
- [[../concepts/Benchmarking]]

## 相关来源

- [[../sources/DeepSeek-V3.2-Exp：Boosting Long-Context Efficiency with DeepSeek Sparse Attention]]

## 冲突与备注

- `Exp` 表示 experimental；作者明确称仍在做更大规模真实场景验证。
- 论文的“唯一架构修改”不表示两模型只相差一次无训练替换：V3.2-Exp 还经历约 `946B` tokens 的两阶段 continued pretraining 及完整 post-training。
- 能力比较受生成 reasoning length 影响；需要在相近输出长度和相同 decoding 设置下解释分数。
