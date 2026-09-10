---
type: entity
entity_type: 模型
topic: 模型架构
sources: 1
updated: 2026-08-27
---

# Qwen3.8-Flash-Next

## 一句话说明

Qwen3.8-Flash-Next 是 Qwen Team 设计的稀疏 MoE 模型，以 GDN/全局注意力混合主干、QSA、四分支 GR 和加速器外 n-gram embedding 联合优化质量、效率与训练稳定性。

## 类型

- 模型

## 核心信息

- 正式模型名为 `Qwen3.8-Flash-Next`，base checkpoint 在评测中写作 `Qwen3.8-Flash-Next-Base`。
- 主干有 `125B` 总参数、每 token 激活 `6B`；此外另有存于加速器外的 `51B` [[../concepts/N-gram Embedding|n-gram embedding]] 参数。不能把 `51B` 含糊写成包含在主干 `125B` 中。
- token mixer 每四层采用 `3` 层 GDN 与 `1` 层全局注意力；在 `256K` CPT 阶段，backbone 与 MTP 的全注意力替换为 [[../concepts/Qwen Sparse Attention|QSA]]。
- residual stream 扩为四分支，并通过 [[../concepts/Gated Residual|GR]] 进行逐 channel gated read 与逐分支标量 write；主训练优化器为 [[../concepts/Muon Optimizer|Muon]]，但并非所有参数都使用 Muon。
- 表 11 中，Qwen3.8-Flash-Next-Base 在 `14` 项预训练评测上全部超过 Qwen3.8-27B-Base；相对 Qwen3.7-Plus-Base（`397B` 总参数、`17B` 激活）胜 `8/14`，最大落后是 MultiPL-E 的 `2.59` 分。
- 论文称相对 Qwen3.7-Plus-Base 约使用 `1/3` 激活参数、`1/3` 训练 tokens 和 `1/9` 训练 FLOPs，但没有给出最终训练的绝对 token 数或 FLOPs。

## 相关概念

- [[../concepts/混合注意力]]
- [[../concepts/线性注意力递归状态]]
- [[../concepts/Qwen Sparse Attention]]
- [[../concepts/Gated Residual]]
- [[../concepts/N-gram Embedding]]
- [[../concepts/Muon Optimizer]]
- [[../concepts/Scaling Laws]]
- [[../concepts/RoPE]]

## 相关来源

- [[../sources/On the Design of Qwen3.8-Next Architecture：Evaluation, Efficiency, and Training Stability]]

## 冲突与备注

- 命名需区分：原始文件名为 `qwen3.8-Next.pdf`，论文标题使用 `Qwen3.8-Next Architecture`，正文模型正式名为 `Qwen3.8-Flash-Next` / `Qwen3.8-Flash-Next-Base`。
- 表 11 的 `397B-A17B` 基线正式名是 `Qwen3.7-Plus-Base`；不能误标为 `Qwen3-Next 397B-A17B`。
- 论文中的 QSA、GR 和稳定性结果来自指定规模、训练阶段或 kernel 设置，不应直接泛化为所有部署条件下的端到端收益。
