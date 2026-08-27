---
type: source
source_kind: 文章
topic: 模型架构
updated: 2026-08-26
---

# 【LLM2】Standford TTT模型(Learn at Test Time)

## 来源信息

- 标题：【LLM2】Standford TTT模型(Learn at Test Time)
- 作者：举个栗子
- 日期：2024-11-21
- 类型：文章 / 论文解读
- 原始文件：[[../../raw/articles/【LLM2】Standford TTT模型(Learn at Test Time)|【LLM2】Standford TTT模型(Learn at Test Time)]]
- 原始链接：[知乎](https://zhuanlan.zhihu.com/p/6827298295)
- 论文：[Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620)
- 代码：[test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch)

## 2-3 条核心摘要

- 文章将 [[../concepts/TTT Layer|TTT Layer]] 概括为一种固定大小但更具表达力的 RNN 状态：隐藏状态不再只是向量，而是小模型 `f` 的权重 `W_t`；每个 token 通过自监督内循环更新 `W_t`，再用更新后的模型产生输出。
- TTT 的外循环学习 `θ_K / θ_V / θ_Q` 等视图投影、初始化与学习率，使内循环采用的多视图重建任务最终服务于语言模型目标；[[../entities/TTT-LM|TTT-LM]] 给出 TTT-Linear 与 TTT-MLP 两种实现。
- 为缓解逐 token 梯度更新的串行性与硬件低利用率，文章介绍 mini-batch TTT 与 dual form：前者在接受质量和并行度间折中，后者避免显式物化大量 `d×d` 梯度矩阵，并把核心计算转成 matmul。

## 值得关注的论断

- 来源称，TTT-Linear / TTT-MLP 在长上下文上的相对优势随序列长度增加，在 Pile 8k 与 Books3 32k 设置中优于 Mamba；Transformer 原始 perplexity 仍有竞争力，但其长上下文 FLOP 成本更高。
- 来源称，TTT-Linear 在 8k 上下文时已经比 Transformer 更快，并在训练时间上接近 Mamba；TTT-MLP 虽有更强隐藏状态表达力，但内存 I/O 与 wall-clock 仍是主要系统瓶颈。这些属于论文与二手文章的特定实验结论。
- 文章最后主动指出现有证据边界：最大规模约 `1.3B`、质量指标主要是 perplexity，且方法包含多种初始化、视图和学习率设计，不能据此直接宣称已经替代 Transformer。

## 关键概念

- [[../concepts/TTT Layer|TTT Layer]]
- [[../concepts/线性注意力递归状态|线性注意力递归状态]]
- [[../concepts/KV Cache|KV Cache]]

## 相关实体

- [[../entities/TTT-LM|TTT-LM]]

## 与现有 wiki 的关系

- 创建 [[../concepts/TTT Layer|TTT Layer]]：记录“状态即学习器”、内外循环、并行化、理论联系与 benchmark 边界。
- 创建 [[../entities/TTT-LM|TTT-LM]]：记录论文与开源项目。
- 更新 [[../concepts/线性注意力递归状态|线性注意力递归状态]]：补充特定 TTT-Linear 与线性注意力的等价边界。
- 更新 [[../concepts/KV Cache|KV Cache]]：补充显式历史、固定递归状态与模型化隐藏状态的区别。
- 与现有 wiki 无直接冲突；它扩展了固定大小递归状态的表达形式。

## 待确认

- 文章一处将内循环自监督损失表述为 next-token prediction；原论文 §2.3 / Eq. 4 显示内循环是 `θ_K / θ_V` 多视图重建，next-token prediction 是外循环语言模型目标。收录时以后者为准。
- 文章摘要写模型规模为 `125M / 250M / 760M / 1.3B`；原论文协议是 `125M / 350M / 760M / 1.3B`，Mamba 对应 `130M / 370M / 790M / 1.4B`。`250M` 视为来源误记。
- 标题中的 `Standford` 是原始文章拼写；正式概念使用 `Stanford` 相关口径，不修改 raw 文件名。
