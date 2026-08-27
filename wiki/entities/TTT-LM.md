---
type: entity
entity_type: 项目
topic: 模型架构
sources: 2
updated: 2026-08-26
---

# TTT-LM

## 一句话说明

`TTT-LM` 是论文《Learning to (Learn at Test Time): RNNs with Expressive Hidden States》及其开源实现所代表的序列建模项目，核心是用测试时持续更新的小型 learner 权重作为 RNN 隐藏状态。

## 类型

- 研究项目 / 开源实现

## 核心信息

- 论文：[arXiv:2407.04620](https://arxiv.org/abs/2407.04620)
- 代码：[test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch)
- 主要层变体：`TTT-Linear`、`TTT-MLP`。
- 论文实验对比 Transformer 与 Mamba，覆盖 Pile `2k/8k`、Books3 `1k–32k`，最大 TTT/Transformer 规模约 `1.3B`。
- 项目重点不只是模型公式，还包括 mini-batch TTT、dual form、JAX/TPU 训练实现与 GPU inference kernel 的系统映射；不同后端支持程度需按版本核实。

## 相关概念

- [[../concepts/TTT Layer]]
- [[../concepts/线性注意力递归状态]]
- [[../concepts/KV Cache]]
- [[../concepts/Benchmarking]]

## 相关来源

- [[../sources/【LLM2】Standford TTT模型(Learn at Test Time)]]
- [[../sources/一文通透TTT：Learning to “Learn at Test Time”，让RNN的隐藏层变成可学习的函数，把T]]

## 冲突与备注

- 第一篇二手来源把中等模型规模写为 `250M`，原论文为 `350M`；本页采用原论文口径。
- 现有结果主要是研究 benchmark，不代表 TTT 已在主流 serving runtime 中成为 Transformer/Mamba 的生产替代。
- 仓库当前 API、checkpoint、硬件 kernel 与训练配方需要绑定 commit 后再记录，避免把论文期实现与后续版本混在一起。
