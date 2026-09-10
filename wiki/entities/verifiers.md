---
type: entity
entity_type: 项目
topic: 推理服务
sources: 1
updated: 2026-08-14
---

# verifiers

## 一句话说明

`verifiers` 是 Prime Intellect 用于构建、运行和评估模型环境的开源项目；文章所述 Recursive Language Model 以实验性 `RLMEnv` 集成其中。

## 类型

- 项目 / 开源代码仓库

## 核心信息

- `RLMEnv` 把持久 Python REPL、外部输入数据、sub-LLM 调用和最终答案变量封装成可用于不同环境的 scaffold。
- 文章使用 `verifiers` 的实验分支在 DeepDive、math-python、Oolong 和 verbatim-copy 环境中比较标准 LLM、RLM 与 RLM+tips。
- `llm_batch()` 用于批量并行子模型调用；用户提供的环境工具只暴露给 sub-LLM，以隔离高 token 工具输出。
- 文章称该实现可以与 `prime-rl` 配合训练，但本文实验仍以 API 模型的 inference-time ablation 为主。

## 相关概念

- [[Recursive Language Model]]
- [[Context Folding]]
- [[LLM Programs]]

## 相关来源

- [[../sources/Recursive Language Models the paradigm of 2026]]

## 冲突与备注

- 文章中的复现实验命令指向 `sebastian/experiment/rlm` 分支，未必等同于后续 `main` 分支行为。
- 当前尚未 ingest repo 源码；具体 API、沙箱隔离和训练接口应以对应版本源码为准。
