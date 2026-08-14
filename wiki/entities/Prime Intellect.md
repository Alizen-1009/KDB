---
type: entity
entity_type: 公司
topic: 推理服务
sources: 1
updated: 2026-08-14
---

# Prime Intellect

## 一句话说明

Prime Intellect 是从事开放模型、强化学习基础设施与 Agent 环境研究的公司；本文来源介绍了其对 Recursive Language Model 与 Context Folding 的研究方向。

## 类型

- 公司

## 核心信息

- 文章将 [[Recursive Language Model]] 视为管理超长输入和长周期 Agent 上下文的重要 scaffold。
- Prime Intellect 在 [[verifiers]] 中实现实验性 `RLMEnv`，并计划结合 `prime-rl` 训练模型学习上下文管理。
- 其 RLM 变体支持持久 Python REPL、并行 sub-LLM、子模型工具访问、沙箱包安装和可迭代答案变量。
- 文章当前实验主要是调用既有模型 API 的 scaffold ablation，不等同于已经完成 RLM 强化学习训练。

## 相关概念

- [[Recursive Language Model]]
- [[Context Folding]]
- [[LLM Programs]]

## 相关来源

- [[../sources/Recursive Language Models the paradigm of 2026]]

## 冲突与备注

- “RLM 将成为 2026 范式”是 Prime Intellect 的研究判断，不应当作行业共识。
- 当前页面仅根据该博客记录；公司、项目和训练计划的后续状态需结合官方 repo 与文档更新。
