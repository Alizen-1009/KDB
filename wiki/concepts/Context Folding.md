---
type: concept
topic: 推理服务
sources: 1
updated: 2026-08-14
---

# Context Folding

## 定义

`Context Folding` 是长周期 Agent 主动控制当前上下文窗口的范式：把完整 rollout 中暂时不需要的细节折叠到分支、摘要、外部状态或可查询数据中，只让当前决策所需的信息进入有限上下文。

## 它解决什么问题

- 长时间 Agent rollout 中上下文持续增长，导致成本增加和 `context rot`。
- 工具输出、网页、日志和中间推理占据大量 token，却未必对后续每一步都相关。
- 一次性压缩容易丢失细节，需要允许 Agent 在必要时重新访问原始状态。

## 核心机制

- 分支与返回：Agent 临时进入保留完整历史的分支，返回主线时只带回选定结果。
- 分层摘要：为单步行动或多个行动维护可合并的摘要。
- 外部知识状态：由生成、反思和整理组件持续更新结构化知识库。
- 程序化折叠：[[Recursive Language Model]] 把原始数据留在 Python 环境中，用代码和 sub-LLM 动态选择进入主上下文的信息。

## 与长上下文 Attention 的关系

- 长上下文 Attention 从模型架构和训练层扩大或改善“当前一次 forward 能看多少、看得多好”。
- Context Folding 从 Agent rollout 和 runtime 层决定“这一刻应该把哪些历史交给模型”。
- 两者都在处理过去信息的选择，但作用层次不同；更好的 Attention 可以延缓退化，Context Folding 可以把有效工作跨度进一步推到模型窗口之外。

## 关键权衡

- 折叠可以降低主上下文成本，但检索和摘要策略可能遗漏关键证据。
- 外部状态保留原文有助于恢复细节，但需要索引、权限、版本和生命周期管理。
- 主动管理比固定滑窗更灵活，也更依赖模型是否学会正确分解、查询和回收上下文。
- 把任务委派给子模型可能提高质量或并行度，但会增加总计算和延迟。

## 相关实体

- [[../entities/Prime Intellect]]
- [[../entities/verifiers]]

## 相关来源

- [[../sources/Recursive Language Models the paradigm of 2026]]

## 相关概念

- [[Recursive Language Model]]
- [[LLM Programs]]
- [[Ring Attention]]

## 研究备注

- 后续可分别补 `Context-Folding`、`AgentFold` 与 `Agentic Context Engineering` 原始论文，比较分支返回、分层摘要和知识库演化三类机制。
- 评估 Context Folding 不能只看最终准确率，还应检查证据召回、压缩后可恢复性、总 token、延迟和长周期状态一致性。
