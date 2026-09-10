---
type: concept
topic: 推理服务
sources: 1
updated: 2026-08-14
---

# Recursive Language Model

## 定义

`Recursive Language Model`（RLM）是一种面向超长输入和长周期 Agent 的 scaffold：主模型不直接吞入全部数据，而是通过持久代码环境检索和转换外部数据，并调用拥有独立上下文的 sub-LLM 完成局部任务。

它不是新的 Transformer 层、递归神经网络或 Attention 算法；“recursive”主要指模型能够把问题和部分上下文递归委派给新的模型调用。

## 它解决什么问题

- 避免将整个超长数据集、代码库或工具输出持续塞入主模型上下文。
- 减轻长上下文下的 token 成本、注意力稀释和 `context rot`。
- 把可并行的检索、分类和验证任务分给多个短上下文 sub-LLM。
- 让模型用 Python 的精确计算、检索和字符串操作补充概率式文本生成。

## 核心机制

- 原始大输入保存在主模型上下文之外，只能通过 Python REPL 等程序化接口读取。
- 主模型编写代码搜索、分块、过滤、聚合或转换输入，并控制哪些片段进入当前上下文。
- 主模型可以调用 fresh sub-LLM，把局部数据和任务说明组成新的短上下文；批量接口可以并行执行多个子任务。
- 工具可由 sub-LLM 使用，使网页、文件等高 token 输出停留在子上下文中，主模型只接收提炼后的结果。
- 最终答案可保存在持久变量中反复检查和局部编辑，而不是必须一次性生成。
- Prime Intellect 文章中的实现递归深度固定为 1；任意深度递归仍属于未来工作。

## 与摘要压缩的区别

传统摘要压缩通常用一段短文本替换较早上下文，原始细节如果没有被摘要保留就会丢失。RLM 把原始数据继续保留在外部环境中，后续仍可重新查询，因此减少了不可逆的全局信息丢失。

但这不代表 RLM 不会遗漏信息：主模型的检索代码、分块边界、sub-LLM 提取和最终聚合都可能选择错误。它避免的是“原文被摘要永久覆盖”，而不是自动保证所有相关证据都会被找到。

## 关键权衡

- 主模型上下文可能显著缩短，但 sub-LLM 会增加总 token、API 调用量和端到端延迟。
- 并行委派适合可分解的信息抽取和研究任务；强耦合推理不一定适合拆成多个独立调用。
- scaffold 本身增加了状态管理、[[Sandbox]]、超时、失败重试和结果聚合复杂度。
- 未针对 RLM 训练的模型可能不会主动并行、会编写低效代码，或因额外接口而在简单任务上退化。
- 递归深度增加会扩大计算能力，也会带来成本失控、重复工作、证据追踪和终止判断问题。

## 相关实体

- [[../entities/Prime Intellect]]
- [[../entities/verifiers]]

## 相关来源

- [[../sources/Recursive Language Models the paradigm of 2026]]

## 相关概念

- [[Context Folding]]
- [[LLM Programs]]
- [[Ring Attention]]
- [[Sandbox]]
- [[Model Context Protocol]]

## 研究备注

- 需要补充 RLM 原始论文 `arXiv:2512.24601`，区分 Alex Zhang 原始设计与 Prime Intellect 变体。
- 后续评测应同时报告主模型 token、sub-LLM token、总 token、wall-clock latency、并行度、失败率和答案质量。
- “通过 RL 学会上下文管理可支持持续数周到数月的 Agent”目前属于研究假设，待训练和长期环境实验验证。
