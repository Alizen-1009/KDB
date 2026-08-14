---
type: source
source_kind: 文章
topic: 推理服务
updated: 2026-08-14
---

# Recursive Language Models: the paradigm of 2026

## 来源信息

- 标题：Recursive Language Models: the paradigm of 2026
- 作者：Sebastian Müller
- 日期：2026-01（文末 BibTeX；具体发布日期待核实）
- 类型：文章
- 原始文件：[[../../raw/articles/Recursive Language Models the paradigm of 2026|Recursive Language Models the paradigm of 2026]]
- 原始链接：[Prime Intellect Blog](https://www.primeintellect.ai/blog/rlm)

## 2-3 条核心摘要

- 文章把长周期 Agent 的瓶颈概括为：上下文越长，单 token 成本越高，模型能力还可能因 `context rot` 下降。相比单纯扩大 Attention 窗口，[[Context Folding]] 让 Agent 主动管理“哪些信息进入当前上下文、哪些信息留在外部环境”。
- [[Recursive Language Model]] 不是新的神经网络层，而是一种 Agent scaffold：超长输入保存在持久 Python REPL 可访问的数据结构中，主模型用代码检索、过滤和转换数据，并把局部任务交给拥有独立上下文的 sub-LLM；主模型只接收经过选择的结果。
- [[../entities/Prime Intellect]] 在 [[../entities/verifiers]] 中实现了实验性 `RLMEnv`。文章在 DeepDive、math-python、Oolong 和 verbatim-copy 上比较标准 LLM、RLM 与带环境提示的 RLM，观察到 RLM 更适合复杂长上下文和高 token 工具调用，但效果依赖任务、模型和提示，并通常增加总 token 与端到端时延。

## 值得关注的论断

- 作者认为，通过强化学习让模型端到端学习上下文管理，将是支持持续数周乃至数月 Agent 任务的重要方向；“the paradigm of 2026”是研究判断，不是实验已经证明的行业事实。
- RLM 将原始数据留在外部 Python 环境，因此不需要用一次不可逆的全局摘要替换原文；但主模型仍可能依赖局部提取或 sub-LLM 摘要，选择错误同样会造成有效信息遗漏。
- 在 Oolong `real` 子集上，文章称 RLM 在约 `1.5M` 字符、估算 `300–400k tokens` 的输入附近明显优于标准 LLM；但实验只抽样 50 个任务，且标准 LLM 的部分长输入被 API 拒绝，因此不能据此形成无条件 benchmark 排名。
- RLM 没有在所有任务上稳定提升：GPT-5-mini 的 math-python 表现下降；DeepDive 需要明确的分解与并行检索提示才改善；不同开放模型对相同环境提示的反应也可能相反。
- 高效长上下文 Attention 与 Context Folding 是互补路线：前者从模型训练和架构层延缓上下文退化，后者在 Agent rollout 层主动组织、检索和委派历史信息。

## 实现机制

- 用户 prompt 直接进入主模型上下文，额外大规模输入只作为 Python 变量暴露，模型必须通过 REPL 才能查看。
- 单次 REPL 输出默认限制为 8192 字符，迫使模型先过滤数据，而不是把全部输入重新打印进上下文。
- `llm_batch()` 支持并行调用多个 fresh sub-LLM，让它们分别读取数据分片、执行研究任务或交叉检查结果。
- Prime Intellect 的变体只把环境工具交给 sub-LLM，以隔离网页、文件等工具产生的大量 token；主 RLM 负责分解、调度与综合。
- 沙箱可安装任意 pip 包；文中 math-python 环境预装 `numpy`、`scipy` 和 `sympy`。
- 最终答案保存在可反复编辑的 `answer["content"]` 中，只有设置 `answer["ready"] = True` 才结束 rollout，因此模型可以检查并局部修订答案。
- 当前实现的递归深度固定为 1：主 RLM 可以调用 sub-LLM，但 sub-LLM 不能继续递归调用下一层。

## 实验观察与边界

| 环境 | RLM 主要用途 | 文章观察 | 关键边界 |
| --- | --- | --- | --- |
| DeepDive | 把搜索问题拆给带网页工具的 sub-LLM | 默认 RLM 未必优于标准 LLM；给出分解、并行和迭代提示后改善 | 工具输出很长，性能高度依赖调度策略 |
| math-python | Python 计算、sub-LLM 验证 | GPT-5-mini 使用 RLM 后更弱且主模型 token 更多 | scaffold 复杂性可能干扰原本已经熟悉 Python tool 的模型 |
| Oolong | 分块长输入、并行抽取、聚合 | 复杂 `real` 数据受益，简单 `synth` 数据可能退化 | 标准 LLM 的超窗 API 拒绝使总体比较偏向 RLM |
| verbatim-copy | 在变量中反复检查和修订精确文本 | GPT-5-mini 多数内容类型提升 | 多轮工具调用提高时延和 token 开销 |

文章还观察到：RLM 经常降低主模型自身的上下文长度，却通过 sub-LLM 使用更多总 token；这更像把推理预算从一个不断膨胀的上下文转移到多个短上下文，而不是无成本压缩。

## 关键概念

- [[Recursive Language Model]]
- [[Context Folding]]
- [[LLM Programs]]

## 相关实体

- [[../entities/Prime Intellect]]
- [[../entities/verifiers]]

## 与现有 wiki 的关系

- 更新概念页：[[Recursive Language Model]]、[[Context Folding]]、[[LLM Programs]]。
- 新增实体页：[[../entities/Prime Intellect]]、[[../entities/verifiers]]。
- 无直接冲突；该来源补充的是 Agent scaffold 与上下文管理层，不应与 Attention kernel、KV Cache 或模型上下文窗口扩展混为一谈。

## 待确认

- raw frontmatter 的 `created: 2026-08-14` 更像采集日期；文末 BibTeX 写的是 2026 年 1 月，具体发布日期待核实。
- 图表未在正文给出完整逐项数值、误差条或统计显著性，当前只保留文章的定性实验观察。
- 实验使用 `verifiers` 的特定实验分支和默认配置，不能直接外推到后续 `RLMEnv` 版本。
- “训练后可解决当前退化”“任意递归深度会进一步提升”等均属于未来工作或作者假设。
