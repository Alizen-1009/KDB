# SGLang 与 vLLM 区别截图整理

## 来源信息

- 标题：SGLang 与 vLLM 区别截图整理
- 作者：用户提供截图；LLM 校正整理
- 日期：2026-05-07
- 类型：截图 / 对话整理
- 原始文件：[[../raw/articles/SGLang 与 vLLM 区别截图整理|SGLang 与 vLLM 区别截图整理]]

## 2-3 条核心摘要

- 截图把 `vLLM` 概括为高性能通用推理引擎，把 `SGLang` 概括为面向复杂任务的可编程推理框架；这个方向基本成立，但表述需要加边界。
- 更准确的区分是：`vLLM` 的抽象重心偏 `serving engine`，强调 `PagedAttention`、continuous batching、OpenAI-compatible API 和高并发 serving；`SGLang` 的抽象重心偏 [[LLM Programs]] runtime，强调 DSL、[[RadixAttention]]、structured output 和复杂工作流执行。
- 不能简单说 `vLLM` 只适合简单问答，也不能把 `SGLang` 在复杂任务中的数倍收益外推到所有请求。二者能力边界正在变模糊，选择时应看请求结构、缓存命中、结构化输出和工程生态。

## 值得关注的论断

- `vLLM` vs `SGLang` 更像“serving 抽象重心不同”，不是“一个只能单轮，一个只能 Agent”。
- [[PagedAttention]] 主要解决 KV cache 显存管理和动态 batch 的底层问题；[[RadixAttention]] 主要利用共享前缀和程序结构做运行时 KV cache 复用。
- `SGLang` 的 structured output / constrained decoding 可以提高格式正确性，但不保证语义正确；性能收益依赖具体模板和任务结构。
- 工程选择可以按阶段推进：通用 API 服务优先 `vLLM` 快速上线；复杂多调用、多分支、结构化输出成为瓶颈后，再评估 `SGLang` 或混合路由。

## 关键概念

- [[SGLang 与 vLLM 对比]]
- [[PagedAttention]]
- [[RadixAttention]]
- [[LLM Programs]]
- [[Constrained Decoding]]
- [[KV Cache]]
- [[Prefix Caching]]
- [[Continuous Batching]]

## 相关实体

- [[../entities/vLLM]]
- [[../entities/SGLang]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`SGLang 与 vLLM 对比`、`LLM Programs`、`RadixAttention`、`PagedAttention`
- 会更新哪些实体页：`vLLM`、`SGLang`
- 是否存在冲突：无直接冲突；本次主要把截图中的二分法校正为更稳妥的工程对比口径。

## 待确认

- 截图里关于 `PagedAttention` 碎片率从 `70%+` 降到 `10%-` 的数字需要回到论文原文核对，wiki 中暂不作为独立事实使用。
- `SGLang` “性能提升数倍”的说法需要限定模型、硬件、cache hit、prompt 结构和 benchmark，不应作为普通单轮请求的默认预期。
