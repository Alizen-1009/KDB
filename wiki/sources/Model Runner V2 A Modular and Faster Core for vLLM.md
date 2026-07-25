---
type: source
source_kind: 文章
topic: 推理服务
updated: 2026-04-23
---

# Model Runner V2 A Modular and Faster Core for vLLM

## 来源信息

- 标题：Model Runner V2: A Modular and Faster Core for vLLM
- 作者：[[vLLM Team]]
- 日期：2026-03-24
- 类型：文章
- 原始文件：`raw/articles/Model Runner V2 A Modular and Faster Core for vLLM.md`

## 2-3 条核心摘要

- 这篇文章介绍了 `vLLM` 的执行核心重构 `Model Runner V2 (MRV2)`。它不是新增一个用户可见 feature，而是围绕 `persistent batching`、异步调度、输入准备和采样的一次底层重新设计，且对外 API 不变。
- MRV2 的三条核心原则是 `modular`、`GPU-native`、`async-first`。其中最关键的变化，是把持久请求状态和每步输入张量解耦，并把大量原先在 CPU 上做的 bookkeeping 和输入准备搬到 GPU 上用 Triton kernel 完成。
- 文章强调 MRV2 的目标不只是“代码更整洁”，而是为 `async scheduling + speculative decoding + multimodal preprocessing` 这类更复杂组合打基础；在文中的特定实验配置下，它分别带来了 `56%` 吞吐提升和 `6.3%` 的 `TPOT` 改善。

## 值得关注的论断

- MRV2 把 `persistent batch` 从“直接作为每步模型输入的布局”改成“稳定 state table + 每步 gather 成输入视图”的模式，本质上是在做状态与视图分离。
- MRV2 把“CPU/GPU 零同步点”当成设计目标，而不是在已有实现上再补异步逻辑；这让它更自然地支持 `async scheduling` 和 `speculative decoding` 的组合。
- 文章里的性能收益是在刻意放大小模型 host-side overhead 的实验设定下测得，更适合解读为“CPU bookkeeping 和同步点确实是瓶颈”，不宜直接外推成通用线上收益。
- 截至 `v0.18.0`，MRV2 仍是实验态，暂不支持 `LoRA`、`logits processors`、部分 `spec decoding` 方法以及 `linear attention models`。

## 关键概念

- [[持久批处理]]
- [[Continuous Batching]]
- [[Speculative Decoding]]
- [[PagedAttention]]

## 相关实体

- [[../entities/vLLM]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`[[Continuous Batching]]`
- 会更新哪些实体页：[[../entities/vLLM]]
- 是否存在冲突：无直接冲突；这篇来源主要把 `vLLM` 的关注点从 `PagedAttention` 扩展到执行核心与运行时架构

## 待确认

- 文中没有展开 `ModelState` 的完整接口演化和各模型家族接入代价，后续若要深入 MRV2，可继续补官方 design doc 或代码实现
