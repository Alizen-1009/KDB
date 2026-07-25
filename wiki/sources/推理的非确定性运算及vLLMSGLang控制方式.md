---
type: source
source_kind: 文章
topic: 推理服务
updated: 2026-05-06
---

# 推理的非确定性运算及vLLMSGLang控制方式

## 来源信息

- 标题：推理的非确定性运算及vLLM/SGLang控制方式
- 作者：kaiyuan新知答主
- 日期：2026-03-20
- 类型：文章
- 原始文件：[[../../raw/articles/推理的非确定性运算及vLLMSGLang控制方式|推理的非确定性运算及vLLMSGLang控制方式]]

## 2-3 条核心摘要

- 文章把大模型推理中的“不一致输出”拆成两类：`采样随机性` 与 `计算非确定性`。前者主要由 `temperature / top-k / top-p` 等随机采样引入，可通过固定 `seed` 或使用贪婪采样控制；后者则即使固定随机种子也可能出现。
- 文章认为推理框架中的计算非确定性，本质上与浮点数非结合性、GPU kernel 的 tiling / reduction 顺序，以及 `dynamic batching / continuous batching / multiprocessing` 带来的 batch 形态变化共同相关。
- 在框架层面，`vLLM` 与 `SGLang` 都提供了“确定性推理”开关，但实现思路并不只是固定随机种子，而是进一步约束调度形态、kernel 选择与规约算法；代价通常是性能下降和支持范围受限。

## 值得关注的论断

- “关闭随机采样”并不等于“推理可复现”，因为真实系统里的非确定性还可能来自计算顺序变化。
- `Continuous Batching` 不只是吞吐优化手段，也会成为推理不可复现的重要来源之一。
- `vLLM` 的 `Batch Invariance` 目标是让不同 batch 形态下仍保持一致数值行为，但文中明确说明它仍处于 `beta` 状态，且会禁用部分优化。
- `SGLang` 的确定性推理不仅支持固定一个种子，还支持按请求传入不同 `sampling_seed`，这对 `RL rollout` 一类“既要可复现、又要多样性”的场景很实用。

## 关键概念

- [[确定性推理]]
- [[Continuous Batching]]
- [[混合精度训练与推理]]

## 相关实体

- [[../entities/vLLM]]
- [[../entities/SGLang]]

## 与现有 wiki 的关系

- 会创建哪些概念页：`确定性推理`
- 会更新哪些概念页：`Continuous Batching`
- 会更新哪些实体页：`vLLM`、`SGLang`
- 是否存在冲突：未发现与现有 wiki 的直接冲突；这篇来源主要补上“推理系统为什么不一定可复现”以及“框架如何工程化地约束非确定性”这条视角

## 待确认

- 文中关于 `vLLM Batch Invariance`、`SGLang deterministic inference` 的支持范围、硬件要求和后端矩阵主要来自二手整理；后续若要做更严格对照，仍建议补官方文档或代码说明。
