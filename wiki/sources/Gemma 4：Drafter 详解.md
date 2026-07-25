---
type: source
source_kind: 文章
topic: 投机解码
updated: 2026-05-07
---

# Gemma 4：Drafter 详解

## 来源信息

- 标题：Gemma 4：Drafter 详解
- 作者：小蚁AIGC
- 类型：小红书图文笔记截图 / 技术解读
- 日期：2026-05-06（页面显示“昨天 08:24”，按 2026-05-07 访问日推断）
- 原始文件：[[../../raw/articles/Gemma 4：Drafter 详解]]
- 原始图片：`raw/images/Gemma 4 Drafter 详解/`
- 外部核对：Google 官方文章《Accelerating Gemma 4: faster inference with multi-token prediction drafters》

## 2-3 条核心摘要

1. 该笔记解释了 Gemma 4 的 `Multi-Token Prediction (MTP) drafter`：小型草稿模型先自回归预测多个候选 token，目标模型再一次 forward 并行验证，从而减少逐 token decode 的延迟。
2. Gemma 4 drafter 的重点不是“随便接一个小模型”，而是通过复用目标模型最后层 activation、共享目标模型 KV cache、使用更小 hidden size，把 drafter 的准确率和速度同时拉高。
3. E2B/E4B drafter 还针对 LM Head 瓶颈使用 efficient embedder / clustering：先预测 token cluster，再只在候选 cluster 内计算 token logits，减少完整词表 logits 的计算。

## 值得关注的论断

- MTP 的收益来自“drafter 顺序生成 + target 并行验证”的分工：目标模型仍是最终裁决者，因此正确实现下可以在保持目标模型输出分布的同时降低延迟。
- Gemma 4 drafter 把 target activations 与 drafter token embedding 拼接并下投影，是一种把目标模型已有计算结果注入草稿模型的方式。
- KV cache 共享在这里是 target model 与 drafter 之间的复用；它和 Gemma 4 目标模型内部层间的 [[../concepts/Shared KV Cache]] 相关但不是同一层级。

## 关联概念

- [[../concepts/MTP Drafter]]
- [[../concepts/Speculative Decoding]]
- [[../concepts/KV Cache]]
- [[../concepts/Shared KV Cache]]
- [[../concepts/Per-Layer Embeddings]]

## 关联实体

- [[../entities/Gemma 4]]
- [[../entities/Google DeepMind]]

## 与现有 wiki 的关系

- 补足 [[../entities/Gemma 4]] 在推理加速方向的内容，和既有 `PLE / Shared KV Cache / 混合注意力` 架构条目互补。
- 更新 [[../concepts/Speculative Decoding]]，把 `MTP` 从方案族名称推进到 Gemma 4 的具体实现案例。
- 更新 [[../concepts/KV Cache]] 与 [[../concepts/Shared KV Cache]]，区分模型内部层间共享和 target-drafter 共享。

## 待确认点

- 笔记中的 `76M`、`4 层`、`256 / 1536` 等数值来自截图转述，应按 Gemma 4 官方文档或模型配置文件核实后再作为严肃规格引用。
- `最高 3x` speedup 是官方口径，但实际收益依赖模型尺寸、硬件、batch size、draft length、接受率和推理框架实现。
- 采样模式下的“零质量损失”需要结合具体 speculative sampling 实现验证。
