# Gemma 4

## 一句话说明

Google DeepMind 在 2026 年发布的开源模型家族，强调参数效率、长上下文能力和全模态支持。

## 类型

- 项目 / 模型家族

## 核心信息

- 提供 E2B、E4B、26B MoE、31B Dense 等多个尺寸。
- 文章强调其能力提升主要来自 `Per-Layer Embeddings`、`Shared KV Cache` 和 `混合注意力 + Dual RoPE` 等架构创新。
- 在长上下文、视觉、多模态与参数效率方面被定位为 Gemma 3 的显著升级。
- 新来源补充了 Gemma 4 的推理加速路线：Google 为 Gemma 4 family 发布 `Multi-Token Prediction (MTP) drafters`，通过轻量草稿模型生成多个候选 token，再由目标模型并行验证；Gemma 4 drafter 还复用目标模型 activation、KV cache，并对 E2B/E4B 使用 efficient embedder / clustering 降低 LM Head 开销。

## 相关概念

- [[Per-Layer Embeddings]]
- [[Shared KV Cache]]
- [[混合注意力]]
- [[Dual RoPE]]
- [[Double-Wide MLP]]
- [[MTP Drafter]]
- [[Speculative Decoding]]

## 相关来源

- [[../sources/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构]]
- [[../sources/Gemma 4：Drafter 详解]]

## 冲突与备注

- 当前页面基于解读文章建立，后续应再 ingest 官方技术报告与实现文档
- `MTP Drafter` 中的部分数值规格（如 E2B drafter 参数量、层数、hidden size）来自截图转述，后续应按官方文档或模型配置核实
