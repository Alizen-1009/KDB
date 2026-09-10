---
type: entity
entity_type: 公司
topic: 模型架构
sources: 1
updated: 2026-08-27
---

# Z.ai

## 一句话说明

Z.ai 是 [[GLM-5 系列]]官方模型仓库、文档、博客与 Hugging Face checkpoint config 的发布方。

## 类型

- 公司

## 核心信息

- 本来源通过 Z.ai 的 GLM-5 仓库 README、GLM-5.2 / 5.3 / 5.3-Flash 文档与博客核对版本关系。
- GLM-5、GLM-5.1、GLM-5.2 与 [[GLM-5.3-Flash]] 的官方 Hugging Face `config.json` 是本文核对架构字段的主要依据。
- Z.ai 官方说明 GLM-5.3 使用与 GLM-5.2 相同的 Base、改进来自 post-training；由于没有独立公开文本 checkpoint config，知识库不把这一点写成直接核对了 5.3 config。

## 相关概念

- [[../concepts/DeepSeek Sparse Attention]]
- [[../concepts/IndexShare]]
- [[../concepts/KDA]]
- [[../concepts/混合注意力]]
- [[../concepts/mHC]]

## 相关来源

- [[../sources/glm-5-architecture-evolution]]

## 冲突与备注

- 本页只记录与该来源直接相关的官方模型和文档证据，不扩展公司历史、组织结构或其它产品信息。
- 模型性能与 FLOPs 数字若后续引用，仍需保留官方声称、checkpoint / context / benchmark 条件与是否本地复测的边界。
