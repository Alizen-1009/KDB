---
type: entity
entity_type: 公司
topic: 模型架构
sources: 5
updated: 2026-05-17
---

# DeepSeek-AI

## 一句话说明

当前知识库里与 `Engram`、`mHC` 等论文工作相关的研究团队 / 组织。

## 类型

- 组织 / 研究团队

## 核心信息

- 在当前 vault 已收录的资料里，`DeepSeek-AI` 直接出现在 `Conditional Memory via Scalable Lookup` 与 `mHC` 两篇论文的作者与机构信息中。
- 从知识库现有材料看，它的公开工作不只覆盖模型模块，也覆盖宏观架构拓扑与系统共设计，例如 `Engram` 的可扩展查表记忆，以及 `mHC` 的稳定化 hyper-connections。
- 当前 wiki 对它的记录以论文线索为主，还不是完整的组织画像。
- 新增面试整理补入 DeepSeek-V2/V3 相关的 [[MLA]] 线索：MLA 通过低秩 KV 联合压缩、decoupled RoPE 和矩阵吸收降低 decode 阶段 KV cache 压力，是 DeepSeek 系列推理效率的重要结构设计之一。
- DeepSeek V4 RoPE 解析补入 [[CSA-HCA|CSA/HCA]] 线索：在压缩 attention 与 `MQA/KV 共享` 下，需要处理 RoPE 注入时机、V 路径位置污染和输出逆旋转。
- 陈巍 FlashMLA 解析补入 [[FlashMLA]] 线索：DeepSeek 开源的 MLA decode kernel/backend 面向 Hopper GPU、paged KV cache、变长序列和 Split-KV 优化，属于模型结构与系统 kernel 共设计的一环。

## 相关概念

- [[Conditional Memory]]
- [[mHC]]
- [[Hyper-Connections]]
- [[MLA]]
- [[DP Attention]]
- [[CSA-HCA|CSA/HCA]]
- [[RoPE]]
- [[FlashMLA]]

## 相关来源

- [[../sources/Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models]]
- [[../sources/mHC: Manifold-Constrained Hyper-Connections]]
- [[../sources/MLA与DP Attention面试整理]]
- [[../sources/DeepSeekV4中RoPE设计解析]]
- [[../sources/陈巍：DeepSeek 开源Day（1）-FlashMLA 深入分析（收录于：DeepSeek技术详解系列）]]

## 冲突与备注

- 后续如果补入 `DeepSeek-V3`、`DeepSeek V4`、`DualPipe`、FlashMLA 官方 repo 等更完整原始资料，这个实体页还可以继续扩充
