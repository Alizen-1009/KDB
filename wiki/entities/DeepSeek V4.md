---
type: entity
entity_type: 模型
topic: 模型架构
sources: 2
updated: 2026-05-17
---

# DeepSeek V4

## 一句话说明

`DeepSeek V4` 是当前来源中讨论的 DeepSeek 系列模型/架构版本；现有资料覆盖 [[RoPE]]、[[CSA-HCA|CSA/HCA]]、压缩 attention，以及 [[DSpark]] 在 Flash / Pro 预览版 serving 中的部署线索。

## 类型

- 模型 / 架构版本

## 核心信息

- 来源称 DeepSeek V4 采用 RoPE 位置编码，但由于 attention 结构升级，需要处理压缩 KV 与 `MQA/KV 共享` 下的位置编码问题。
- 来源中提到 `CSA/HCA`、`C128A`、窗口通道 `SWA`、压缩 KV、上采样 Q 和输出 O 等 RoPE 相关位置。
- 在 HCA 的示例中，压缩 KV 的 RoPE 位置采用每 128 段的起始位置 `128 * t`。
- [[../sources/DSpark：结合半自回归生成与置信度调度的投机解码技术]] 称 DSpark 已部署于 DeepSeek-V4 Flash / Pro 预览版线上 serving；相同系统吞吐下，单用户生成速度分别提升 `60%–85%` 与 `57%–78%`。该结论属于二手论文解读中的来源声称，缺少完整硬件、流量与 baseline 配置。

## 相关概念

- [[../concepts/RoPE]]
- [[../concepts/MLA]]
- [[../concepts/CSA-HCA|CSA/HCA]]
- [[../concepts/KV Cache]]
- [[../concepts/DSpark]]
- [[../concepts/Speculative Decoding]]

## 相关来源

- [[../sources/DeepSeekV4中RoPE设计解析]]
- [[../sources/DSpark：结合半自回归生成与置信度调度的投机解码技术]]

## 冲突与备注

- `DeepSeek V4` 命名、CSA/HCA 结构和 C128A 细节需按官方论文、repo 或公开实现版本核实后再扩展。
- DSpark 的 V4-Flash / V4-Pro 部署、SLA 与速度提升目前也来自论文解读文章；在官方生产报告补齐之前，不视为可外推 benchmark。
