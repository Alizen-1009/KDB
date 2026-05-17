# DeepSeek V4

## 一句话说明

`DeepSeek V4` 是当前来源中讨论的 DeepSeek 系列模型/架构版本；本页目前只记录与 [[RoPE]]、[[CSA-HCA|CSA/HCA]] 和压缩 attention 相关的线索。

## 类型

- 模型 / 架构版本

## 核心信息

- 来源称 DeepSeek V4 采用 RoPE 位置编码，但由于 attention 结构升级，需要处理压缩 KV 与 `MQA/KV 共享` 下的位置编码问题。
- 来源中提到 `CSA/HCA`、`C128A`、窗口通道 `SWA`、压缩 KV、上采样 Q 和输出 O 等 RoPE 相关位置。
- 在 HCA 的示例中，压缩 KV 的 RoPE 位置采用每 128 段的起始位置 `128 * t`。

## 相关概念

- [[../concepts/RoPE]]
- [[../concepts/MLA]]
- [[../concepts/CSA-HCA|CSA/HCA]]
- [[../concepts/KV Cache]]

## 相关来源

- [[../sources/DeepSeekV4中RoPE设计解析]]

## 冲突与备注

- 当前页面只基于单篇解析文章；`DeepSeek V4` 命名、CSA/HCA 结构和 C128A 细节需按官方论文、repo 或公开实现版本核实后再扩展。
