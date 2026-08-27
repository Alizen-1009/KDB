---
type: entity
entity_type: 模型
topic: 模型架构
sources: 2
updated: 2026-05-17
---

# DeepSeek V4

## 一句话说明

`DeepSeek V4` 是采用 [[CSA-HCA|CSA/HCA]] 混合长程压缩注意力、共享 `K=V` MQA 与 [[mHC]] 残差拓扑的 DeepSeek 系列模型架构；当前资料还包含 [[DSpark]] 在 Flash / Pro 预览版 serving 中的二手部署线索。

## 类型

- 模型 / 架构版本

## 核心信息

- 当前官方 Transformers 文档把长程层分为两类：`CSA` 以默认 `m=4` 的 overlapping compressor 生成低压缩条目，再由 Lightning Indexer 选 top-k；`HCA` 以默认 `m'=128` 的 non-overlapping compressor 生成重压缩条目，不设 Indexer并对全部可见条目做 dense attention。
- CSA/HCA 共用共享 `K=V` 的 MQA 骨架与 local sliding-window `K=V` 支路；前者侧重较细压缩后的稀疏选择，后者侧重更粗压缩后的全量读取。
- 模型还采用 [[mHC]] 管理多路 residual streams；这属于跨层残差拓扑，不是 attention 或 KV 压缩机制。
- 早期二手来源提到 `C128A`、窗口通道 `SWA`、压缩 KV、上采样 Q 和输出 O 等 RoPE 路径，并以 `128*t` 解释压缩块位置。正式 CSA/HCA 定义现以官方文档为准，旧推导只保留为版本相关实现线索。
- [[../sources/DSpark：结合半自回归生成与置信度调度的投机解码技术]] 称 DSpark 已部署于 DeepSeek-V4 Flash / Pro 预览版线上 serving；相同系统吞吐下，单用户生成速度分别提升 `60%–85%` 与 `57%–78%`。该结论属于二手论文解读中的来源声称，缺少完整硬件、流量与 baseline 配置。

## 相关概念

- [[../concepts/RoPE]]
- [[../concepts/MLA]]
- [[../concepts/CSA-HCA|CSA/HCA]]
- [[../concepts/mHC]]
- [[../concepts/KV Cache]]
- [[../concepts/DSpark]]
- [[../concepts/Speculative Decoding]]

## 相关来源

- [[../sources/DeepSeekV4中RoPE设计解析]]
- [[../sources/DSpark：结合半自回归生成与置信度调度的投机解码技术]]

## 官方资料

- [DeepSeek V4 Transformers 文档](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)
- [先进大模型架构知识图谱](../../output/reports/先进大模型架构知识图谱.html)

## 冲突与备注

- CSA/HCA 的分离定义与默认压缩率已由当前官方 Transformers 文档确认；具体层调度、tensor shape、dtype 和 checkpoint 配置仍应绑定版本核实。
- 早期 `C128A`/RoPE 解析与当前官方结构说明可能来自不同版本或抽象层，不能无条件合并成同一实现。
- DSpark 的 V4-Flash / V4-Pro 部署、SLA 与速度提升目前也来自论文解读文章；在官方生产报告补齐之前，不视为可外推 benchmark。
