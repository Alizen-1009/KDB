---
type: entity
entity_type: 组织
topic: GPU 编程
sources: 1
updated: 2026-08-21
---

# 阿里云 PAI 团队

## 一句话说明

`阿里云 PAI 团队` 是 [[阿里巴巴]] 旗下阿里云人工智能平台相关研发团队；当前知识页聚焦其 FlashAttention-4 大 head dimension kernel 工作。

## 类型

- 组织 / 工程研发团队

## 核心信息

- 来源称团队为 Qwen3.5 等模型的 `head_dim=256` 训练需求设计了专用 FA4 Forward/Backward kernel。
- 方案围绕 [[Tensor Memory]] 容量、2-CTA/DSMEM、`tcgen05.mma`、warp specialization 与异步 pipeline 重构数据流。
- 来源称实现通过 Dao-AILab/flash-attention PR #2412 合入官方仓库，并已支撑千卡规模训练；需要以 merge commit、发布版本和训练配置核实。
- 文中 benchmark 的 L20A/L20C 名称和峰值吞吐口径不一致，不能把其数字视为跨版本稳定结果。

## 相关概念

- [[Tensor Memory]]
- [[FlashAttention]]
- [[GPU执行模型]]
- [[Tiling]]
- [[重计算]]

## 相关来源

- [[../sources/PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化]]

## 冲突与备注

- 原始资料 frontmatter 作者为“阿里云大数据AI”，正文称“阿里云 PAI 团队”；本页按正文团队名归档，两者是否对应同一正式组织/内容账号待核实。
- “合入官方仓库”“千卡规模训练”和性能数字均保留为来源声称，需要仓库与训练记录佐证。
