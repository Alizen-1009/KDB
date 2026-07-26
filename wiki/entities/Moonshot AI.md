---
type: entity
entity_type: 公司
topic: 模型架构
sources: 3
updated: 2026-05-06
---

# Moonshot AI

## 一句话说明

`Kimi` 系列模型背后的研究团队 / 组织，在当前知识库里主要通过 `Attention Residuals` 这条论文线进入。

## 类型

- 组织 / 研究团队

## 核心信息

- 在当前 vault 已收录资料中，`Moonshot AI` 最直接对应的是 `Kimi Team` 发布的 `Attention Residuals` 论文与官方仓库。
- 从这篇工作看，它不仅关注模型模块本身，也很重视训练栈可落地性，例如 block 级残差聚合、cache-based pipeline communication 和 two-phase computation。
- 当前 wiki 对它的记录仍以单篇论文为锚点，不构成完整组织画像。
- 新增二手来源将 Kimi K3 与 Stable LatentMoE、KDA/Attention Residuals、Quantile Balancing 和 MXFP4/MXFP8 联系起来；这些具体规格和训练机制尚未由官方一手资料核实，只作为后续研究线索。

## 相关概念

- [[Attention Residuals]]
- [[PreNorm Dilution]]
- [[LatentMoE]]

## 相关来源

- [[../sources/Attention Residuals]]
- [[../sources/Kimi新作《Attention Residuals》：对Transformer中残差结构的调整]]
- [[../sources/2026 年MoE 架构正在发生一次关键变化]]

## 冲突与备注

- 后续如果补 `Kimi Linear`、`Mooncake` 或更多 Moonshot AI 原始资料，这个实体页可以继续扩展。
- Kimi K3 的 2.8T、896 experts、Top-16 等图片规格暂不视为已确认事实；应等待官方技术博客、报告或代码。
