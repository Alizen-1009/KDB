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
- 当前 wiki 最初以 Attention Residuals 论文为锚点；现已通过官方 Kimi K3 技术报告补充 KDA、Stable LatentMoE、Quantile Balancing、MoonEP 与 serving 系统共设计。
- 官方资料确认 Kimi K3 为 2.8T 总参数、约104.2B active parameters、896 routed experts/Top-16、69 KDA + 24 Gated MLA，并对 expert weights 使用 MXFP4、expert activations 使用 MXFP8。

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
- Kimi K3 规格现已有官方仓库与 `raw/papers/k3_tech_report.pdf` 支撑；具体部署显存、吞吐和最优并行拓扑仍需绑定 checkpoint、硬件与 serving engine 版本。
