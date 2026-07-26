---
type: source
source_kind: 文章
topic: GPU 编程
updated:
---

# MegaMoE — 让 all-to-all 消失

## 来源信息

- 标题：MegaMoE — 让 all-to-all 消失
- 作者：待确认
- 日期：待确认
- 类型：二手技术章节
- 原始文件：[[../../raw/articles/MegaMoE — 让 all-to-all 消失.md]]
- 原始链接：https://igloomatics.github.io/DeepSeek-V4-book/chapters/ch07.html

## 2-3 条核心摘要

- MegaMoE 将 MoE 的 dispatch、L1 GEMM、activation、L2 GEMM、combine 五阶段组织为融合流水，并把本批命中的 experts 分成多个 wave，使相邻 waves 的通信与计算交错。
- “All-to-All 消失”指通信在理想条件下从关键路径上被计算完全隐藏，实际 dispatch/combine、网络流量和同步并未消失。若计算时间不足以覆盖通信，只能部分隐藏。
- 来源给出 `C/B <= 2*d_ff` 的硬件平衡条件：硬件计算/互联带宽比不高于 workload 的计算/通信比时，通信具备被计算覆盖的条件。该推导依赖 FLOPs、dtype 和通信字节口径，不是所有 MoE 的通用常数。

## 值得关注的论断

- Wave 数越多，流水越深，但每 wave 的 token/expert 工作量更小，会降低 Grouped GEMM 和消息效率并增加 barrier/signal 开销，因此存在最优折中。
- MegaMoE 与 DBO 不在同一层：前者在单 batch 内按 expert wave 做细粒度融合，后者由 runtime 交错两个 microbatch。
- 来源声称 Figure 5 理论加速 `1.92x`、一般推理 `1.50–1.73x`、延迟敏感最高 `1.96x`，但当前未摄入其所称 DeepSeek-V4 一手报告，实验条件待核实。
- Pull/Push 术语和“GPU-initiated put 属于 pull 风格”的表述可能混合了发起者、搬运方向与消费节奏，不能作为标准定义。
- 去掉 SwiGLU sigmoid/exp/div 属于模型架构变化，需要重训和质量验证，不是可直接替换已有模型的纯 kernel 优化。

## 关键概念

- [[../concepts/MegaMoE]]
- [[../concepts/Dual Batch Overlap]]
- [[../concepts/通信-计算重叠]]
- [[../concepts/Expert Parallelism]]
- [[../concepts/Megakernel]]
- [[../concepts/算子融合]]

## 相关实体

- 暂无独立项目实体；当前资料不足以确认公开仓库/API 边界。

## 与现有 wiki 的关系

- 新增 MegaMoE，并与 vLLM DBO、广义 Megakernel 和通信-计算重叠做边界澄清。
- 更新 MoE、Expert Parallelism、Megakernel 与算子融合页面。
- 未发现直接冲突；标题需校正为“隐藏而非消灭 All-to-All”。

## 待确认

- DeepSeek-V4/V4-Pro 一手技术报告、Figure 5 与完整 benchmark 配置。
- MegaMoE 是单 kernel、persistent kernel 还是多个协同 kernels。
- Wave 的真实调度算法、生产取值与 pull/IBGDA 通信语义。
