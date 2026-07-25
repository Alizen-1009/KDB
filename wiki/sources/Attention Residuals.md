---
type: source
source_kind: 论文
topic: 模型架构
updated: 2026-05-06
---

# Attention Residuals

## 来源信息

- 标题：Attention Residuals
- 作者：Kimi Team
- 日期：2026-03-16（arXiv v1）
- 类型：论文 / 技术报告
- 原始文件：`raw/papers/2603.15031v1.pdf`
- 相关资料：MoonshotAI 官方仓库 `Attention-Residuals`；知乎解读 [[Kimi新作《Attention Residuals》：对Transformer中残差结构的调整]]

## 2-3 条核心摘要

- 论文把标准 PreNorm 残差重新解释为深度方向上的固定单位权重聚合：每层接收所有历史层输出的等权累加，因此会带来 hidden-state magnitude 随深度增长和 `PreNorm Dilution`。
- `Attention Residuals (AttnRes)` 用 depth-wise softmax attention 替代固定残差累加。每层只引入一个可学习 pseudo-query，并对历史层输出或 block 表示做选择性聚合；RMSNorm 用于避免大幅度表示主导 attention 权重。
- 为了让方案适合大规模训练，论文提出 `Block AttnRes`：块内维持标准残差累加，块间对 block representation 做 attention，将内存与跨 pipeline stage 通信从 `O(Ld)` 降到 `O(Nd)`，再配合 cross-stage caching 与 two-phase computation 控制训练和推理开销。

## 值得关注的论断

- 论文的核心类比是 “time-depth duality”：RNN 在时间维度上被 Transformer attention 替代，而标准 residual 在深度维度上也可以从 recurrence 式累加改为 softmax attention 式选择性读取。
- 作者把标准 residual、Highway 等残差变体统一看成 depth-wise linear attention，而 AttnRes 被表述为 depth-wise softmax attention；这个视角比单纯“加一个跨层 attention”更适合纳入概念页。
- Scaling-law 实验显示 Full / Block AttnRes 在五个模型尺度上都低于 baseline loss；论文称 Block AttnRes 在 5.6 PFLOP/s-days 附近达到 baseline 约 `1.25x` compute 的 loss 水平。
- 在 Kimi Linear 48B total / 3B activated MoE 架构上，Block AttnRes 训练 1.4T tokens 后，论文报告其在所有评测任务上不低于 baseline，提升更集中在多步推理、数学和代码任务。

## 关键概念

- [[Attention Residuals]]
- [[PreNorm Dilution]]
- [[Scaling Laws]]
- [[流水线并行]]
- [[Online Softmax]]
- [[重计算]]
- [[mHC]]

## 相关实体

- [[../entities/Moonshot AI]]

## 与现有 wiki 的关系

- 更新概念页：`Attention Residuals`、`PreNorm Dilution`、`流水线并行`、`Scaling Laws`
- 更新实体页：`Moonshot AI`
- 是否存在冲突：未发现与现有 wiki 的直接冲突；本页将论文本体作为一手来源，与既有知乎解读来源页分开记录。

## 待确认

- 论文引用 `Kimi Linear`、`Moonlight`、`DeepSeek-V3` 等架构作为实验背景；当前 vault 还没有 `Kimi Linear` 独立来源页，后续如果补原始报告，可进一步拆分 MoE、KDA/MLA 和长上下文训练细节。
- `PreNorm Dilution` 当前仍主要来自这篇论文线索；后续可以用更系统的 PreNorm / PostNorm 文献核对该术语是否已被社区广泛采用。
