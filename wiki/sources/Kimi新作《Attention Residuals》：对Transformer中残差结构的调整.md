---
type: source
source_kind: 文章
topic: 模型架构
updated: 2026-04-23
---

# Kimi新作《Attention Residuals》：对Transformer中残差结构的调整

## 来源信息

- 标题：Kimi新作《Attention Residuals》：对Transformer中残差结构的调整
- 作者：Loster分享AI算法，多模态大模型，AIGC，agent内容～
- 日期：2026-03-19（文章编辑时间）；原始剪藏入库时间为 2026-04-22
- 类型：知乎文章 / 论文解读
- 原始文件：`raw/articles/Kimi新作《Attention Residuals》：对Transformer中残差结构的调整.md`
- 补充核对资料：`raw/papers/2603.15031v1.pdf`；MoonshotAI 官方仓库 `Attention-Residuals`

## 2-3 条核心摘要

- 这篇文章围绕 Moonshot AI / Kimi 团队的 `Attention Residuals` 论文展开，核心观点是：标准 PreNorm Transformer 中固定单位权重的残差累加会带来 hidden state 幅度增长和 `PreNorm Dilution`，而 `AttnRes` 用深度方向上的 softmax 注意力替代固定加和。
- 为了让“对历史层做 attention”在大模型训练里可落地，论文进一步提出 `Block AttnRes`：块内保持普通残差累加，块间只对 block representation 做 attention，从而把全深度方案的内存 / 通信负担压到可训练范围。
- 这篇资料的价值不只在架构点子本身，还在于它把 `cache-based pipeline communication` 和 `two-phase computation` 一起放进方案里，说明作者在设计之初就把训练栈和分布式实现一起考虑了。

## 值得关注的论断

- 文章强调 `Block AttnRes` 是标准残差的 `drop-in replacement`，这点与论文摘要和官方仓库 README 的说法一致，但“几乎无额外开销”更适合理解成工程上可接受，而不是零成本。
- 文章引用的“相当于 1.25 倍 baseline compute”的说法与官方仓库 README 一致，较适合作为 scaling-law 实验结论记录，而不是泛化成所有模型都能稳定获得的固定收益。
- 文末“成为下一代大语言模型基础架构标准”更像作者判断，不是论文直接结论；知识库里应保留为趋势判断，而非既成事实。

## 关键概念

- [[Attention Residuals]]
- [[PreNorm Dilution]]
- [[流水线并行]]
- [[Scaling Laws]]

## 相关实体

- [[../entities/Moonshot AI]]

## 与现有 wiki 的关系

- 会创建哪些概念页：`Attention Residuals`、`PreNorm Dilution`
- 会创建哪些实体页：`Moonshot AI`
- 会更新哪些概念页：`流水线并行`、`Scaling Laws`
- 是否存在冲突：与现有 wiki 无直接冲突，但它和 `mHC` 一样都属于“改造跨层信息聚合 / 残差拓扑”的路线，后续值得并排比较

## 待确认

- 当前仓库里还没有 `Kimi Linear` 本身的独立来源页；如果后续补官方技术报告，可以把 `Attention Residuals` 在该架构中的部署细节拆得更完整
