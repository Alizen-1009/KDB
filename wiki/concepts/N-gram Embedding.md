---
type: concept
topic: 模型架构
sources: 1
updated: 2026-08-27
---

# N-gram Embedding

## 定义

Qwen3.8-Flash-Next 的 `N-gram Embedding` 是以局部 n-gram 作为确定性寻址键、从加速器外大型 embedding table 取回向量并增强 token 表示的单层条件记忆模块。

## 它解决什么问题

- 在不按比例增加 backbone 每 token 计算的情况下扩展静态参数容量。
- 把适合确定性查表的局部模式放到主干 MoE 之外。
- 借助 host prefetch 将超大 embedding table 放在加速器外。

## 核心机制

- 最终模型只放置一层，位置为 Layer 2，使 host-memory prefetch 可以与 Layer 1 计算重叠。
- 最终 n-gram tables 有 `51B` 参数，位于 `125B` 主干之外并存放在加速器外。
- 查找采用确定性寻址。论文没有披露具体 n-gram 阶数、slot/hash、压缩方式，因此不能把其它 conditional-memory 实现的细节强加给 Qwen。
- n-gram table 使用无 weight decay 的 Adam；其 key/value projection 中真正的二维 linear maps 可使用 Muon。

## 实验观察

- 在固定 MoE 参数量、只增加 n-gram 参数的实验中，vocabulary scale 从 `20×` 增至 `200×` 时 loss 从 `1.553` 降至 `1.526`。
- 多数下游准确率随扩表出现饱和或波动，中文 C-Eval/CMMLU 较持续改善；因此 loss 与下游排序不能等同。
- 固定总参数预算、用 n-gram table 替换部分 MoE experts 时，loss optimum 与 downstream optimum 也不一致。

## 关键权衡

- 论文称额外 FLOPs 与 latency 可忽略，但没有给出 host-device 带宽、prefetch 延迟、缓存命中率或端到端 serving 消融。
- 确定性寻址便于预取，但 table 越大，容量收益与系统带宽/存储成本之间的关系仍需单独测量。
- 单层 placement 足够是该模型与该训练配方下的结果，不代表所有架构都应固定在 Layer 2。

## 相关实体

- [[../entities/Qwen3.8-Flash-Next]]
- [[../entities/Engram]]

## 相关来源

- [[../sources/On the Design of Qwen3.8-Next Architecture：Evaluation, Efficiency, and Training Stability]]

## 相关概念

- [[Conditional Memory]]
- [[Sparsity Allocation]]
- [[MoE]]
- [[Muon Optimizer]]

## 研究备注

- 需要后续实现或系统论文补充：table layout、n-gram orders、寻址/碰撞处理、缓存层次、带宽、延迟与命中率。
