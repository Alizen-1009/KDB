# Speculative Decoding

## 定义

一种先用较便宜的机制生成候选 token，再由 target model 并行验证并接受/拒绝的推理加速方法；常见实现既包括 `draft model`，也包括 `辅助层` 与 `数据匹配` 路线。

## 它解决什么问题

- 降低逐 token 自回归生成的高延迟
- 把 target model 的一部分顺序生成开销转化为更适合并行检查的工作

## 核心机制

- 先由较便宜的机制连续猜出若干 token
- target model 对这些候选做并行检查
- 按接受/拒绝规则保留一部分候选，并在需要时回退到 target 分布

## 常见路线

- `草稿模型`：用一个更小的 draft model 先生成候选，再由大模型校验
- `辅助层 / 多头预测`：在主模型尾部增加额外 heads 或模块来生成候选，如 `Medusa`、`EAGLE`、`MTP`
- `数据匹配预测`：利用 prompt 或历史数据中的高频模式直接猜测后续 token，如 `ngram`、`suffix decoding`

## 关键权衡

- 能显著改善吞吐和单 token 生成效率
- 效果依赖猜测机制质量、接受率和系统实现开销
- 如果候选经常在第一次校验就失败，总计算量可能反而高于普通 decode
- 不同路线的代价结构差异很大：`draft model` 更吃额外模型协同，`辅助层` 更吃训练耦合，`数据匹配` 更依赖场景重复率

## 框架实现影响

- 不会改写推理系统“每轮完成一次前向”的基本调度逻辑
- 会改变 `KV Cache` 的管理方式：需要为 speculative token 预留位置，并在候选未被采纳时支持回退或覆盖
- 对运行时输入准备、采样和异步调度提出更高要求，这也是 `vLLM MRV2` 强调 speculative decoding 兼容性的原因之一

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/vLLM]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 10 - Inference systems and optimization]]
- [[../sources/Model Runner V2 A Modular and Faster Core for vLLM]]
- [[../sources/LLM提速利器：投机推理的原理与常见方案]]

## 相关概念

- [[KV Cache]]
- [[Continuous Batching]]

## 研究备注

- 现有来源已经把 speculative decoding 从“单一 draft-target 机制”扩展成了一个方案族；后续若频繁引用 `Medusa / EAGLE / MTP`，可再拆独立概念页
- 不同接受规则（阈值比较、拒绝采样、校准）对精确采样分布、收益和实现复杂度的影响，当前 wiki 仍写得偏粗，后续可继续细化
