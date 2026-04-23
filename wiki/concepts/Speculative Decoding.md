# Speculative Decoding

## 定义

一种利用小 draft model 先生成候选 token，再由大 target model 并行验证并接受/拒绝的推理加速方法。

## 它解决什么问题

- 降低逐 token 自回归生成的高延迟
- 把 target model 的一部分顺序生成开销转化为更适合并行检查的工作

## 核心机制

- draft model 先连续猜出若干 token
- target model 对这些候选做并行检查
- 按接受/拒绝规则保留一部分候选，并在需要时回退到 target 分布

## 关键权衡

- 能显著改善吞吐和单 token 生成效率
- 效果依赖 draft model 质量、接受率和系统实现开销

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 10 - Inference systems and optimization]]

## 相关概念

- [[KV Cache]]
- [[Continuous Batching]]

## 研究备注

- 后续可补 Medusa、EAGLE 以及不同 draft-target 组合对接受率和收益的影响
