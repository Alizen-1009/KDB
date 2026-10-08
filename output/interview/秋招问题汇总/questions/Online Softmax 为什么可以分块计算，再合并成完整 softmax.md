# Online Softmax 为什么可以分块计算，再合并成完整 softmax？

- 整理状态：已整理
- 题目出处：既有面试原稿（见文末具体章节）

## 其他问法

- Online Softmax 与普通多遍 Softmax 的计算量、访存量分别如何比较？

## 30 秒回答

对每行 score 维护最大值 `m`、相对该最大值的指数和 `l`，attention 再维护未归一化的加权值和 `u`。合并旧状态 `(m₁,l₁,u₁)` 和新块 `(m₂,l₂,u₂)` 时，令 `m=max(m₁,m₂)`，则 `l=exp(m₁-m)l₁+exp(m₂-m)l₂`，`u=exp(m₁-m)u₁+exp(m₂-m)u₂`，最后 `O=u/l`。这样每个块先用自己的最大值保持数值稳定，再缩放到共同基准，结果与对整行做 stable softmax 等价。若只保存局部归一化输出 `oᵢ` 和 `lseᵢ`，则全局输出是 `Σ exp(lseᵢ-L)oᵢ`，其中 `L=logsumexp(lseᵢ)`。

## 深入解释

对两个块分别有 `uᵢ=Σⱼ exp(sⱼ-mᵢ)vⱼ`、`lᵢ=Σⱼ exp(sⱼ-mᵢ)`。把两块统一到 `m=max(m₁,m₂)` 后，各自乘 `exp(mᵢ-m)` 即可相加。这个缩放同时作用于分子和分母，因此不能直接把两个局部 softmax 输出相加。FP 浮点实现会有归约顺序造成的舍入差异，“精确”指算法没有近似截断。

## 关联知识

- [[../../../../wiki/concepts/Online Softmax|Online Softmax]]

## 参考来源与待核实

- [[../../面试经验#1. FlashAttention|面试经验]]

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)，分块 attention 与 online softmax 的算法依据；原面试原稿仅证明该题曾被收录。

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/Kimi AI Infra一面|Kimi AI Infra 一面]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
