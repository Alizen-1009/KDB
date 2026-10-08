# 为什么 TP 往往与 Sequence Parallelism 一起使用？

- 整理状态：已整理
- 题目出处：既有面试原稿（见文末具体章节）

## 30 秒回答

TP 分摊矩阵权重与部分计算，但 LayerNorm、dropout、残差等激活可能仍在各 TP rank 复制。SP 沿 sequence 切这些激活，复用 TP group 内的 all-gather/reduce-scatter 边界，以减少训练时 activation 显存。

## 深入解释

要核算额外通信和子层输入形状；SP 不会自动减少所有 attention KV 的存储。

## 关联知识

- [[../../../../wiki/concepts/GPU执行模型|GPU执行模型]]
- [[../../../../wiki/concepts/Sequence Parallelism|Sequence Parallelism]]
- [[../../../../wiki/concepts/Tensor Parallelism|Tensor Parallelism]]

## 参考来源与待核实

- [[../../多卡与推理系统面试梳理#TP 的面试重点|多卡与推理系统面试梳理]]

- 待核实 / 原稿边界：仅列为追问并提示缓解激活显存，没有给出 SP 的切分、通信或 attention 依赖细节；保留空答，不补新知识。

## 所属题单

- [[../sets/分布式训练|分布式训练]]
- [[../README|秋招问题汇总]]
