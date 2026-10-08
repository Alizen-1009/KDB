# 什么是 Sequence Parallelism？

- 整理状态：已整理
- 题目出处：既有面试原稿（见文末具体章节）

## 30 秒回答

在常见 Megatron 式 TP+SP 中，把部分 activation 沿序列维分到 TP ranks，让 LayerNorm、dropout、残差等逐 token 操作只保存本地 token shard；进入需要完整序列的子层时 all-gather，输出可 reduce-scatter 回序列 shard。目标主要是降低 activation 显存。

## 深入解释

它不能让标准 causal attention 只看本地 KV；这与 Ulysses/Ring 等 context parallel 协议不同。

## 关联知识

- [[../../../../wiki/concepts/GPU执行模型|GPU执行模型]]
- [[../../../../wiki/concepts/Sequence Parallelism|Sequence Parallelism]]

## 参考来源与待核实

- [[../../大模型系统面试题地图#A. 已经有较好锚点，可以直接展开|大模型系统面试题地图]]


## 所属题单

- [[../sets/分布式训练|分布式训练]]
- [[../README|秋招问题汇总]]
