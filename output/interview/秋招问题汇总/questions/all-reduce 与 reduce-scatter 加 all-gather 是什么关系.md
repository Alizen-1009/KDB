# all-reduce 与 reduce-scatter 加 all-gather 是什么关系？

- 整理状态：已整理
- 题目出处：既有面试原稿（见文末具体章节）

## 30 秒回答

从结果看，all-reduce 可以分解为 reduce-scatter 再 all-gather：先把各 rank 的对应分片求和并分散持有，再收集完整归约结果到每个 rank。

## 深入解释

这是语义分解，具体 collective 算法、通信量、启动次数和重叠方式取决于实现；若后续只需本地分片，可停在 reduce-scatter，省掉 all-gather。

## 关联知识

- [[../../../../wiki/concepts/集合通信|集合通信]]

## 参考来源与待核实

- [NCCL collectives 文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [[../../多卡与推理系统面试梳理#9. 最后给一个高频追问清单|多卡与推理系统面试梳理]]

- 待核实 / 原稿边界：原稿只给出该追问；前文分别解释通信原语，但未回答此组合关系，故不补答案。

## 所属题单

- [[../sets/分布式训练|分布式训练]]
- [[../README|秋招问题汇总]]
