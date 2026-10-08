# RoPE 在 prefill 和 decode 阶段分别如何处理？

- 整理状态：已整理
- 题目出处：既有面试原稿（见文末具体章节）

## 30 秒回答

Prefill 为一段 token 按各自 position id 旋转 Q/K，再写入已经旋转的 K；decode 对当前 token 的 Q/K 用其逻辑位置旋转，只把新 K/V 追加到 cache，并以当前 Q 与历史 K 做 attention。两阶段必须使用同一位置约定和 RoPE 配置。

## 深入解释

分批、前缀复用或 sliding window 中，position id 不是当前 batch 的下标；要从请求的逻辑位置恢复。

## 关联知识

- [[../../../../wiki/concepts/RoPE|RoPE]]

## 参考来源与待核实

- [RoFormer 原论文](https://arxiv.org/abs/2104.09864)
- [[../../面试经验#4. 然后往 AI infra 方向深入|面试经验]]


## 所属题单

- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
