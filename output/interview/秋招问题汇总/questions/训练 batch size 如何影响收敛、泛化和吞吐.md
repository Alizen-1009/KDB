# 训练 batch size 如何影响收敛、泛化和吞吐？

- 整理状态：已整理
- 题目出处：用户提供的 SDE / 强化学习岗位面试回忆

## 30 秒回答

全局 batch 决定每次更新平均多少样本或 token。增大 batch 通常降低梯度估计方差、提高设备吞吐，但固定 token 预算下更新次数减少。超过 critical batch 后，达到同一 loss 的步数收益递减。

## 深入解释

等长样本下，全局 batch = 每个 DP rank 的 microbatch × 梯度累积步数 × DP size；TP/PP 不是独立样本副本。变长 LLM 用全局有效 token 计预算和 loss 分母。独立同分布时梯度均值协方差为 Σ/B，相关样本或 token 不满足此近似。固定训练步数与固定 token 数会得出不同结论；linear scaling 和 warmup 是启发式，受优化器和任务影响。做公平对比时对齐 token 预算、调各自的 LR/schedule，比较验证 loss、达到目标所需 tokens、wall time 和吞吐。梯度累积要先累积再更新，BatchNorm 等依赖 microbatch 的运算会破坏简单等价。

## 关联知识

- [[../../../../wiki/concepts/Critical Batch Size|Critical Batch Size]]

## 参考来源与待核实

- [参考 1](https://arxiv.org/abs/1812.06162)

## 所属题单

- [[../sets/SDE与强化学习岗位面经|SDE与强化学习岗位面经]]
- [[../README|秋招问题汇总]]
