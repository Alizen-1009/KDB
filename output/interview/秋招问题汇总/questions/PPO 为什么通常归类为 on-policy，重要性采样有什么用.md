# PPO 为什么通常归类为 on-policy，重要性采样有什么用？

- 整理状态：已整理
- 题目出处：用户提供的 SDE / 强化学习岗位面试回忆

## 30 秒回答

标准 PPO 通常归类为 on-policy 或近似 on-policy：用最近的旧策略采一批数据，更新若干轮后丢弃并重新 rollout。每轮更新内用新旧动作概率比修正旧策略下的采样，允许有限复用。

## 深入解释

固定状态 $s$ 时，$\mathbb E_{a\sim\pi_\theta}f(s,a)=\mathbb E_{a\sim\pi_{\rm old}}[\rho f(s,a)]$，前提是旧策略覆盖新策略需要的支持集。这个动作比不能修正新旧状态访问分布的变化；精确轨迹比会连乘，长序列方差高。PPO 采用局部 surrogate 和 clipping 控制偏移，因截断和冻结状态分布而引入近似。它可以用 stale rollout 做工程变体，但要额外监控策略版本滞后、ratio、KL 和 clip fraction。$\pi_{\rm ref}$ 是 RLHF 的参考锚点，$\pi_{\rm old}$ 是实际采样分布，两者用途不同。

## 关联知识

- [[../../../../wiki/concepts/RL 训推不一致|RL 训推不一致]]

## 参考来源与待核实

- [参考 1](https://arxiv.org/abs/1707.06347)

## 相关面试题

- [[PPO 在 RLHF 中优化什么目标，目标函数各项如何解释]]

## 所属题单

- [[../sets/SDE与强化学习岗位面经|SDE与强化学习岗位面经]]
- [[../README|秋招问题汇总]]
