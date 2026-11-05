# GRPO 如何估计优势，DAPO 相比它改了什么？

- 整理状态：已整理
- 题目出处：用户提供的 SDE / 强化学习岗位面试回忆

## 30 秒回答

GRPO 对同一 prompt 采 G 个回答，用组内奖励均值和标准差构造相对 advantage，替代 PPO 的 learned critic，保留旧策略概率比和 clipped surrogate。DAPO 针对长 CoT 加入 Clip-Higher、动态采样、token-level loss 和过长奖励处理。

## 深入解释

Outcome reward 场景中，$A_i=(R_i-\bar R)/(\sigma_R+\epsilon)$，该回答的有效 token 共享 $A_i$。每个 token 用 $\rho_{i,t}=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}})$ 做 clipped surrogate。DeepSeekMath 的 GRPO 目标含对参考策略的 KL 正则；DAPO 论文的目标去掉显式参考 KL，这与四个主要设计分开记。

| 改动 | 解决什么 | 追问 |
|---|---|---|
| Clip-Higher | 上界 $1+\epsilon_{\rm high}$ 更宽，正优势低概率 token 有更多上升空间 | 下界独立，clip 不保证全局 KL |
| Dynamic Sampling | 二元奖励下筛去全对或全错组，继续采直到有效组够一批 | 组内优势为零；过滤会改变训练分布并增加 rollout 成本 |
| Token-level loss | 从每回答先按长度平均，再按回答平均，改成全批有效 token 统一平均 | 长回答总权重更大；变长跨 rank 用全局 token 分母 |
| Overlong reward shaping | 处理截断造成的奖励噪声，软惩罚在靠近长度上限的缓冲区逐渐增加 | 不应把截断统一解释成推理失败；监控长度、截断率和奖励 |

实现链路：带权重版本的 rollout → reward → 按 prompt 分组 → mask/advantage → Trainer 重算当前 logprob → 优化更新 → 同步权重。优先排查采样温度、模板、position、mask 和旧 logprob 是否来自实际行为分布。

## 关联知识

- [[../../../../wiki/concepts/RL 训推不一致|RL 训推不一致]]

## 参考来源与待核实

- [参考 1](https://arxiv.org/abs/2503.14476)
- [参考 2](https://arxiv.org/abs/2402.03300)

## 相关面试题

- [[PPO 在 RLHF 中优化什么目标，目标函数各项如何解释]]

## 所属题单

- [[../sets/SDE与强化学习岗位面经|SDE与强化学习岗位面经]]
- [[../README|秋招问题汇总]]
