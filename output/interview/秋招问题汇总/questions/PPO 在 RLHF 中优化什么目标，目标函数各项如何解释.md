# PPO 在 RLHF 中优化什么目标，目标函数各项如何解释？

- 整理状态：已整理
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 本页从历史原稿提炼；本轮未逐题重新核验技术事实或运行代码，不表示答案已通过事实复核。

## 30 秒回答

原稿先用奖励模型分数减去相对参考策略的 KL 惩罚，再以 PPO 目标 E[min(rA, clip(r,1−ε,1+ε)A)] 稳定更新。r 是当前与采样旧策略的动作概率比，A 是优势，ε 限制更新幅度；实际还可配 value loss、entropy bonus 和 KL 约束。

## 深入解释

沿文末原稿章节阅读完整机制、推导或代码；本页保留短答，不复制长篇正文。

## 关联知识

- [[../../../../wiki/concepts/流水线并行|流水线并行]]

## PPO-Clip 公式推导

从 policy gradient 出发，$\nabla J=\mathbb E[\nabla\log\pi_\theta(a_t\mid s_t)A_t]$。采样固定在 $\pi_{\rm old}$，冻结优势和状态访问分布，使用动作比 $r_t=\pi_\theta(a_t\mid s_t)/\pi_{\rm old}(a_t\mid s_t)$。由于 $\nabla r_t=r_t\nabla\log\pi_\theta$，在旧策略处（$r_t=1$）优化 $\mathbb E_{\rm old}[r_t\hat A_t]$ 与该处 policy gradient 一致。用 clip 构造保守的逐样本代理目标：

$$
L^{\rm CLIP}=\mathbb E_t\!\left[\min(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)\right].
$$

- $\hat A>0$ 时，$r>1+\epsilon$ 的继续增大不给额外收益。
- $\hat A<0$ 时，$r<1-\epsilon$ 的继续减小不给额外收益。
- 比例向不利方向移动仍受惩罚；clip 作用于目标，不是参数硬约束，也不保证全局 KL。

典型 actor-critic PPO 中，$\delta_t=r_t^{\rm reward}+\gamma V(s_{t+1})-V(s_t)$，终止时按规则屏蔽 bootstrap，$\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$。若做最小化，总 loss 可写 $-L^{\rm CLIP}+c_vL_V-c_eH$。RLHF 可另用固定或慢变 $\pi_{\rm ref}$ 的 KL 作 token 奖励或正则项；采样旧策略 $\pi_{\rm old}$ 用于概率比。实现应使用真实 rollout logprob 并正确 mask prompt/padding。

- [PPO 原论文](https://arxiv.org/abs/1707.06347)
- [[PPO 为什么通常归类为 on-policy，重要性采样有什么用|PPO 的策略分类与重要性采样]]

## 参考来源与待核实

- [[../../面试经验#6. PPO 在 RLHF 框架中优化的核心目标是什么？请写出其目标函数并对其中每一项进行解释|面试经验]]

- 待核实 / 原稿边界：需区分参考模型 π_ref 与采样旧策略 π_old；原稿使用概念级目标，没有完整 token 奖励、GAE 或训练实现。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

## 所属题单

- [[../sets/SDE与强化学习岗位面经|SDE与强化学习岗位面经]]
- [[../sets/分布式训练|分布式训练]]
- [[../README|秋招问题汇总]]
