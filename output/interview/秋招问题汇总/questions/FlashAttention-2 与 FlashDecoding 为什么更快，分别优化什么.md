# FlashAttention-2 与 FlashDecoding 为什么更快，分别优化什么？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 历史内容从原稿提炼，核实边界见文末；本次英伟达五面补充单独注明公式假设和依据，不表示历史原稿全部复核或代码已实测。

## 其他问法

- 讲一下 FlashAttention？FA 怎么实现的？FlashDecoding 是什么？

- FlashAttention-2 / FlashDecoding 的加速原理？为什么比原生注意力快 3-10 倍
- FlashDecoding 的加速原理是什么？

## 30 秒回答

FlashAttention 用分块、Online Softmax 和片上输出累积，避免完整 score/probability 矩阵写入 HBM。Flash Decoding 面向小 Q、长 KV，在 KV/context 维增加 split 并行，再按 Softmax 统计合并；不是近似 Attention，也不是投机解码。

## 深入解释

### 英伟达五面补充

- 本次题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，AI Infra 与推理系统第 2 题；2026-11-05 收录，实际面试日期与具体轮次未提供。
- 本次原题：讲一下 FlashAttention？FA 怎么实现的？FlashDecoding 是什么？

设 Attention 为 O=softmax(QKᵀ/√d)V。FA 将 Q/K/V 分块，维护每行最大值 m、分母 l 和未归一化输出 u。若新 tile 的缩放后得分为 s：

$$m'=\max(m,\max s)$$
$$l'=e^{m-m'}l+\sum_j e^{s_j-m'}$$
$$u'=e^{m-m'}u+\sum_j e^{s_j-m'}v_j,\qquad O=u/l$$

这样不必把完整 S×S score/probability 写入显存。标准稠密 Attention 的算术复杂度仍是二次的；“精确”指数学 Attention 公式不变，不代表浮点 bitwise 一致。

Decode 通常每请求只有一个新 Q。小 batch 时，只沿 batch/head 并行可能用不满 GPU。Flash Decoding 将历史 KV 分成多个 split，算局部归一化输出 Oᵢ 和局部 log-sum-exp Lᵢ，再合并：

$$L=\operatorname{logsumexp}_i L_i,\qquad O=\sum_i e^{L_i-L}O_i$$

局部输出不能直接相加。额外 split 会引入 workspace、合并与启动成本；batch 已大或 KV 较短时，不保证更快。这里解释通用机制，不把具体 backend 的 tile、grid 或版本策略写成统一规则。

历史长篇解释沿文末原稿阅读；本次仅补充上述题干对应的机制、公式和边界。

## 关联知识

- [[../../../../wiki/concepts/Online Softmax|Online Softmax]]

- [[../../../../wiki/concepts/FlashAttention|FlashAttention]]
- [[../../../../wiki/concepts/Flash Decoding|Flash Decoding]]

## 参考来源与待核实

- [[../../大模型系统面试题全答#6. FlashAttention-2 / FlashDecoding 的加速原理？为什么比原生注意力快 3-10 倍|大模型系统面试题全答]]
- [[../../大模型系统面试题地图#1. 硬件与性能模型|大模型系统面试题地图]]

- 待核实 / 原稿边界：保留整体对比题，不把两段答案小标题拆题。题设“3–10 倍”无配置和来源，待核实；原稿对 FLOPs 的表述为简化机制口径。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

### 本次补充依据与边界

- [FlashAttention 原始论文](https://arxiv.org/abs/2205.14135)。
- [Flash Decoding 作者说明](https://pytorch.org/blog/flash-decoding/)。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]

- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
