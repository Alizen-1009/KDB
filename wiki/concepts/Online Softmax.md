---
type: concept
topic: 注意力机制
sources: 4
updated: 2026-04-23
---

# Online Softmax

## 定义

一种支持按块精确计算 softmax 的数值稳定方法。它在不知道全局最大值的前提下，持续维护每一行的最大值 `m` 与分母累计 `d`，并在新块到来时通过指数补偿把局部结果合并进全局结果。

## 它解决什么问题

- 传统 safe softmax 往往需要对完整 score 多次遍历和存取，容易放大 HBM 读写成本
- 分块 attention 在任意时刻只看到局部 logits，但仍然需要得到与全局 softmax 完全一致的结果
- 为 [[FlashAttention]] 这类 IO-aware attention kernel 提供可分块、可流式执行的精确归一化机制

## 核心机制

- 对每个 block 先计算局部行最大值 `m_block` 与局部分母 `d_block`
- 更新全局最大值：`m_new = max(m_old, m_block)`
- 用 `exp(m_old - m_new)` 和 `exp(m_block - m_new)` 把旧统计与新统计拉回同一参考系
- 分母 `d` 与未归一化输出分子 `O` 独立维护，所有 block 处理完后再统一做一次 `O / d`

## 在 Ring Attention 与 DCP 中

[[Ring Attention]] 固定本地 Q，让 K/V blocks 分轮到达；Online Softmax 维护跨 blocks 的 running max、normalizer 和加权输出，使每个 Q shard 最终得到精确全序列结果。

[[Decode Context Parallel]] 则让 KV 保持在不同 ranks，本地计算 `local_lse/local_out`，再用相同的 log-sum-exp 缩放原则跨 ranks 合并。二者共享数学基础，但一个是 KV block 流水，一个是分布式 partial-output reduction。

## 关键权衡

- 它是 exact reformulation，不是 approximate attention
- 额外 bookkeeping 会增加实现复杂度，需要处理逐行广播、mask 与 block 边界
- 只有把它和 [[Tiling]]、kernel 融合、本地缓存复用结合起来，才会转化成真正的系统性能收益

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 5 - GPUs]]
- [[../sources/Flash Attention 详细解释推演与Pytorch代码实现]]
- [[../sources/秋招CUDA手撕题复盘（附代码）]]
- [[../sources/vllm PCP 与 DCP 深度解析]]

## 相关概念

- [[FlashAttention]]
- [[Tiling]]
- [[Roofline 模型]]
- [[Ring Attention]]
- [[Decode Context Parallel]]

## 研究备注

- 一个容易混淆的点是：`Online Softmax` 不是“近似算 softmax”，而是把稳定 softmax 改写成可流式合并的等价形式
- 在面试语境里，常见追问不是只会背公式，而是能否从普通 safe softmax 一步步推到在线递推与 block combine 版本
