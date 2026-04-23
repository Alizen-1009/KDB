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

## 相关概念

- [[FlashAttention]]
- [[Tiling]]
- [[Roofline 模型]]

## 研究备注

- 一个容易混淆的点是：`Online Softmax` 不是“近似算 softmax”，而是把稳定 softmax 改写成可流式合并的等价形式
- 在面试语境里，常见追问不是只会背公式，而是能否从普通 safe softmax 一步步推到在线递推与 block combine 版本
