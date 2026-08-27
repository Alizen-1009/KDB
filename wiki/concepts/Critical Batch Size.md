---
type: concept
topic: 训练与 Scaling
sources: 2
updated: 2026-04-23
---

# Critical Batch Size

## 定义

指 batch size 增长到某个范围后，继续增大 batch 对目标 loss 或训练效率的收益明显递减的临界区域。

## 它解决什么问题

- 帮助判断训练该优先增加 batch，还是优先增加步数、模型规模或数据量
- 为数据并行扩展和全局 batch 配置提供边界感

## 核心机制

- 小 batch 阶段，增大 batch 往往能明显减少达到目标 loss 所需步数
- 超过某个临界点后，进一步增大 batch 带来的样本效率改善会迅速变弱
- 临界 batch 还会随目标 loss 水平和训练规模变化而移动

## Qwen3.8-Flash-Next 的模型级实验

- 在 `20` 层 `10.8B-A0.89B` MoE、`4T` tokens 的等 token 预算实验中，batch `25.2M` 的 loss 为 `1.5702`；旧配方 `12.6M` 为 `1.5774`，更大的 `37.7M` 为 `1.5707`。这说明预测点附近已进入平坦区域，而不是证明 `25.2M` 是所有模型的固定 critical batch。
- batch 从 `6.3M` warmup 到 `25.2M` 不优于一开始就用 `25.2M`，且同 token 预算需要多 `18.8%` optimizer steps；该结论绑定 [[Muon Optimizer|Muon]]、稀疏 MoE、模型规模和训练配方，不能泛化为“batch warmup 总是无效”。
- 在更大的 `48` 层 `156B-A7B`、`419B` tokens 实验中，新 scaling fit 预测 `B=8.4M`；邻近 batch/LR 组合只各评测一次，细小排序被论文视为噪声。

## 关键权衡

- 更大的 batch 往往更利于并行硬件利用
- 但过大 batch 可能只是在吃更多资源，却没有换来成比例的收敛收益

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/Qwen3.8-Flash-Next]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 9 - Scaling laws basics]]
- [[../sources/On the Design of Qwen3.8-Next Architecture：Evaluation, Efficiency, and Training Stability]]

## 相关概念

- [[Scaling Laws]]
- [[数据并行]]
- [[Muon Optimizer]]

## 研究备注

- 后续可补临界 batch size 与学习率缩放、gradient accumulation、并行策略之间的关系
