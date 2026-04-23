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

## 关键权衡

- 更大的 batch 往往更利于并行硬件利用
- 但过大 batch 可能只是在吃更多资源，却没有换来成比例的收敛收益

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 9 - Scaling laws basics]]

## 相关概念

- [[Scaling Laws]]
- [[数据并行]]

## 研究备注

- 后续可补临界 batch size 与学习率缩放、gradient accumulation、并行策略之间的关系
