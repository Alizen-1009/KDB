---
type: concept
topic: 训练与 Scaling
sources: 1
updated: 2026-04-23
---

# Chinchilla Scaling

## 定义

围绕固定训练计算预算下的模型大小与训练数据量最优配比而形成的一组 scaling 结论，通常用来讨论 compute-optimal training。

## 它解决什么问题

- 回答“固定训练 FLOPs 下，大模型少训练还是小模型多训练”这类资源分配问题
- 为 token / parameter 比例和训练配方设计提供更系统的依据

## 核心机制

- 在固定 compute 预算下比较不同模型大小和训练 token 数的性能
- 通过 IsoFLOPs、joint fits 等方法拟合 compute-optimal 曲线
- 强调学习率调度、训练长度和拟合方法会影响最优常数项

## 关键权衡

- 对 pretraining 阶段的资源配置很有指导意义
- 但“train-optimal”不等于“deployment-optimal”，真实系统中推理成本和长期使用量也会反过来影响最优选择

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 9 - Scaling laws basics]]

## 相关概念

- [[Scaling Laws]]
- [[数据缩放定律]]

## 研究备注

- 后续可补 Kaplan、Chinchilla、Llama / Mistral 等公开训练配方之间的 token-per-parameter 差异
