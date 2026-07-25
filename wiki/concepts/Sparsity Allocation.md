---
type: concept
topic: 模型架构
sources: 1
updated: 2026-04-23
---

# Sparsity Allocation

## 定义

`Sparsity Allocation` 指在固定总参数与训练计算预算下，如何在不同稀疏容量之间分配预算的问题；在这篇论文里，具体是如何在 `MoE` 专家容量与 `Engram` 静态记忆容量之间分配。

## 它解决什么问题

- 避免把所有稀疏预算都默认投入同一种机制，例如纯 MoE
- 为“计算容量”和“记忆容量”之间的平衡提供可实验、可量化的设计框架

## 核心机制

- 固定总参数与每 token 激活参数 / FLOPs 预算
- 把未激活参数预算的一部分分给 MoE 专家，另一部分分给记忆表
- 通过 allocation ratio 扫描不同分配比例，比较验证损失与下游表现
- 论文观察到 U 型规律：纯 MoE 和过度记忆化都不是最优，中间区域表现最好

## 关键权衡

- 过度偏向 MoE 时，模型缺少原生静态记忆原语，仍要用深层计算重建固定模式
- 过度偏向记忆时，动态条件计算能力下降，会伤害真正需要推理的任务
- 最优点不是纯理论决定，而是随训练预算、骨干架构和系统限制共同变化

## 相关实体

- [[../entities/Engram]]

## 相关来源

- [[../sources/Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models]]

## 相关概念

- [[Conditional Memory]]

## 研究备注

- 后续可补它与 scaling laws、active / inactive parameters、以及 memory-compute co-design 的关系
