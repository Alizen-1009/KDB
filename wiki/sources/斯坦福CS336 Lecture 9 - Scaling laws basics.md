---
type: source
source_kind: 课程
topic: 训练与 Scaling
updated: 2026-04-23
---

# 斯坦福CS336 Lecture 9 - Scaling laws basics

## 来源信息

- 标题：斯坦福CS336 Lecture 9 - Scaling laws basics
- 作者：[[../entities/Stanford CS336]]
- 日期：2025 Spring
- 类型：课程讲义
- 原始文件：[[../../raw/articles/斯坦福CS336 Lecture 9 - Scaling laws basics|斯坦福CS336 Lecture 9 - Scaling laws basics]]

## 2-3 条核心摘要

- 这讲把“大模型该怎么设计”改写成 scaling 问题：给定固定算力与时间，模型大小、数据量、训练步数和 batch size 之间存在可外推的经验规律。
- Stanford 把 scaling laws 的价值说得很清楚：不是直接训练大模型试错，而是先在小模型上拟合 power-law 关系，再预测大模型设计与资源分配。
- 讲义从数据 scaling 走到模型-数据联合 scaling，再走到 compute-optimal 训练和 Chinchilla，形成了一条非常完整的设计逻辑链。

## 值得关注的论断

- 很多“大模型超参数选择”其实可以通过小规模 scaling 实验提前预测，而不必等到大模型上再暴力试错。
- `critical batch size` 说明 batch 并不是越大越好，超过某个点后收益会迅速递减。
- Chinchilla 的真正价值不是某个固定比例，而是把“固定训练 FLOPs 下最优模型-数据配比”变成了可系统研究的问题。

## 关键概念

- [[Scaling Laws]]
- [[数据缩放定律]]
- [[Critical Batch Size]]
- [[Chinchilla Scaling]]

## 相关实体

- [[../entities/Stanford CS336]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`Scaling Laws`、`数据缩放定律`、`Critical Batch Size`、`Chinchilla Scaling`
- 会更新哪些实体页：`Stanford CS336`
- 是否存在冲突：与现有 wiki 无直接冲突，但后续需要把这些缩放规律和真实开源模型训练配置联系起来

## 待确认

- 后续可继续补 `Lecture 11 - Scaling details`，把这讲的基础规律推进到更细的训练配方与 scaling 细节
