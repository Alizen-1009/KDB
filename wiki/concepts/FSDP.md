---
type: concept
topic: 并行与分布式
sources: 1
updated: 2026-04-23
---

# FSDP

## 定义

一种以模块或 block 为粒度，在前向和反向过程中按需 all-gather 参数、按需 reduce-scatter 梯度的全分片训练方式，常被视作 ZeRO Stage 3 的工程化实现路径。

## 它解决什么问题

- 让超大模型在更多设备上以接近线性的参数内存缩放方式训练
- 避免所有设备始终完整持有全部参数

## 核心机制

- 参数默认分片存放
- 前向到某个 block 时临时 all-gather 所需参数
- 反向后再把梯度 reduce-scatter，并尽快释放不再需要的完整参数/梯度

## 关键权衡

- 参数内存缩放很好
- 但通信时机更细粒度，对 overlap、调度和网络质量更敏感

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 7 - Parallelism basics]]

## 相关概念

- [[ZeRO]]
- [[数据并行]]
- [[重计算]]

## 研究备注

- 后续可补 FSDP block 划分粒度、prefetch 策略和 activation checkpointing 的协同关系
