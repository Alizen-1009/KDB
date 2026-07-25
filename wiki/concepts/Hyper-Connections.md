---
type: concept
topic: 模型架构
sources: 1
updated: 2026-04-23
---

# Hyper-Connections

## 定义

`Hyper-Connections (HC)` 是一种宏观架构设计：把单一路 residual stream 扩展为多路并行 residual streams，并引入可学习映射来控制层输入读出、层输出写回以及流内混合。

## 它解决什么问题

- 在不按同样比例增加单层 FLOPs 的前提下，提高网络的拓扑复杂度与信息流宽度
- 把 residual stream 的容量从单个层函数的计算规模里部分解耦出来
- 为模型提供一条区别于“更大 attention / 更宽 MLP / 更多 experts”的新扩展轴

## 核心机制

- 把每层输入扩展成 `n` 路 residual streams，而不是只保留单一路残差
- 用可学习映射控制 residual stream 到 layer input 的读出、layer output 回写到 stream，以及 stream 之间的混合
- 这些额外映射本身计算量不大，但会持续作用在跨层组合映射上

## 关键权衡

- 表达力和拓扑灵活性增强，而且理论上不需要按主干 FLOPs 等比例扩张
- 但无约束的 residual mixing 会破坏标准 residual connection 的 identity mapping，深层组合后容易出现信号爆炸或衰减
- widened residual stream 还会带来额外 memory access 成本，因此系统效率不再只由 FLOPs 决定

## 相关实体

- [[../entities/DeepSeek-AI]]

## 相关来源

- [[../sources/mHC: Manifold-Constrained Hyper-Connections]]

## 相关概念

- [[mHC]]
- [[算子融合]]
- [[重计算]]

## 研究备注

- 当前仓库里 `HC` 主要通过 `mHC` 论文中的问题重构被引入；其原始 `HC` 论文仍值得单独补 ingest
