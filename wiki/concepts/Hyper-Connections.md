---
type: concept
topic: 模型架构
sources: 2
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

## 与 Gated Residual 的边界

[[Gated Residual|GR]] 与 HC 都把 residual stream 扩为多分支，但表达力放置不同：GR 用四分支、逐分支逐 channel sigmoid gated read 与逐分支动态标量 write，并移除 `Hres` 分支混合矩阵；每个 block 的 attention 与 MLP 各有独立 GR。论文在 `25B-A3B`、`560B` tokens 的消融中观察到 GR loss/平均分为 `1.590/54.66`，dynamic mHC 为 `1.594/54.47`；这是特定设置下的接近结果，不能推出 GR 普遍优于 HC/mHC。

## 关键权衡

- 表达力和拓扑灵活性增强，而且理论上不需要按主干 FLOPs 等比例扩张
- 但无约束的 residual mixing 会破坏标准 residual connection 的 identity mapping，深层组合后容易出现信号爆炸或衰减
- widened residual stream 还会带来额外 memory access 成本，因此系统效率不再只由 FLOPs 决定

## 相关实体

- [[../entities/DeepSeek-AI]]
- [[../entities/Qwen3.8-Flash-Next]]

## 相关来源

- [[../sources/mHC: Manifold-Constrained Hyper-Connections]]
- [[../sources/On the Design of Qwen3.8-Next Architecture：Evaluation, Efficiency, and Training Stability]]

## 相关概念

- [[mHC]]
- [[算子融合]]
- [[重计算]]
- [[Gated Residual]]

## 研究备注

- 当前仓库里 `HC` 主要通过 `mHC` 论文中的问题重构被引入；其原始 `HC` 论文仍值得单独补 ingest
