---
type: concept
topic: 训练与 Scaling
sources: 1
updated: 2026-08-27
---

# Muon Optimizer

## 定义

`Muon` 是一种对矩阵参数的 Nesterov momentum update 做 Newton–Schulz 正交化的优化器；Qwen3.8-Flash-Next 只将它用于真正充当二维 linear map 的权重，并与 AdamW/Adam 分工。

## 它解决什么问题

- 为二维线性映射提供基于矩阵结构的更新方向，而不是逐元素自适应更新。
- 在 Qwen3.8-Flash-Next 的训练配方中扩大可用学习率与 batch size 的稳定区域。
- 通过分布式负载平衡、语义拆分与 CUDA Graph 降低大规模 Muon step 的工程开销。

## 核心机制

- Nesterov momentum 系数 `μ=0.95`，每步执行 `8` 次 Newton–Schulz（NS）迭代。
- Muon 只用于真正的二维 linear maps，包括 attention q/k/v 与 output projections、GDN input/output projections、routed/shared experts 的 fc1/fc2，以及 n-gram layer 的 key/value projections。
- input embedding、LM head、MoE router、GR 的低秩/门投影使用 AdamW；n-gram embedding table 使用关闭 weight decay 的 Adam。
- 对融合 qkv、GDN input projection 和 SwiGLU fc1，必须先按语义子矩阵拆分，再分别做正交化。对拼接矩阵整体正交化会混合无关子块的 singular directions，并使用错误的整体 shape scaling。
- qkv 与 GDN input 按 per-head 语义拆分；fc1 拆为 gate/up 两半。GDN decay/beta 等每 head 标量投影不适合正交化，output gates 的消融中 AdamW 与 Muon 相当或略优。

## 分布式实现线索

- 论文中的 Canzona 以 whole tensor 为单位，按估计 NS FLOPs 将矩阵分配到 DP ranks，不在 tensor 内切分 optimizer ownership。
- 对 TP 分片权重，通过 fused All-to-All 重构 Muon owner 所需的完整矩阵，再执行与单设备数学等价的 Muon step。
- 语义拆分后每层会产生大量小矩阵，论文用 CUDA Graph capture 整个 optimizer step，减少小 kernel launch overhead。
- Canzona 在本文中是实现线索；不要据此声称它已经开源，也不在本知识库单独建立实体。

## Scaling 与稳定性观察

- 新架构与 Muon 共同改变了 near-optimal batch size 和 learning rate，因此旧 Qwen3.5 配方不能直接沿用。
- `28` 层 `25B-A3B` stress test 在恒定 `2×` optimal LR 下，AdamW 为 `4.3` spikes/万步，两组 Muon 为 `0.2`；在 `4×` 下，AdamW 为 `183` spikes/万步且 `19932` 步中 `213` 次越过 `0.5` clipping threshold，两组 Muon 均不越阈，Muon+GR 为 `0` spike。
- 这些是为复现大规模不稳定而设计的中尺度加压代理；Muon、GR 与结构同时变化时，组合收益不能完全因果拆分。

## 关键权衡

- NS 成本取决于矩阵 shape，而不只是参数量；按元素均分 optimizer state 可能造成严重 straggler。
- 更准确的 `8` 步正交化在论文 stress test 中更稳定，但也增加矩阵运算成本。
- Muon 不是全参数替换 AdamW 的方案；router、embedding、门控和高度细长的低秩矩阵可能不适合正交化。

## 相关实体

- [[../entities/Qwen3.8-Flash-Next]]

## 相关来源

- [[../sources/On the Design of Qwen3.8-Next Architecture：Evaluation, Efficiency, and Training Stability]]

## 相关概念

- [[Scaling Laws]]
- [[Critical Batch Size]]
- [[Gated Residual]]
- [[数据并行]]
- [[Tensor Parallelism]]
- [[CUDA Graph 执行模式]]

## 研究备注

- 论文未说明 Canzona 已开源；具体通信拓扑、bucket geometry 与性能需要绑定实现版本验证。
