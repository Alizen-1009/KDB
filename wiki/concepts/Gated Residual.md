---
type: concept
topic: 模型架构
sources: 1
updated: 2026-08-27
---

# Gated Residual

## 定义

`Gated Residual (GR)` 是 Qwen3.8-Flash-Next 的四分支残差结构：每个子层通过逐分支、逐 channel 的 sigmoid gate 读取 widened residual stream，再用每分支动态标量写回。

## 它解决什么问题

- 扩大 residual stream 容量，让不同跨层信息路径不必竞争同一条单流残差。
- 用数据相关 read/write 决定四分支容量如何被当前 attention 或 MLP 使用。
- 通过 bounded gate 提供重缩放并改善高学习率压力下的训练稳定性。

## 核心机制

- residual state 有 `4` 个分支，每个分支使用独立 RMSNorm gain。
- read gate 按分支、按 channel 生成 sigmoid 权重，并从所有分支形成子层输入；低秩瓶颈 rank 为 `r=d/8`。
- write 为每个分支生成一个动态标量，把子层输出写回所有分支。
- 每层的 attention block 与 MLP block 各自拥有独立 GR。
- 相比 HC/mHC，GR 移除 `Hres` 分支混合矩阵；在 read/write 已足够表达时，论文消融未见显著收益，移除它还少一次完整 residual state 读取。
- GR 的 gated read 已包含归一化，因此替代 block 原有的 pre-normalization，而不是在其前面叠加一层。

## 评测与稳定性观察

- 在 `25B-A3B`、`560B` tokens 的表 5 中，pre-norm / static mHC / dynamic mHC / GR 的 loss 为 `1.617 / 1.596 / 1.594 / 1.590`，九项平均分为 `50.91 / 52.49 / 54.47 / 54.66`。
- dynamic mHC 相对 static mHC 的 loss 只降 `0.002`，但 benchmark 平均分增 `1.98`；这说明预训练 loss 与下游表现不总是同序。
- top-2 sparse read 在预训练 loss/benchmark 上近乎无损，但 post-training 明显退化，因此未采用。
- 来源观察称 FP8 residual storage 相对 BF16 将残差状态流量减半，且几乎无质量损失；这不是未限定条件的端到端加速结论。
- `28` 层特定对比中，未加 GatedNorm 的 Full AttnRes 与 GR（结构本身包含 gated read/GatedNorm）的 loss 都是 `1.762`；给 Full AttnRes 加 GatedNorm 后为 `1.758`。这些并非完全同构的配置，该消融不支持宣称 GR 或 AttnRes 存在普遍优劣。

## 关键权衡

- 四分支会增加 residual state 的内存容量与读写流量，收益不能只按新增 FLOPs 判断。
- dense per-channel read 提供表达力，但稀疏读取在 post-training 上暴露质量风险。
- GR、Muon、GDN hybrid 同时改变训练动态；组合 stress test 不能完全拆出每个组件的独立因果贡献。

## 相关实体

- [[../entities/Qwen3.8-Flash-Next]]

## 相关来源

- [[../sources/On the Design of Qwen3.8-Next Architecture：Evaluation, Efficiency, and Training Stability]]

## 相关概念

- [[Hyper-Connections]]
- [[mHC]]
- [[Attention Residuals]]
- [[RMSNorm]]

## 研究备注

- 单变量 GatedNorm 实验在 `28` 层中尺度模型、恒定 `3×` optimal LR 下将 spike rate 从 `32.0` 降到 `3.2`/万步，越过 `0.5` clip threshold 的次数从 `256` 降到 `20`；这是加压代理结果，不等价于 production run 的普遍因果结论。
