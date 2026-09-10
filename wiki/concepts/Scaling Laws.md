---
type: concept
topic: 训练与 Scaling
sources: 4
updated: 2026-05-06
---

# Scaling Laws

## 定义

用简洁的经验函数去描述模型性能如何随数据规模、模型规模或计算预算变化的规律，常见形式是在 log-log 坐标下近似线性。

## 它解决什么问题

- 帮助研究者在小规模实验基础上预测大规模模型训练趋势
- 把“模型该做多大、训多久、吃多少数据”从纯经验调参推进到可外推的设计问题

## 核心机制

- 通过一系列不同规模实验拟合误差、loss 或下游性能与规模变量的关系
- 常用 power-law 形式描述数据、模型、compute 与误差之间的关系
- 用拟合到的小规模规律预测更大规模下的资源配置和性能走势

## Qwen3.8-Flash-Next 的超参数外推验证

- 新架构与 [[Muon Optimizer|Muon]] 改变了 near-optimal batch size 和 learning rate，论文因此重新拟合 Qwen3.5 系列使用的超参数 scaling law，而不是直接沿用旧配方。
- `20` 层 `10.8B-A0.89B`、`4T` tokens 验证中，`B=25.2M` 的 loss 为 `1.5702`，旧 `B=12.6M` 为 `1.5774`，`B=37.7M` 为 `1.5707`；从 `6.3M` warmup 到 `25.2M` 不更好且多 `18.8%` optimizer steps。
- `48` 层 `156B-A7B`、`419B` tokens 验证中，预测配置为 `B=8.4M / LR=1.76e-3`，旧配方为 `4.2M / 6.8e-4`；预测配置最终 loss 优 `7.8e-3`，七项平均准确率为 `60.55 vs 56.41`。
- 预测点附近的四组设置最终 loss 相差不超过 `7e-4`，且只各做一次下游评测；论文明确把细小排名视为观测噪声。以上验证只支持该模型族与训练配方内的外推，不能变成通用 batch/LR 定律。

## 关键权衡

- 能显著降低大模型试错成本
- 但它是经验规律，不是绝对定律，常数项、训练配方和目标指标变化都可能改变结论

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/Qwen3.8-Flash-Next]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 9 - Scaling laws basics]]
- [[../sources/Attention Residuals]]
- [[../sources/Kimi新作《Attention Residuals》：对Transformer中残差结构的调整]]
- [[../sources/On the Design of Qwen3.8-Next Architecture：Evaluation, Efficiency, and Training Stability]]

## 相关概念

- [[数据缩放定律]]
- [[Critical Batch Size]]
- [[Chinchilla Scaling]]
- [[Muon Optimizer]]

## 研究备注

- 后续可补 scaling laws 在 pretraining loss、下游能力和对齐后行为上的差异
- `Attention Residuals` 是一个比较直观的案例：官方材料把 `Block AttnRes` 的收益表述为“达到 baseline 约 1.25x compute 的 loss 水平”，说明 scaling law 也可以被用来评估架构改动是否真正带来 compute-efficient 增益
- 论文本体的拟合设置是五个 MoE 模型尺度、相同训练超参、8192 token context，并使用 `L = A * C^-alpha` 形式拟合 validation loss 与 compute；这个结论应限定在该训练配方和模型族内理解
