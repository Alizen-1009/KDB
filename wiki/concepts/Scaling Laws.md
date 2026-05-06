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

## 关键权衡

- 能显著降低大模型试错成本
- 但它是经验规律，不是绝对定律，常数项、训练配方和目标指标变化都可能改变结论

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 9 - Scaling laws basics]]
- [[../sources/Attention Residuals]]
- [[../sources/Kimi新作《Attention Residuals》：对Transformer中残差结构的调整]]

## 相关概念

- [[数据缩放定律]]
- [[Critical Batch Size]]
- [[Chinchilla Scaling]]

## 研究备注

- 后续可补 scaling laws 在 pretraining loss、下游能力和对齐后行为上的差异
- `Attention Residuals` 是一个比较直观的案例：官方材料把 `Block AttnRes` 的收益表述为“达到 baseline 约 1.25x compute 的 loss 水平”，说明 scaling law 也可以被用来评估架构改动是否真正带来 compute-efficient 增益
- 论文本体的拟合设置是五个 MoE 模型尺度、相同训练超参、8192 token context，并使用 `L = A * C^-alpha` 形式拟合 validation loss 与 compute；这个结论应限定在该训练配方和模型族内理解
