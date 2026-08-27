---
type: concept
topic: 模型架构
sources: 3
updated: 2026-04-23
---

# mHC

## 定义

`mHC` 是 `Manifold-Constrained Hyper-Connections` 的缩写：它在 `HC` 的多路 residual stream 设计上加入流形约束，把 residual mixing 投影到 `doubly stochastic` 空间，以恢复更接近 identity mapping 的稳定传播性质。

## 它解决什么问题

- 缓解 `HC` 在大规模训练中的 loss / gradient 不稳定
- 避免多层复合 residual mixing 导致的信号放大或衰减
- 在保留 expanded residual topology 收益的同时，降低 widened residual stream 带来的系统效率损失

## 核心机制

- 保留 `HC` 的多路 residual streams 与跨流混合框架
- 对 residual mixing 矩阵执行 `Sinkhorn-Knopp` 迭代，把它投影到 `doubly stochastic matrices` 构成的约束空间
- 行和与列和为 `1` 的约束，使每层映射更像对各路 stream 的凸组合，从而更接近“平均信号守恒”
- 由于 doubly stochastic 矩阵在乘法下封闭，这种稳定性会在跨层组合映射中部分保留下来
- 为了让该机制在真实训练里可用，论文同时引入了 kernel fusion、mixed precision、自定义 backward kernel、stage-aligned recomputing 和 `DualPipe` 通信重叠

## 与 Gated Residual 的消融边界

[[Gated Residual|GR]] 保留四路 widened residual stream，但把容量集中在逐 channel sigmoid gated read 和逐分支标量 write，并删除 mHC/HC 的 `Hres` mixing，从而少一次完整 residual state 读取。`25B-A3B`、`560B` tokens 的表 5 中，static mHC / dynamic mHC / GR 的 loss 为 `1.596 / 1.594 / 1.590`，平均分为 `52.49 / 54.47 / 54.66`。dynamic 相对 static 的 loss 只降 `0.002`、平均分却增 `1.98`，说明 loss 不能单独代表下游收益；这些数据也不支持脱离该设置宣称 GR 普遍胜过 mHC。

## GLM-5.3-Flash 配置观察

[[../entities/GLM-5.3-Flash]] 的 checkpoint config 包含：

```text
mhc               = true
hc_mult           = 4
hc_eps            = 1e-6
hc_sinkhorn_iters = 20
```

这可确认模型启用了四路 mHC 及相应数值参数，但只是配置观察；它不构成新的 mHC 机制、训练稳定性或性能收益证据，本页机制仍以 mHC 论文本体为准。

## 关键权衡

- 相比原始 `HC`，训练稳定性和扩展性更强，下游效果也更稳
- 但代价是实现复杂度显著增加，需要自定义 kernel、近似投影和系统层共设计
- 论文报告该方法并非零成本，扩展率 `n=4` 时仍有约 `6.7%` 的训练时间开销

## 相关实体

- [[../entities/DeepSeek-AI]]
- [[../entities/Qwen3.8-Flash-Next]]
- [[../entities/GLM-5.3-Flash]]

## 相关来源

- [[../sources/mHC: Manifold-Constrained Hyper-Connections]]
- [[../sources/On the Design of Qwen3.8-Next Architecture：Evaluation, Efficiency, and Training Stability]]
- [[../sources/glm-5-architecture-evolution]]

## 相关概念

- [[Hyper-Connections]]
- [[算子融合]]
- [[重计算]]
- [[流水线并行]]
- [[Gated Residual]]

## 研究备注

- `mHC` 属于宏观连接拓扑设计，不要和 `MLA`、`MoE`、`Engram` 这类模块级结构改造混在一起理解
