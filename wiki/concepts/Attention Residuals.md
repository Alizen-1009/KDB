---
type: concept
topic: 模型架构
sources: 4
updated: 2026-05-06
---

# Attention Residuals

## 定义

`Attention Residuals (AttnRes)` 是一种对 Transformer 残差路径的改造：它不再把所有先前层输出按固定单位权重直接累加，而是让当前层通过深度方向上的 softmax 注意力，自适应聚合更早的层表示。

## 它解决什么问题

- 缓解标准 PreNorm 残差在深层网络中出现的 hidden-state 幅度增长
- 缓解早期层不断累加后对单层贡献的稀释，也就是 `PreNorm Dilution`
- 让跨层聚合从“固定加和”变成“输入相关、内容相关的选择性读取”

## 核心机制

- 对于第 `l` 层，不再直接使用历史层输出的单位权重和
- 改为对前面层的表示做 depth-wise attention，得到从历史层到当前层的注意力权重
- 原论文使用每层一个可学习的 `pseudo-query` 向量，与历史层表示做打分；由于历史表示本身依赖当前输入，上述聚合仍然是 content-aware 的
- `RMSNorm` 被放在 key 侧，用于避免大幅度历史表示天然压过其他来源；pseudo-query 需要零初始化，使训练初期近似等权平均，降低训练波动
- 完整版本 `Full AttnRes` 直接对所有先前层做聚合；工程可落地版本 `Block AttnRes` 则在 block 内保留普通残差，只在 block 级表示之间做注意力
- 论文把标准 residual / Highway 等 recurrence 式残差统一看作 depth-wise linear attention，而 AttnRes 是 depth-wise softmax attention

## 关键权衡

- 好处是跨层信息聚合更灵活，训练动态更平滑，并在论文报告里带来稳定的 scaling 和下游收益
- 代价是需要保留更多历史表示并引入新的跨层通信路径，因此 full 版本在显存和通信上不够友好
- `Block AttnRes` 是典型的系统折中：牺牲部分“全深度自由度”，换取更好的工程可部署性
- 在 pipeline parallelism 场景下，论文用 cross-stage caching 减少重复传输，用 two-phase computation 和 online softmax merge 摊薄推理 I/O；报告的训练额外开销小于 4%，典型推理延迟额外开销小于 2%

## vLLM K3 推理实现

K3的93层网络若朴素实现Block AttnRes，会增加跨层表示读写、归约和Normalization Launch。vLLM Preview称Release Branch已集成Triton与NVIDIA Kernel，在支持shape上融合Residual Update、AttnRes Mixing和Output RMSNorm，并用Sequence Parallel分片跨rank流量；文章只称早期Kernel结果积极，端到端收益仍在不同Prefill长度和并行配置下测量。

## 与 Gated Residual 的特定对比

Qwen3.8 论文只在特定 `28` 层设置下比较 [[Gated Residual|GR]] 与 AttnRes：未加 GatedNorm 的 Full AttnRes 与 GR（结构本身包含 gated read/GatedNorm）的最终训练 loss 都为 `1.762`；给 Full AttnRes 加 GatedNorm 后为 `1.758`。Block AttnRes 在 `S=2/4` 时分别为 `1.770/1.773`，加入 GatedNorm 后为 `1.766/1.768`。这些并非完全同构的配置，不能推出 GR 与 AttnRes 的普遍优劣；二者机制也不同，AttnRes 对历史层输出做 depth-wise softmax attention，GR 则读写四分支 residual accumulators。

## 相关实体

- [[../entities/Moonshot AI]]
- [[../entities/vLLM]]
- [[../entities/Qwen3.8-Flash-Next]]

## 相关来源

- [[../sources/Attention Residuals]]
- [[../sources/Kimi新作《Attention Residuals》：对Transformer中残差结构的调整]]
- [[../sources/A Preview of Production-Scale Kimi K3 Support on vLLM]]
- [[../sources/On the Design of Qwen3.8-Next Architecture：Evaluation, Efficiency, and Training Stability]]

## 相关概念

- [[PreNorm Dilution]]
- [[流水线并行]]
- [[Scaling Laws]]
- [[Online Softmax]]
- [[mHC]]
- [[Gated Residual]]

## 研究备注

- `Attention Residuals` 和 `mHC` 都在动“跨层聚合 / 残差路径”这件事，但前者是 depth-wise attention 选择性读取历史层，后者是对 widened residual stream 的 mixing 加稳定性约束
- 论文报告的 `1.25x compute` 是基于 scaling-law 拟合的 loss 等效收益，适合记录为该实验设置下的 compute-efficient 增益，不宜泛化成固定性能倍数
