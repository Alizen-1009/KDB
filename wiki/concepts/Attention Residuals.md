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
- 完整版本 `Full AttnRes` 直接对所有先前层做聚合；工程可落地版本 `Block AttnRes` 则在 block 内保留普通残差，只在 block 级表示之间做注意力

## 关键权衡

- 好处是跨层信息聚合更灵活，训练动态更平滑，并在论文报告里带来稳定的 scaling 和下游收益
- 代价是需要保留更多历史表示并引入新的跨层通信路径，因此 full 版本在显存和通信上不够友好
- `Block AttnRes` 是典型的系统折中：牺牲部分“全深度自由度”，换取更好的工程可部署性

## 相关实体

- [[../entities/Moonshot AI]]

## 相关来源

- [[../sources/Kimi新作《Attention Residuals》：对Transformer中残差结构的调整]]

## 相关概念

- [[PreNorm Dilution]]
- [[流水线并行]]
- [[Scaling Laws]]
- [[mHC]]

## 研究备注

- `Attention Residuals` 和 `mHC` 都在动“跨层聚合 / 残差路径”这件事，但前者是 depth-wise attention 选择性读取历史层，后者是对 widened residual stream 的 mixing 加稳定性约束
