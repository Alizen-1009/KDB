# PreNorm Dilution

## 定义

`PreNorm Dilution` 指的是在标准 PreNorm Transformer 里，随着层数增加，先前层输出被不断累加，导致任意单层对当前表示的相对贡献逐渐被稀释的现象。

## 它解决什么问题

- 它不是一个方法，而是一个被指出的结构性问题
- 这个问题帮助解释：为什么固定残差累加在非常深的网络里，可能让后层难以保持清晰、可辨识的增量贡献
- 也帮助解释 hidden-state 幅度与梯度在深度方向上的不均衡

## 核心机制

- 标准残差路径默认把历史输出按固定权重直接并入当前表示
- 当层数加深时，累加项不断变多，单层新增信息在总和中的相对占比下降
- 对 PreNorm 架构而言，这会和 hidden-state 幅度增长一起出现，进而影响训练稳定性与层间贡献分布
- `Attention Residuals` 论文的实验观察是：baseline 的 hidden-state magnitude 随深度单调增长，早期层梯度更大；Block AttnRes 通过 block 边界处的选择性聚合，让输出幅度呈受限的周期性模式，梯度分布也更均匀

## 关键权衡

- 固定残差的优点是结构简单、稳定、实现便宜
- 但它隐含了“所有历史层等权重要”的假设，这在超深网络里可能并不理想
- `Attention Residuals` 的意义之一，就是把这条固定假设替换成选择性聚合

## 相关实体

- [[../entities/Moonshot AI]]

## 相关来源

- [[../sources/Attention Residuals]]
- [[../sources/Kimi新作《Attention Residuals》：对Transformer中残差结构的调整]]

## 相关概念

- [[Attention Residuals]]

## 研究备注

- 这个术语在当前知识库里主要来自 `Attention Residuals` 论文线索；后续若补更系统的 PreNorm / PostNorm 文献，可进一步核对它是否是更广泛采用的标准术语
