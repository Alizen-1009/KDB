# Warp Divergence

## 定义

当同一个 warp 内的线程走到不同控制流分支时，GPU 往往需要串行执行各个分支路径，这种执行效率下降称为 `warp divergence`。

## 它解决什么问题

- 解释为什么看似简单的 `if/else` 在 GPU 上会显著拖慢吞吐
- 帮助分析不规则条件逻辑、mask 或边界处理对 kernel 性能的影响

## 核心机制

- warp 是 SIMT 执行的基本调度单位
- 如果 warp 内线程条件不同，硬件通常需要按分支逐段执行，并屏蔽不活跃线程
- 常见优化方式包括重排数据、减少分支、用 predication 或更规则的 block 划分替代复杂控制流
- 一个重要边界是：divergence 发生在 warp 内部，不同 warp 走不同分支通常可以各自全速前进
- 若一个 warp 内 `16` 条线程走 A、`16` 条走 B，常可粗略理解为分支成本接近两段串行执行；若每条线程都走不同路径，代价会迅速放大
- 热点循环里的 per-thread 条件通常比末尾边界判断更危险；后者往往只影响最后一个 warp，前者则可能在每次迭代都触发分歧

## 关键权衡

- 少量边界分支通常可接受，但热点路径上的严重分歧会快速吞掉吞吐
- 消除分支有时会引入额外计算或更复杂的数据布局，需要结合 bottleneck 判断
- 一个常见修复原则是让分支尽量按 warp 边界对齐，或先做数据重排，把同路径线程聚到同一 warp

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/你一定要知道：CUDA优化六要]]
- [[../sources/CUDA优化维度框架]]
- [[../sources/多卡GPU监控与SM执行模型面试整理]]

## 相关概念

- [[GPU执行模型]]
- [[CUDA Kernel]]

## 研究备注

- 后续可补 causal mask、ragged batch 和稀疏路由在 GPU 上引入分支分歧的案例
- LLM 中容易出现 warp divergence 的位置包括 attention 的 causal/window mask、ragged batch 边界、变长序列、MoE 稀疏路由和采样逻辑。推理引擎通常通过 padding/bucketing、sequence packing、专用 attention kernel、warp/block 级规则划分来降低热点路径上的分歧。
