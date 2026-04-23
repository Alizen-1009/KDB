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

## 关键权衡

- 少量边界分支通常可接受，但热点路径上的严重分歧会快速吞掉吞吐
- 消除分支有时会引入额外计算或更复杂的数据布局，需要结合 bottleneck 判断

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/你一定要知道：CUDA优化六要]]

## 相关概念

- [[GPU执行模型]]
- [[CUDA Kernel]]

## 研究备注

- 后续可补 causal mask、ragged batch 和稀疏路由在 GPU 上引入分支分歧的案例
