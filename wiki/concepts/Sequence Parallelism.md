# Sequence Parallelism

## 定义

把部分按序列维度独立的激活与逐点操作沿 sequence 轴切分到多个设备上，以继续压缩 activation memory 的并行方式。

## 它解决什么问题

- 在张量并行已经分掉参数计算后，继续降低不会自动缩放的 activation memory
- 缓解 layer norm、dropout 和某些逐点操作在长序列训练中的显存压力

## 核心机制

- 将按序列维度可分的中间张量分散到多个设备
- 在需要完整视图时使用 all-gather
- 在适合回收时使用 reduce-scatter 把结果重新切回去
- 对标准 causal self-attention，SP 不能简单让每张卡只看自己的 token shard；在需要跨 token 依赖的位置，通常会先 all-gather 出完整序列视图，或使用 context parallel / ring attention 这类协议在 attention 内交换 K/V block。

## Attention 依赖如何保证

设 sequence parallel size 为 `P`，总 token 数为 `T`，hidden size 为 `H`。在常见 Megatron-style TP+SP 口径中，子层边界可以保存为：

```text
hidden_local: [T / P, H]
```

但进入 attention 这类需要全局上下文的模块前，会通过通信恢复完整序列视图：

```text
all_gather_seq(hidden_local) -> hidden_full: [T, H]
```

随后每个 TP rank 基于完整序列计算自己负责的 Q/K/V head shard，并在 causal mask 下做 attention：

```text
Q_local, K_local, V_local: [T, heads_local, head_dim]
attention_out_local:      [T, heads_local, head_dim]
```

attention output projection 后，再用 reduce-scatter 把完整 token 维输出切回 sequence shard：

```text
reduce_scatter_seq(attn_out_partial) -> hidden_local: [T / P, H]
```

因此，SP 省的是 layernorm、dropout、残差、部分 MLP/activation 等 token-wise 路径上的激活副本；它不会改变 causal attention 的数学依赖。每个 token 是否能看到前面所有 token，由 all-gather 后的完整 K/V 视图或 attention 内部的 K/V 通信协议保证。

需要区分的是，`Sequence Parallelism` 通常指与 TP 配合的 activation memory 优化；真正为超长上下文把 attention 的序列维也分布式计算的方案，常被称为 `Context Parallelism`、`Ring Attention` 或 `Ulysses` 等。

## 关键权衡

- 能把 activation memory 更接近线性地扩展到多设备
- 但会引入额外通信与更复杂的并行调度

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 7 - Parallelism basics]]

## 相关概念

- [[Tensor Parallelism]]
- [[流水线并行]]
- [[重计算]]

## 研究备注

- 后续可补 Sequence Parallel 在 Megatron-LM 中的具体实现位置，以及它和 context parallel 的边界
