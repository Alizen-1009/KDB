---
type: concept
topic: 并行与分布式
sources: 1
updated: 2026-07-26
---

# DeepSpeed Ulysses

## 定义

`DeepSpeed Ulysses` 是一种 sequence parallel/context parallel 方法：Attention 前对 Q、K、V 一起执行 All-to-All，把布局从“局部 sequence × 全部 heads”重排为“完整 sequence × 部分 heads”；本地完成这些 heads 的 Attention 后，再 All-to-All 恢复 sequence-sharded layout。

## 数据流

初始 sequence-sharded：

```text
每 rank: Q/K/V [B, S/P, H, D]
```

第一次 All-to-All 后：

```text
每 rank: Q/K/V [B, S, H/P, D]
```

此时每 rank 对负责的 heads 拥有完整 sequence，可调用本地 FlashAttention，Softmax 无需跨 ranks 合并。第二次 All-to-All 将输出从：

```text
[B, S, H/P, D]
```

恢复为：

```text
[B, S/P, H, D]
```

供 Output Projection、Residual、Norm、MLP 等后续 sequence-local 操作使用。

## 不是只交换 Q Head

Ulysses 的核心是 Q/K/V 共同的 layout transpose，而非仅对 Q heads 做 All-to-All：

```text
Sequence Shard -> Head Shard -> Local Attention -> Sequence Shard
```

因为单个 head 的 Attention 需要完整 Q/K/V sequence；只交换 Q 而不重排 K/V 无法本地完成完整 Attention。

## Head 数限制

并行度通常受 attention heads、尤其 GQA/MQA 的 KV heads 可切分性约束。若 CP degree 大于可独立分配的 KV heads，实现可能需要复制 KV、按 query groups 重排、限制 degree 或组合其它并行。MLA latent cache 也不能直接套用传统 KV-head 解释。

## 与 Ring Attention 的区别

| 维度 | Ulysses | [[Ring Attention]] |
| --- | --- | --- |
| 核心动作 | Q/K/V Sequence↔Head layout All-to-All | 固定 Q，K/V 沿 ring 轮转 |
| 通信阶段 | 通常前后各一次 All-to-All | 多轮 P2P |
| Softmax | 每个本地 head 看到完整 sequence | 跨 KV blocks Online Softmax |
| 峰值状态 | 完整 sequence × 部分 heads | 本地 Q + 当前 K/V block |
| 主要限制 | Head/KV-head 可切分性 | Ring 轮数与 latency chain |

## 关键权衡

- 能复用成熟的本地 Attention kernel，collective 次数固定。
- All-to-All 形成明确阶段边界，通常较 Ring 难做细粒度计算重叠。
- 对 head 数少、KV heads 少或网络 All-to-All 弱的配置不一定适合。
- 变长序列和多请求仍需额外负载均衡与 padding 管理。

## 相关来源

- [[../sources/vllm PCP 与 DCP 深度解析]]

## 相关概念

- [[Prefill Context Parallel]]
- [[Ring Attention]]
- [[Sequence Parallelism]]
- [[FlashAttention]]
- [[集合通信]]

## 研究备注

- 待 DeepSpeed Ulysses 原论文/实现核实推理支持、GQA/MLA 处理和 vLLM PCP 采用方式。
