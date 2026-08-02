---
type: concept
topic: 并行与分布式
sources: 1
updated: 2026-07-26
---

# DeepSpeed Ulysses

## 定义与归属

`DeepSpeed Ulysses` 是 **Microsoft DeepSpeed 团队**提出并维护的长序列训练 sequence parallel 方法，不是 DeepSeek 提出的技术；`DeepSpeed` 与 `DeepSeek` 只是名字相近。

它在 Attention 前对 Q、K、V 一起执行 All-to-All，把布局从“局部 sequence × 全部 heads”重排为“完整 sequence × 部分 heads”；本地完成这些 heads 的 Attention 后，再 All-to-All 恢复 sequence-sharded layout。原始目标是训练/微调极长序列 Transformer，而不是 LLM serving 的逐 token Decode。

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

## Head 数限制与现代 GQA/MQA 支持

- Q head 数必须能被 sequence-parallel size 整除；当前 DeepSpeed HF 实现直接检查 `attn_head_count % SP == 0`。
- 对 KV heads，当前实现允许 `H_kv % SP == 0` 或 `SP % H_kv == 0`。当 `SP > H_kv` 时，会复制 KV heads；源码示例写明 `4 KV heads + SP8` 会做 2 倍 KV replication。
- 因此 GQA/MQA 并非完全不能使用 Ulysses，但少 KV-head 模型会用复制换取 Q/sequence 侧并行，KV 通信与内存收益会变差。
- MLA latent cache 不是标准 per-head K/V；若要套 Ulysses，通常需要物化/适配 QKV 布局或复制 latent KV，不能直接从“Q heads 很多”推导出标准 Ulysses 可无代价使用。

## 与 Ring Attention 的区别

| 维度 | Ulysses | [[Ring Attention]] |
| --- | --- | --- |
| 核心动作 | Q/K/V Sequence↔Head layout All-to-All | 固定 Q，K/V 沿 ring 轮转 |
| 通信阶段 | 通常前后各一次 All-to-All | 多轮 P2P |
| Softmax | 每个本地 head 看到完整 sequence | 跨 KV blocks Online Softmax |
| 峰值状态 | 完整 sequence × 部分 heads | 本地 Q + 当前 K/V block |
| 主要限制 | Head/KV-head 可切分性 | Ring 轮数与 latency chain |

## 适用场景

更适合：

- 训练或 SFT 中的超长 sequence，Attention 计算和激活内存成为瓶颈；
- Q heads 足够多且可被 SP degree 整除的 MHA/GQA Transformer；
- GPU 间 All-to-All 很强，且每轮 Attention 计算足以摊薄 collective latency；
- 希望继续调用本地 SDPA/FlashAttention，而不实现跨 KV blocks 的 distributed online softmax；
- 长文本、长视频/视觉 token 等训练。当前 DeepSpeed 还提供 HF Transformers 集成和 ViT Ulysses wrapper。

不太适合：

- Decode `q_len≈1` 的在线 serving：没有足够 Query sequence 可切；
- 短序列或小模型：两次 layout All-to-All 可能比本地 Attention 更贵；
- `SP > Q heads`，或 Q heads 不能整除 SP；
- MQA/MLA 等 KV 维极窄且复制成本/适配复杂度不可接受的路径；
- All-to-All 跨慢网络或拓扑不友好的部署。

## 关键权衡

- 能复用成熟的本地 Attention kernel，collective 次数固定。
- All-to-All 形成明确阶段边界，通常较 Ring 难做细粒度计算重叠。
- 并行度受 Q head 数约束；Ring Attention 则更容易把 CP degree 扩到 head 数之外。
- GQA/MQA 可通过 KV replication 支持，但会削弱 Ulysses 在 KV 侧的内存/通信效率。
- 变长序列、causal 负载、labels/loss 边界仍需额外处理；DeepSpeed HF adapter 会在数据加载阶段切 sequence 并预先 shift labels。

## 相关来源

- [[../sources/vllm PCP 与 DCP 深度解析]]

## 相关概念

- [[Prefill Context Parallel]]
- [[Ring Attention]]
- [[Sequence Parallelism]]
- [[FlashAttention]]
- [[集合通信]]

## 官方依据

- [DeepSpeed 长序列训练教程](https://github.com/deepspeedai/DeepSpeed/blob/5cc06170ff89812a29b25193f0a418f1a18226f0/docs/_tutorials/ds-sequence.md#L1-L100)
- [DeepSpeed HF/ALST Ulysses 教程](https://github.com/deepspeedai/DeepSpeed/blob/5cc06170ff89812a29b25193f0a418f1a18226f0/docs/_tutorials/ulysses-alst-sequence-parallelism.md)
- [当前 HF 实现中的 Q/KV head 约束与 KV replication](https://github.com/deepspeedai/DeepSpeed/blob/5cc06170ff89812a29b25193f0a418f1a18226f0/deepspeed/runtime/sequence_parallel/ulysses_sp.py#L117-L150)

## 研究备注

- 已核对 vLLM 官方 `main` commit `1ad5182`：官方 PCP 部署文档只列 partial-Q/full-KV AllGather 与 partial-Q/partial-KV Ring 两条路线，仓库中未发现 Ulysses 实现。当前 MRV2 PCP 是 MLA-only 的 AllGather 路径，因此不能把 Ulysses 写成当前 vLLM PCP backend。
- “DeepSeek 自己的 MLA 用不上，所以 Ulysses 没意义”包含两个误区：Ulysses 不是 DeepSeek 项目，且它主要服务通用长序列训练；现代 DeepSpeed 也能通过 KV replication 支持部分 GQA/MQA，只是对单 latent-KV 的 MLA 并非最自然路线。
