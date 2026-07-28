---
type: concept
topic: 注意力机制
sources: 1
updated: 2026-07-26
---

# Ring Attention

## 定义

`Ring Attention` 是沿 sequence 切分 Attention 的流式算法：每个 rank 固定本地 Query chunk，让 K/V blocks 沿设备 ring 轮转；每收到一块就计算局部 block Attention，并用 [[Online Softmax]] 精确累积，直到本地 Q 看过全部 K/V。

## 数据流

初始每 rank 持有：

```text
Q_r, K_r, V_r: [B, S/P, H, D]
```

执行 `P` 轮左右：

```text
Round 0: Q_r × (K_r, V_r)
Round 1: Q_r × (K_{r-1}, V_{r-1})
...
K/V 每轮发送给下一个 rank
```

Q 不动，K/V 轮流到达。每个 rank 只需本地 Q、当前 K/V block 和在线 Softmax 状态，不必一次持有完整序列 K/V。

## Online Softmax 合并

局部 `softmax` 输出不能直接相加。算法对每个 Query row 维护 running max、normalizer 和未归一化加权和；新 KV block 到达时把旧统计和新统计缩放到同一最大值参考系后合并，最终与完整 Attention 数学等价。

## Causal 负载均衡

连续 sequence chunks 在 causal mask 下工作量不同：较晚 Q chunk 需要看更多历史 K/V。生产实现常用 zigzag/striped 分片、对称 token 分配或跳过全 mask blocks，避免最后 ranks 成为 tail。

## 与 Ulysses 的区别

- Ring：Q shard 固定，K/V 多轮 P2P；跨 blocks 用 Online Softmax。
- [[DeepSpeed Ulysses]]：Q/K/V 一次 All-to-All 变为完整 sequence + 部分 heads，本地做常规 Attention，再 All-to-All 切回。
- Ring 对 head 数依赖较小、峰值内存低且易 overlap，但轮数随 CP degree 增长；Ulysses collective 次数少，但受 heads/KV heads 可切分性限制。

## 关键权衡

- 优点：低峰值内存、适合超长序列，P2P 可与 block Attention 流水重叠。
- 代价：多轮通信、长依赖链、online-softmax bookkeeping 和 causal imbalance。
- 序列不够长时，每轮计算不足以隐藏 P2P latency。

## 相关来源

- [[../sources/vllm PCP 与 DCP 深度解析]]

## 相关概念

- [[Prefill Context Parallel]]
- [[DeepSpeed Ulysses]]
- [[Online Softmax]]
- [[通信-计算重叠]]
- [[Tiling]]

## 研究备注

- 当前来源为二手解读；精确通信复杂度、反向过程和 vLLM 实现应结合 Ring Attention 原论文/RFC 核实。
