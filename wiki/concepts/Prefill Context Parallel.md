---
type: concept
topic: 并行与分布式
sources: 1
updated: 2026-07-26
---

# Prefill Context Parallel

## 定义

`Prefill Context Parallel`（PCP）是在 Prefill 阶段沿 sequence/token 维切分长 Prompt，让多个 GPU 并行计算同一个请求的 Attention，从而分摊长上下文的计算、激活和部分状态压力。

## 它解决什么问题

- 超长 Prompt 的 causal Attention 计算随序列长度近似二次增长，单 GPU TTFT 过高。
- TP 沿 head/hidden 切分，但每个 rank 仍处理完整 sequence；当序列本身成为主导维度时，需要沿 sequence 继续并行。
- [[Chunked Prefill]] 降低单步峰值并改善调度，但 chunks 对同一请求通常按时间串行，不能单独让多个 GPU 同时完成一个 chunk。

## 核心机制

初始布局通常是 sequence-sharded：

```text
每 rank: Q/K/V [B, S/P, H, D]
```

逐 token 的 Norm、MLP、Residual 可本地计算；Attention 中每个本地 Query 仍需看到全序列 K/V，因此需要：

- [[Ring Attention]]：Q shard 固定，K/V blocks 沿 ring 流动；
- [[DeepSpeed Ulysses]]：Q/K/V All-to-All，从 Sequence Shard 重排为 Head Shard。

两者都保持精确 Attention，但通信模式、内存峰值和可扩展限制不同。

## 与 Chunked Prefill

二者是不同层级：

```text
Scheduler：把长 Prompt 切成时间上的 prefill chunks
PCP：把本轮 chunk 再沿 sequence 分给多个 GPUs 并行计算
```

Chunking 控制公平性、峰值显存和与 Decode 的干扰；PCP 分摊单个 chunk 的计算。过小 chunk 会削弱 PCP 的计算粒度，实际需要联合调优。

## 与 DCP

- PCP：Prefill 有大量 Queries，主要分摊计算，目标偏向 TTFT。
- [[Decode Context Parallel]]：Decode 每步 Query 很少，主要分片历史 KV，目标偏向容量、带宽和吞吐。
- 二者都沿 context 维并行，但数据流和 collective 不能混用。

## 关键权衡

- 可降低单个超长 Prompt 的 Prefill 延迟和单 rank 激活压力。
- 引入 P2P/All-to-All，短序列或通信较慢时可能得不偿失。
- Causal Attention 的不同 query chunks 工作量不等，需要 zigzag/striped 等负载均衡。
- vLLM 的 PCP 在当前来源中仍带 RFC/未来版本语境，不能把示例 CLI 当作稳定接口。

## 相关实体

- [[../entities/vLLM]]

## 相关来源

- [[../sources/vllm PCP 与 DCP 深度解析]]

## 相关概念

- [[Ring Attention]]
- [[DeepSpeed Ulysses]]
- [[Decode Context Parallel]]
- [[Chunked Prefill]]
- [[Online Softmax]]
- [[通信-计算重叠]]

## 研究备注

- 待官方文档/commit 确认 PCP 是否已合并、正式 group 构造、参数名和与 DCP 的联合约束。
