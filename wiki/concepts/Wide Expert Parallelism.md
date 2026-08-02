---
type: concept
topic: 并行与分布式
sources: 1
updated: 2026-07-26
---

# Wide Expert Parallelism

## 定义

`Wide Expert Parallelism`（Wide-EP）是面向大规模 MoE serving 的部署模式：Attention 按数据并行副本组织，各副本独立承载请求和 KV Cache；MoE experts 则跨更宽的 EP group 分布，由多个 Attention DP ranks 共享 expert pool。

```text
Attention DP replicas（请求/KV 独立）
        -> MoE dispatch All-to-All
Wide EP group（experts 分布）
        -> MoE combine All-to-All
返回原 Attention replica
```

它不只是把 `EP size` 数字调大，而是把 [[DP Attention]] 与 [[Expert Parallelism]] 组合成适合 MLA/MoE 的 serving 拓扑。

## 它解决什么问题

- MLA latent KV/projection 不像传统多头 KV 那样适合普通 TP；TP 可能在 ranks 间形成重复 cache，限制有效 batch。
- Attention 数据并行让各 replica 保存不同请求的 KV Cache，提高部署总有效 KV 容量。
- Experts 按 EP 分布，避免每个 Attention replica 完整复制庞大的 expert 权重。

## 核心机制

- 请求被分配到某个 Attention DP rank，该 rank 维护请求生命周期和 KV Cache。
- 到 MoE 层时，Router 选择逻辑 experts，token activation 被 dispatch 到 expert 所在 EP rank。
- Expert 计算完成后 combine 回 token 原属的 Attention rank，继续后续层。
- vLLM 可选 DeepEP、FlashInfer 或 NCCL-based AllGather-ReduceScatter 等 All-to-All backend。
- vLLM 官方单机示例用 `TP=1, DP=8, EP=8` 部署 DeepSeek-V3：Attention 权重在8个 DP ranks 间复制，expert 权重跨8卡分片。若 `TP>1`，Attention 在每个 DP engine 内走 TP，而 EP size 自动为 `TP×DP`；例如 `TP2/DP4` 仍形成 EP8。

## 配套优化

- [[Dual Batch Overlap]]：用两个 microbatch 交错隐藏部分 dispatch/combine 等待。
- [[Expert Parallel Load Balancing]]：按真实流量动态调整逻辑 expert 到物理 rank 的 placement。
- [[PD分离]]：隔离 compute-bound prefill 对整个 EP group 的阻塞，并让两侧选择不同通信 backend。

## 与 AFD 的区别

Wide-EP 仍在一套模型执行拓扑里组合 Attention DP 与共享 Expert pool；[[Attention-FFN 分离]] 则把 Attention 与 FFN 变成独立服务，可使用不同 rank 数并独立扩缩。两者都发生 A/F 激活交换，但服务边界、生命周期和部署自由度不同。

## 关键权衡

- 优点：减少 MLA KV 重复、增大有效 batch，并分散 expert 权重。
- 代价：EP 越宽，All-to-All、同步和 expert imbalance 越突出。
- 单个 prefill 请求可能因 EP group 的层间 collective 协调拖慢其它 ranks。
- 更适合高并发总体吞吐；低负载下可能无法覆盖通信和同步成本。

## 相关实体

- [[../entities/vLLM]]
- [[../entities/DeepEP]]

## 相关来源

- [[../sources/vLLM Large Scale Serving DeepSeek @ 2.2k toksH200 with Wide-EP]]

## 相关概念

- [[Expert Parallelism]]
- [[DP Attention]]
- [[Dual Batch Overlap]]
- [[Expert Parallel Load Balancing]]
- [[PD分离]]
- [[Attention-FFN 分离]]
- [[MLA]]

## 研究备注

- `Wide` 没有脱离实现版本的统一阈值；应以 DP/EP topology、节点边界和通信 backend 描述具体部署。
