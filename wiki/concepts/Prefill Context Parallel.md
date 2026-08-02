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

逐 token 的 Norm、MLP、Residual 可本地计算；Attention 中每个本地 Query 仍需看到全序列 K/V。通用算法可以采用 [[Ring Attention]] 或 [[DeepSpeed Ulysses]]，但这不代表 vLLM 已实现两者。

vLLM 官方 `main` commit `1ad5182` 的部署文档只定义两条 PCP 路线：

1. `partial query, full key/value`：各 rank 计算局部 Q，AllGather 完整 K/V 后计算本地 Q 对应输出；
2. `partial query, partial key/value`：每 rank 只持有局部 Q/K/V，用 Ring Attention 逐块发送 K/V。

官方文档没有提到 Ulysses，源码全仓也没有 `Ulysses` 实现标识。当前 MRV2 MLA PCP 代码实际可见的是第一条 AllGather 路径：`pcp.py` 对各 rank 的 partitioned prefill latent-KV/cache inputs 做 `pcp_group.all_gather`，并在 sampling 前 AllGather hidden states。Ring 路线在官方文档中作为超长输入策略提出，但两条策略都仍标为 active development。

所以 [[DeepSpeed Ulysses]] 应理解为 Microsoft DeepSpeed 面向长序列训练的通用 Context/Sequence Parallel 算法，不是 DeepSeek 技术，也不是当前 vLLM PCP backend。当前 DeepSpeed HF 实现可以在 `SP > H_kv` 时复制 KV heads 来适配部分 GQA/MQA，但 Q heads 仍必须能被 SP 整除；对 MLA/MQA 这类 KV 维极窄的模型，这会削弱 KV 侧收益。Ring 不依赖把 KV heads 分给每个 CP rank，因此更常出现在极长上下文和少 KV-head 的 PCP 讨论中。

## vLLM 官方拓扑：PCP 与 TP 正交

已按 vLLM 官方仓库 `main` commit `1ad5182`（2026-07-29）核对：当前实现明确规定 PCP **扩张 process world size**，而不是像无 PCP 时的 DCP 那样复用 TP ranks。

```text
world_size = PP × PCP × TP
layout order = ExternalDP × DP × PP × PCP × TP
```

因此在 `PP=1, DP=1, TP=8, PCP=4` 时，需要 `32` 个 workers/GPUs，而不是 8 个。global ranks 的逻辑布局为：

```text
PCP rank 0: global ranks  0..7   （一个 TP8 group）
PCP rank 1: global ranks  8..15  （一个 TP8 group）
PCP rank 2: global ranks 16..23  （一个 TP8 group）
PCP rank 3: global ranks 24..31  （一个 TP8 group）

固定 TP rank 0 的 PCP group: [0, 8, 16, 24]
固定 TP rank 1 的 PCP group: [1, 9, 17, 25]
...
固定 TP rank 7 的 PCP group: [7, 15, 23, 31]
```

所以若暂用“TP 沿 head 分片”的简化 MHA 直觉，每个四成员 PCP group 只对应一个 `H/8` 的 TP head shard，再把这个 shard 的 prefill sequence computation 分给 4 个 PCP ranks；它不是“两组 PCP、每组一半 heads”。当前 Model Runner V2 的实际 PCP 代码只支持 MLA，不能把该简化 head 映射直接当成所有模型的已支持行为。

当前实现还使用 `DualChunkSwap` 做 causal 负载均衡：`PCP=4` 将一次 prefill 切成 8 个 chunks，rank 0 取首尾 chunks `0,7`，rank 1 取 `1,6`，rank 2 取 `2,5`，rank 3 取中间 `3,4`，而不是每 rank 只拿一个连续四分之一。

官方配置约束还包括：PCP 暂不支持与 `DP>1` 组合；启用 PCP 后，DCP size 只能取 `1`、`PCP` 或 `TP×PCP`，即禁用 DCP、让 DCP 跨 PCP 轴，或让 DCP 跨完整 TP×PCP block。若启用 EP，MoE experts 的并行组按 `TP×PCP×DP` ranks 组织。

官方依据：

- [ParallelConfig：PCP 扩张 world size](https://github.com/vllm-project/vllm/blob/1ad5182ba95a6f1de23b537d57b860082912b28e/vllm/config/parallel.py#L126-L131)
- [world_size = PP × TP × PCP](https://github.com/vllm-project/vllm/blob/1ad5182ba95a6f1de23b537d57b860082912b28e/vllm/config/parallel.py#L831-L837)
- [rank layout 与 TP/PCP group 构造](https://github.com/vllm-project/vllm/blob/1ad5182ba95a6f1de23b537d57b860082912b28e/vllm/distributed/parallel_state.py#L1812-L1881)
- [PCP/DCP 配置约束](https://github.com/vllm-project/vllm/blob/1ad5182ba95a6f1de23b537d57b860082912b28e/vllm/config/parallel.py#L524-L539)
- [MRV2 PCP 支持边界与 DualChunkSwap](https://github.com/vllm-project/vllm/blob/1ad5182ba95a6f1de23b537d57b860082912b28e/vllm/v1/worker/gpu/pcp_manager.py#L125-L161)

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
- vLLM 官方 `main` commit `1ad5182` 已有 `--prefill-context-parallel-size/-pcp`、独立 PCP process group 与 MRV2 runtime 实现，但官方部署文档仍称两类 PCP 策略处于 active development；当前代码只支持 MLA，并存在 PP、DP、多模态、LoRA、投机解码和 CUDA Graph 等限制，不能等同于通用稳定支持。

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

- 官方 `main` commit `1ad5182` 已确认参数名、world-size 公式、group construction 与 PCP/DCP 组合约束；正式 release 版本是否包含同样实现，仍需按部署版本核对。
- 官方部署文档描述两条 PCP 数据流：partial-Q/full-KV 的 AllGather 路线，以及 partial-Q/partial-KV 的 Ring Attention 路线；两者仍标为 active development。
