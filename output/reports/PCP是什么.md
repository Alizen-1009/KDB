# PCP 是什么

## 背景

`PCP` 指 **Prefill Context Parallel（预填充上下文并行）**：在 Prefill 阶段，把同一个长 Prompt 沿 sequence/token 维切给多个 GPU 并行计算。

相关页面：[[../../wiki/concepts/Prefill Context Parallel|Prefill Context Parallel]]、[[../../wiki/concepts/Decode Context Parallel|Decode Context Parallel]]、[[../../wiki/concepts/Ring Attention|Ring Attention]]、[[../../wiki/concepts/DeepSpeed Ulysses|DeepSpeed Ulysses]]。

## 核心观点

PCP 与 DCP 都会切一个请求的长度，但切分发生的阶段和主要目标不同：

- PCP：切 Prefill 中大量 Query/Key/Value token，主要分摊长 Prompt 的 Attention 计算与激活，改善 TTFT。
- DCP：切 Decode 中已经生成的历史 KV Cache，主要分摊 KV 容量与读取带宽。

## Rank 布局

假设 Prompt 长度为 `S`，`PCP=4`，初始可抽象为：

```text
rank 0: token [0, S/4)
rank 1: token [S/4, S/2)
rank 2: token [S/2, 3S/4)
rank 3: token [3S/4, S)

每 rank: Q/K/V [B, S/4, H, D]
```

逐 token 的 Norm、Residual、MLP 可以在本地 token shard 上计算。但 Attention 不能只看本地 KV：每个本地 Query 必须看到因果范围内来自其他 ranks 的 K/V。因此 PCP 需要额外通信。

## vLLM 官方定义的两条 PCP 路线

### Partial Query、Full KV（当前源码可见路径）

每个 rank 只计算自己的 Q token chunk，但通过 PCP group AllGather 收集完整 K/V，然后使用本地 Q 对完整 KV 计算 Attention：

```text
rank r: Q_r, K_r, V_r
              ↓ AllGather K/V
rank r: Q_r, K_all, V_all
              ↓
输出 O_r
```

vLLM `main@1ad5182` 的 MRV2 MLA PCP 在 `vllm/model_executor/layers/attention/pcp.py` 中对 partitioned prefill latent-KV/cache inputs 执行 `pcp_group.all_gather`；sampling 前也会 AllGather hidden states。该路线实现直接、适合仍放得下完整 KV 的中等长输入，但 PCP ranks 会持有完整 KV，不能按 PCP 倍数降低 KV 峰值。

### Partial Query、Partial KV（Ring Attention 方向）

每个 rank 固定本地 Q chunk，让 K/V blocks 沿 ring 轮转：

```text
rank r: Q_r 固定
round 0: Q_r × KV_r
round 1: Q_r × KV_{r-1}
...
```

每收到一个 KV block，就更新 Online Softmax 状态。它不需要每 rank 同时持有完整 KV，更适合极长输入；代价是多轮 P2P、依赖链和 causal load imbalance。vLLM 官方部署文档列出了该方向，但同时注明两条 PCP 路线仍处于 active development。

### Ulysses 的位置

Ulysses 是通用 sequence/context parallel 算法：通过 Q/K/V All-to-All 把 `[S/P, H]` 转置为 `[S, H/P]`，本地计算后再转置回来。但 vLLM 官方 PCP 文档没有提到 Ulysses，`main@1ad5182` 仓库也没有 Ulysses 实现；因此它不能算当前 vLLM PCP backend。它还要求 heads/KV-heads 可切，和当前 MLA-only、KV-head 很少的 PCP 支持面并不自然匹配。

## 与 DP Attention、DCP 的对比

| 并行方式 | 切什么 | 一个请求是否跨 rank | 主要优化目标 |
| --- | --- | --- | --- |
| DP Attention | 请求/batch | 通常否 | 多请求吞吐、总 KV 容量 |
| PCP | Prefill Prompt token | 是 | 超长 Prompt 计算、激活、TTFT |
| DCP | Decode 历史 KV token | 是 | KV Cache 容量、HBM 带宽、长上下文 decode |

8 卡上的直觉例子：

```text
DPA=8: 8 张卡分别处理不同请求
PCP=8: 8 张卡共同 Prefill 一个长 Prompt
DCP=8: 8 张卡共同读取一个请求的历史 KV 来生成下一个 token
```

## vLLM 官方定义下的 `TP=8, PCP=4`

已核对 vLLM 官方仓库 `main` commit [`1ad5182`](https://github.com/vllm-project/vllm/commit/1ad5182ba95a6f1de23b537d57b860082912b28e)（2026-07-29）。当前 `ParallelConfig` 明确写明 PCP 会扩张 process world size，并在初始化时计算：

```text
world_size = PP × TP × PCP
```

来源：[PCP 字段定义](https://github.com/vllm-project/vllm/blob/1ad5182ba95a6f1de23b537d57b860082912b28e/vllm/config/parallel.py#L126-L131)、[world size 计算](https://github.com/vllm-project/vllm/blob/1ad5182ba95a6f1de23b537d57b860082912b28e/vllm/config/parallel.py#L831-L837)。

因此 `PP=1, DP=1, TP=8, PCP=4` 需要 **32 个 workers/GPUs**，不是在8张卡里把 TP8 折叠成 `2 heads × 4 sequence`。

官方 rank 顺序是：

```text
ExternalDP × DP × PP × PCP × TP
```

对于 TP8/PCP4：

```text
PCP rank 0: global ranks  0..7   = TP group 0
PCP rank 1: global ranks  8..15  = TP group 1
PCP rank 2: global ranks 16..23  = TP group 2
PCP rank 3: global ranks 24..31  = TP group 3

PCP group for tp_rank 0: [0, 8, 16, 24]
PCP group for tp_rank 1: [1, 9, 17, 25]
...
PCP group for tp_rank 7: [7, 15, 23, 31]
```

来源：[官方 rank layout 与 group construction](https://github.com/vllm-project/vllm/blob/1ad5182ba95a6f1de23b537d57b860082912b28e/vllm/distributed/parallel_state.py#L1812-L1881)。

若暂用8-head MHA 的简化直觉，每个 TP8 group 先切成8个 TP head shards；每个四成员 PCP group固定一个 TP shard，再沿 prefill sequence 分工。因此是 **8个 PCP groups，每组4 ranks**，而不是2个 PCP groups、每组一半 heads。不过当前 Model Runner V2 的 PCP 代码只支持 MLA，普通8-head MHA只是帮助理解 group topology，不代表当前已支持配置。

当前代码还采用 DualChunkSwap：PCP4 把 prefill 分成8块，四个 PCP ranks 分别拿 `(0,7)`、`(1,6)`、`(2,5)`、`(3,4)`，以平衡 causal Attention 工作量。来源：[PCPManager](https://github.com/vllm-project/vllm/blob/1ad5182ba95a6f1de23b537d57b860082912b28e/vllm/v1/worker/gpu/pcp_manager.py#L195-L215)。

配置边界：当前 PCP 不支持 `DP>1`；PCP 开启后，DCP size 只能为 `1`、`PCP` 或 `TP×PCP`。来源：[官方配置校验](https://github.com/vllm-project/vllm/blob/1ad5182ba95a6f1de23b537d57b860082912b28e/vllm/config/parallel.py#L524-L539)。

## 与 Chunked Prefill

二者不相同：

```text
Chunked Prefill：调度器把长 Prompt 分成多个时间上串行的 chunks
PCP：多个 GPU 并行计算当前 prefill chunk 的不同 token shards
```

Chunked Prefill 主要控制调度公平性、峰值和对 Decode 的干扰；PCP 主要缩短单个超长 chunk/request 的并行计算时间。二者可以组合，但 chunk 太小会让 PCP 缺少足够工作量。

## 工程权衡

- 长 Prompt、Attention 计算占主导且互联较快时更有价值。
- 短 Prompt 下通信可能超过计算收益。
- Causal Attention 中靠后的 Query chunk 工作更多，需要 zigzag/striped 等负载均衡。
- Prefill 产生的 KV Cache 必须按后续 Decode 所需布局落盘；如果 Decode 使用 DCP，runtime 需要完成对应的 cache placement/reorganization。

## 版本边界与待核实

本报告已用 vLLM 官方 `main` commit `1ad5182` 确认 CLI、world-size 公式、group topology 及 PCP/DCP 组合约束；但这不代表任意已发布版本都具备同样能力。官方部署文档仍将 partial-Q/full-KV 与 Ring Attention 两条 PCP 策略标为 active development，当前 MRV2 代码也只支持 MLA，并限制 PP、DP、多模态、LoRA、投机解码与部分 CUDA Graph 模式。生产部署必须对照实际安装的 vLLM commit/version。
