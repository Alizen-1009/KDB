# Kimi K3 为什么常采用 TP8 部署

## 核心结论

Kimi K3 采用 TP8，主要不是因为“有8个 heads”，而是四个因素共同决定：

1. **2.8T 总权重的显存压力**；
2. **104B active parameters 的单 token 计算量**；
3. **KDA、AttnRes、LatentMoE 的分布式 kernel 设计本身利用 TP collective**；
4. **8-GPU 节点高速互联和 TP8/TP16 专用 kernel shape**。

TP8 是当前单实例低延迟/可部署性的实用起点，不是唯一生产拓扑。高并发场景仍可能使用 TP×DP、TP+EP、DP+EP 或 PD 分离。

相关页面：[[../../wiki/entities/Kimi K3|Kimi K3]]、[[../../wiki/concepts/Tensor Parallelism|Tensor Parallelism]]、[[../../wiki/concepts/LatentMoE|LatentMoE]]、[[../../wiki/concepts/MLA|MLA]]。

## 1. “MLA只有一个KV head”不等于整个模型只有一个head

Kimi K3 不是纯 MLA 模型，而是：

```text
69层 KDA + 24层 Gated MLA
总 Q/attention heads = 96
```

MLA 在 absorbed decode/cache 视角下接近一个共享 latent KV head，但 Query 侧仍有96个 heads，Q/O projection、hidden/MLP/LatentMoE 权重也仍可沿 TP 切分。`TP8` 时每 rank 大约负责12个 Q heads。

对69个 KDA层，head-wise recurrent states也可以沿96个 heads分到8个 TP ranks。只有24个 Gated MLA层的 latent KV cache在当前纯 TP 路径下会因单 latent head而复制。

因此准确账本是：

```text
TP8节省/分摊：模型权重、Q heads、KDA heads、projection与active compute
TP8浪费：当前24层Gated MLA的单-latent KV cache存在8路复制
```

这确实是浪费，但只是整个显存/计算账本中的一个子项。K3模型权重约3T级，且MLA cache本身已经压缩；在中短context或较小batch下，模型权重和active compute通常比这部分重复cache更先决定能否部署。到超长context/大batch时，复制代价会放大。

当前 vLLM K3 MLA实现明确不支持DCP/PCP，所以暂时不能用DCP8消除这部分复制。官方recipe通过FP8 KV、限制不同硬件的max-model-len，以及生产中的DP/EP和P/D分离做权衡；未来若K3 backend接入DCP，才可能同时保留TP8权重分片并消除MLA context复制。

## 2. Sparse activation 不等于只加载104B参数

官方规格：

```text
总参数：2.8T
激活参数：约104.2B/token
Routed experts：896
Top-k：16
Shared experts：2
```

MoE 每个 token 只计算16个 routed experts，但下一批 token 可能选择其它 experts，因此 serving 实例仍需让整个 expert pool 的权重保持可访问。不能按104B active parameters估算模型常驻显存。

Kimi K3 从 SFT 起对 expert weights 做 MXFP4、expert input activation 做 MXFP8；但 attention projection、LatentMoE projection、shared experts 和 router 保持更高精度。vLLM K3 recipe 在 checkpoint 正式发布前估计：

```text
2.8T × 0.5 byte × 1.2 headroom ≈ 1.68 TB
```

这是部署估算而非最终实测，但足以解释单卡或少量 GPU 无法承载。TP8 可以把大量权重分散到一个8-GPU高速互联域。

## 3. TP8 分摊每 token 的大计算

K3 每 token 激活约104B参数，即使权重量化，矩阵计算和权重读取仍然很重。TP8 让同一个请求的层内 GEMM、attention heads 和投影由8张卡协同执行，偏向降低单请求 latency。

如果改成纯 DP，多个 GPU 各处理不同请求，可以提高吞吐，但每个 DP replica 必须承载 replicated attention/dense/shared路径以及自己的 expert shard。vLLM recipe 因此把 DEP 的最低规模设为16 GPUs，而 TP/TP+EP 从8 GPUs起步。

## 4. 模型维度天然适合8路切分

```text
attention heads 96 / 8 = 12 heads/rank
hidden size 7168 / 8 = 896
latent dimension 3584 / 8 = 448
expert hidden 3072 / 8 = 384
```

这些整除关系让 Q/head、hidden、LatentMoE projection 和 expert intermediate shard 具有规整 shape。它不证明 TP8 必然最优，但减少 padding 与不规则 shard，是 TP8 成为自然工程点的重要条件。

## 5. K3 serving kernel 主动利用 TP

官方技术报告并不只是把 TP 当作“放下权重”的工具。

### Block AttnRes

Prefill 时，若每个 TP rank 都物化完整 block representation，会产生冗余显存和 I/O。K3 将 TP AllReduce 拆为：

```text
ReduceScatter
→ sequence-sharded Block AttnRes kernel
→ AllGather
```

每个 token 的 block representation 只在一个 rank 上物化。

### Stable LatentMoE

官方报告称：

- latent weight matrices 跨 ranks 分片；
- output AllGather 融入 GEMM epilogue；
- 通信与 shared-expert computation 重叠。

因此 TP group 是 K3 专用 fusion/overlap 设计的一部分，而非纯粹通用 vLLM 配置。

### vLLM 专用 kernel

当前 vLLM K3 NVIDIA 实现中的 SM100 latent-MoE tail fusion 明确只支持：

```text
TP size ∈ {8, 16}
hidden=7168
latent=3584
BF16
SM100
```

TP8 因而能命中专门优化的 collective + RMSNorm + latent up-projection 路径；其它 TP degree 可能回退或无法启用该 fusion。

## 6. 为什么不直接 DP8/EP8

纯 `DP8/TP1/EP8` 对许多 DeepSeek/MLA MoE 是常见高吞吐路线，但 K3 更大：

- 总权重约为3T级；
- non-expert 模块、shared experts、KDA/MLA/AttnRes 与 vision encoder 仍有 replicated 部分；
- 每个 DP rank 还需要 cache、workspace、CUDA Graph 和 runtime buffer；
- 当前 vLLM recipe 将 DEP 最低规模设为16 GPUs。

所以固定8张高显存 GPU 时，TP8/TP8+EP 往往比 DP8 更容易先把模型放下并获得单请求性能。资源扩大后，DP/EP 才更适合扩展吞吐。

## 7. TP8 的代价

- 每层 TP collective 频繁，跨节点 TP 对 latency 很敏感；
- 请求级并行宽度低于 DP；
- K3 的24层 Gated MLA latent KV 在纯 TP 下可能复制；
- 当前 vLLM K3 MLA 明确不支持 DCP/PCP，暂不能用 `DCP8` 消除这部分复制；
- 小 batch 下某些本地 GEMM 变得更窄，需要专用 skinny-GEMM kernel 才能高效。

K3 有69层 KDA与24层 Gated MLA；KDA recurrent state固定大小且可按heads切分，MLA cache才随sequence增长。其混合 cache 行为不能完全套用纯 MLA 模型。

## 8. 实际部署不是只有TP8

vLLM recipe 给出多种策略：

- 单节点 TP；
- 多节点 TP；
- TP + EP；
- DP + EP；
- TP × DP；
- Prefill/Decode 分离。

其中 PD recipe 使用：

```text
Prefill：TEP，TP=8
Decode：DEP，TP=1
```

这说明 TP8 更适合 Prefill/单实例计算与权重分摊；高并发 Decode 可以通过更宽 DP/EP 提高吞吐。TP8 是组件，不是所有阶段都必须一致。

## 官方依据

- [MoonshotAI Kimi K3 README](https://github.com/MoonshotAI/Kimi-K3)
- [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)，尤其 §5.4.2 High-Performance Kernels
- [vLLM Kimi K3 recipe](https://github.com/vllm-project/recipes/blob/72626067968e70856b79a2e4841edea5d6846012/models/moonshotai/Kimi-K3.yaml)
- [vLLM K3 latent-MoE tail TP8/16约束](https://github.com/vllm-project/vllm/blob/1ad5182ba95a6f1de23b537d57b860082912b28e/vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py#L20-L28)
- [vLLM K3 MLA 暂不支持 DCP/PCP](https://github.com/vllm-project/vllm/blob/1ad5182ba95a6f1de23b537d57b860082912b28e/vllm/models/kimi_k3/nvidia/mla.py#L1-L29)

## 一句话

> Kimi K3 上 TP8 首先是为了把约3T级权重与104B级 active compute 分到一个8-GPU高速域，同时命中围绕 TP collective、sequence sharding 和 LatentMoE fusion设计的专用 kernel；它不是由 head 数单独决定，也不是高并发生产部署的唯一答案。
