# MLA 模型常见部署拓扑

## 背景

MLA 只定义 Attention/KV Cache 结构，不单独决定部署拓扑。实际选择还取决于模型是否为 MoE、权重能否放下、请求并发、上下文长度，以及目标是吞吐、单请求延迟还是 TTFT。

相关页面：[[../../wiki/concepts/MLA|MLA]]、[[../../wiki/concepts/DP Attention|DP Attention]]、[[../../wiki/concepts/Decode Context Parallel|Decode Context Parallel]]、[[../../wiki/concepts/Wide Expert Parallelism|Wide Expert Parallelism]]、[[../../wiki/concepts/Prefill Context Parallel|Prefill Context Parallel]]。

## 最常见：Attention DP + Wide EP

对 DeepSeek 这类 `MLA + MoE` 模型，vLLM 官方推荐路线之一是：

```text
Attention：Data Parallel
MoE：Expert Parallel
```

8卡单机典型配置：

```text
TP=1, DP=8, EP=8, DCP=1
```

每张 GPU：

- 是一个独立 Attention DP rank，承载不同请求并维护独立 MLA KV Cache；
- 复制 Attention/dense 权重；
- 只持有一部分 experts；
- 进入 MoE 层时，token 在 EP8 group 内 dispatch/combine。

vLLM 当前无需单独配置 EP size；启用 `--enable-expert-parallel` 后，在无 PCP 时：

```text
EP size = TP × DP
```

官方8卡示例：

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel
```

来源：[vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)、[Data Parallel Deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)。

该拓扑适合高并发吞吐：8个 Attention replicas 可以同时承载不同请求，MLA latent KV 不会因 TP 在同一请求内重复。

## 权重或计算需要 TP：DP × TP + EP

若 Attention/dense/shared 权重无法在单卡放下，或希望用 TP 降低单请求计算时延，可使用混合拓扑。固定8卡例如：

```text
TP=2, DP=4, EP=8
```

- 4个 Attention DP engines，各自内部使用 TP2；
- expert layers 跨全部 `TP×DP=8` ranks 做 EP；
- 每个请求通常落到一个 DP engine，再由该 engine 的2个 TP ranks协同执行。

MLA 在 TP 下可能复制单-head/latent KV；若 backend/model 支持，可加：

```text
DCP=2
```

形成 `TP=2, DCP=2`，让每个请求的历史 KV 在该二成员 TP group 内沿 context 分片。DCP 不增加卡数。

## 长上下文 Decode 优先：TP + DCP

若目标不是最大并发，而是让单个超长请求跨多卡保存/读取 KV，可以减少 DP、增大 TP/DCP。8卡极端例子：

```text
TP=8, DP=1, DCP=8, EP=8
```

- 一个 Attention request 跨8个 ranks；
- dense/attention 权重按 TP8 切分；
- 单个 MLA latent KV context 按 DCP8 切分；
- experts 仍跨8 ranks 做 EP。

这消除纯 TP8 对单 KV head 的八路完整 cache 复制，但牺牲请求级并行度并增加 DCP 通信。vLLM 官方文档以 DeepSeek-R1 `TP8 + DCP8` 作为去除 MLA KV duplication 的案例。

来源：[vLLM Context Parallel Deployment](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)。

## 8卡上的连续权衡

| 配置 | Attention 请求并行 | 单请求 context 分片 | EP size | 偏向 |
| --- | ---: | ---: | ---: | --- |
| `DP8, TP1, DCP1` | 8组 | 1 | 8 | 高并发吞吐 |
| `DP4, TP2, DCP2` | 4组 | 2 | 8 | 平衡吞吐与长上下文 |
| `DP2, TP4, DCP4` | 2组 | 4 | 8 | 更长上下文/更低单卡 KV |
| `DP1, TP8, DCP8` | 1组 | 8 | 8 | 单请求容量优先 |

这些是逻辑拓扑示例，不代表每个模型/backend/version 均支持全部组合。

## 超长 Prefill：PCP

若瓶颈是超长 Prompt 的 TTFT，而非 Decode KV 容量，才考虑 PCP。当前 vLLM `main@1ad5182` 中 PCP：

- 与 TP 正交并扩张 world size：`world_size = PP × TP × PCP`；
- 当前 MRV2 只支持 MLA；
- 暂不支持 `DP>1` 和 `PP>1`；
- 当前源码是 partial-Q/full-KV AllGather 路径，Ring Attention 是官方 active-development 方向；
- 不属于当前常规 Wide-EP 生产配置的默认选项。

因此8卡 `DP8/TP1/EP8` 不能直接再打开 `PCP4` 而仍保持8卡；当前官方定义会增加 PCP workers，并且 PCP 暂不支持 DP>1。

## 大规模部署配套

- `DeepEP low-latency`：偏 Decode；
- `DeepEP high-throughput`：偏 Prefill；
- [[../../wiki/concepts/Dual Batch Overlap|Dual Batch Overlap]]：用双 microbatch 隐藏 EP dispatch/combine；
- [[../../wiki/concepts/Expert Parallel Load Balancing|Expert Parallel Load Balancing]]：缓解热门 expert 和 tail imbalance；
- [[../../wiki/concepts/PD分离|PD分离]]：把 compute-bound Prefill 与 latency-sensitive Decode 分到不同资源池。

## 结论

对常见 `MLA + MoE` serving，可按以下顺序选择：

1. 高并发默认从 `Attention DP + Wide EP` 开始；
2. 单卡放不下 dense/attention 权重时加入 TP；
3. TP 导致单-head latent KV 重复、或单请求 context 太长时，在 TP group 内加入 DCP；
4. 超长 Prompt TTFT 成为独立瓶颈时，再评估仍处于快速演进期的 PCP；
5. 大规模集群用 DBO、EPLB、DeepEP 和 PD 分离处理通信与阶段干扰。

## 待核实

具体可用组合取决于 vLLM release/commit、模型、attention backend、量化格式、GPU 显存与互联。尤其 `DP×TP×DCP×EP` 的混合配置应在目标版本上实际启动和压测，不能只从维度公式推导支持性。
