---
type: entity
entity_type: 模型
topic: 模型架构
updated: 2026-07-30
sources: 0
---

# Kimi K3

## 一句话说明

[[Moonshot AI]] 发布的 2.8T 参数原生多模态 MoE 模型，采用 69 层 KDA、24 层 Gated MLA、Attention Residuals 与 Stable LatentMoE，支持最长 1M context。

## 核心信息

- 总参数量 2.8T，单 token 激活约 104.2B 参数；稀疏激活减少每 token 计算，但部署仍需让 896 个 routed experts 的权重可被路由访问。
- 模型宽度 7168，96 attention heads；Stable LatentMoE latent dimension 3584、单 expert hidden dimension 3072，每 token 激活 16/896 routed experts，并有 2 个 shared experts。
- MoE expert 权重在部署感知后训练中使用 MXFP4、输入 activation 使用 MXFP8；attention projection、latent MoE projection、shared experts 和 router 保持更高精度。
- KDA 与 Gated MLA 形成混合 cache：KDA recurrent state 固定大小，MLA KV cache 随 context 增长；官方 serving 系统联合管理两类 cache。
- “MLA只有一个 latent KV head”不等于模型只有一个 attention head：K3 有96个 Q/attention heads，且69/93层是 KDA。TP8 可把 Q heads、KDA head states、dense/projection权重与 active compute 分摊到8 ranks；当前24个 Gated MLA层的 latent KV cache会在纯TP路径下复制，这是局部代价。

## 为什么常见 TP8

- 模型总权重极大；vLLM K3 recipe 对 MXFP4 checkpoint 的运行时权重与余量估计约 1.68TB。即使每 token 只激活部分 experts，也不能只加载 104B active parameters。
- TP8 把 dense/attention/投影以及非 EP 路径的权重和计算分到一个常见 8-GPU 高速互联域，降低单 rank 权重压力与单请求计算时延。
- 主要维度可整除8：96 heads→每 rank 12 heads，7168→896，3584→448，3072→384。
- 官方技术报告中的 Block AttnRes prefill 用 TP collective 拆成 ReduceScatter + AllGather 来实现 activation sequence parallel；Stable LatentMoE projection 也跨 ranks 分片，并把 output AllGather 融入 GEMM epilogue。
- vLLM 当前 K3 SM100 latent-MoE tail fusion 只接受 TP8/TP16，说明专用 kernel 对这些部署 shape 有明确特化。

TP8 不是唯一拓扑。vLLM recipe 同时提供 multi-node TP、TP+EP、DP+EP、TP×DP 和 PD 分离；其PD示例明确采用`Prefill TP8/TEP`与`Decode TP1/DEP`：前者用TP分摊单请求的KDA/投影计算和head-sharded state，后者用Attention DP+EP扩大高并发Decode吞吐。按当前公开shape估算，69层KDA FP32 recurrent state在TP8时约51.75MiB/request/rank，而TP1请求需在单rank承担完整数百MiB级state，因此拓扑还会改变单请求状态带宽与容量。当前 vLLM K3 MLA 路径不支持 DCP/PCP，不能直接套用普通 MLA 的 `TP8+DCP8` 建议。

## 参数构成粗估

按92个 MoE层、latent `ℓ=3584`、expert intermediate `m=3072`、896 experts和 gated FFN的 `3ℓm` 估算：

- routed experts约 `2.723T`，占2.78T总参数约 `97.9%`；
- 加上2个shared experts、latent上下投影与router后，MoE相关约 `2.740T / 98.6%`；
- 剩余约 `39.8B / 1.4%` 包含KDA/MLA Attention、embedding/LM head、401M vision encoder、第一层dense FFN、norm等，不能全部算作Attention。

每token active账本中，16个routed experts约 `48.6B`，加shared experts、latent projection和router后，MoE active path粗估约 `66.1B / 63.4%`；其余Attention和dense模块约 `38.1B / 36.6%`。因此存储几乎由expert权重主导，但运行计算并非98%都在MoE。

纯TP8时，每rank保留全部896个逻辑experts的约1/8 tensor shard；TP8+EP8时，则更接近每rank放置112个experts并通过All-to-All路由token。两者每rankexpert存储都约为全局1/8，但通信模式不同。

## KDA Prefix Cache

K3生产Serving将KDA固定大小状态与MLA Paged KV联合管理：Prefix命中必须在同一token边界同时恢复MLA KV与所有KDA状态组。Physical page、fine-grained prefix hash与稀疏KDA checkpoint采用不同粒度；命中快照复制为请求私有running state后才能继续更新。详见 [[../../output/reports/Kimi K3的KDA部署与Prefix Cache|Kimi K3的KDA部署与Prefix Cache]]。

## 后续阅读重点

在KDA与Prefix Cache之后，最值得继续推导的是Stable LatentMoE、Quantile Balancing与MoonEP的算法—训练—部署闭环；阅读优先级见 [[../../output/reports/Kimi K3技术报告后续阅读重点|Kimi K3技术报告后续阅读重点]]。

## 相关概念

- [[KDA]]
- [[Chunked Gated Delta Rule]]
- [[线性注意力递归状态]]
- [[MLA]]
- [[LatentMoE]]
- [[Attention Residuals]]
- [[Tensor Parallelism]]
- [[Expert Parallelism]]
- [[MoonEP]]
- [[PD分离]]

## 官方资料

- [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)
- [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)
- [vLLM Kimi K3 recipe](https://github.com/vllm-project/recipes/blob/72626067968e70856b79a2e4841edea5d6846012/models/moonshotai/Kimi-K3.yaml)

## 待核实

- vLLM recipe 在权重正式发布前把约 1.68TB 标为估算值；实际 safetensors、加载后显存、CUDA Graph、KV/state cache 和 workspace 需按最终 checkpoint 与硬件实测。
- TP8、TP16、TEP、DEP 与 PD 分离的最优点取决于 GPU 显存、节点互联、batch/context 分布与 SLO，不存在脱离硬件和负载的固定答案。
