---
type: source
source_kind: 文章
topic: 推理服务
updated: 2026-07-26
---

# vLLM Large Scale Serving DeepSeek @ 2.2k toksH200 with Wide-EP

## 来源信息

- 标题：vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP
- 作者：vLLM Team
- 日期：2025-12-17
- 类型：官方博客
- 原始文件：[[../../raw/articles/vLLM Large Scale Serving DeepSeek @ 2.2k toksH200 with Wide-EP.md]]
- 原始链接：https://vllm.ai/blog/2025-12-17-large-scale-serving

## 2-3 条核心摘要

- vLLM 的 Wide-EP 将 Attention 数据并行与 MoE Expert Parallelism 结合：各 DP rank 独立处理请求并维护 MLA KV Cache，experts 则跨宽 EP group 分布，使系统避免普通 TP 对 MLA latent projection/cache 的低效复制，同时共享更大的 expert pool。
- 高 EP degree 会放大 dispatch/combine All-to-All、rank 同步和真实流量下的 expert imbalance。vLLM 用 DeepEP 等通信 backend、Dual Batch Overlap（DBO）和 Expert Parallel Load Balancing（EPLB）分别处理通信路径、跨 microbatch 重叠与动态 expert placement。
- PD 分离在 Wide-EP 下尤其有价值：一个 compute-bound prefill 请求可能拖慢整个 EP group；分离后 Prefill/Decode 可使用不同资源和 DeepEP 高吞吐/低延迟路径。

## 值得关注的论断

- DBO 把 batch 切为两个 microbatch，在一个 microbatch 等待 dispatch/combine 时推进另一个；它不打破同一 microbatch 的 `dispatch -> expert compute -> combine` 依赖。
- EPLB 不修改模型 Router 选择的逻辑 expert，而是根据滑动窗口负载统计调整逻辑 expert 到物理 rank 的映射，并通过权重 shuffle 让新 placement 生效。
- 官方博客引用社区 benchmark：CoreWeave H200 多节点、InfiniBand 与 ConnectX-7 环境达到持续 `2.2k tok/s/H200`，高于此前约 `1.5k tok/s/GPU`。收益来自 DBO 和多项 kernel/bug-fix 的组合，不能归因于单独开启 Wide-EP。
- 正文未完整列出模型版本、输入输出分布、并发、SLA、DP/EP size 等条件，数字需沿 llm-d 原始 benchmark 进一步核实。

## 关键概念

- [[../concepts/Wide Expert Parallelism]]
- [[../concepts/Dual Batch Overlap]]
- [[../concepts/Expert Parallel Load Balancing]]
- [[../concepts/Expert Parallelism]]
- [[../concepts/DP Attention]]
- [[../concepts/通信-计算重叠]]
- [[../concepts/PD分离]]

## 相关实体

- [[../entities/vLLM]]
- [[../entities/vLLM Team]]
- [[../entities/DeepEP]]
- [[../entities/Nvidia Dynamo]]

## 与现有 wiki 的关系

- 新增 Wide-EP、DBO 与 EPLB 页面，补足 DeepSeek/MLA 大规模 serving 的并行组合和运行时优化。
- 更新 vLLM、DP Attention、Expert Parallelism、通信-计算重叠、PD 分离与 AFD 页面。
- 未发现直接冲突；该来源补充了现有 `DPA + EP` 研究备注的具体 vLLM 实现。

## 待确认

- `2.2k tok/s/H200` 的完整 benchmark 参数和吞吐统计口径。
- DBO 阈值、线程/CUDA Graph 行为和 All-to-All backend 支持随 vLLM 版本的变化。
- EPLB 权重 shuffle、冗余 experts 与在线请求一致性的精确实现。
