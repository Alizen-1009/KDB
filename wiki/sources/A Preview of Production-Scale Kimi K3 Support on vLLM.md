---
type: source
source_kind: 文章
topic: 推理服务
updated: 2026-08-02
---

# A Preview of Production-Scale Kimi K3 Support on vLLM

## 来源信息

- 标题：A Preview of Production-Scale Kimi K3 Support on vLLM
- 作者：vLLM Team
- 日期：2026-07-22
- 类型：官方博客 / 预览
- 原始文件：[[../../raw/articles/A Preview of Production-Scale Kimi K3 Support on vLLM.md]]
- 原始链接：https://vllm.ai/blog/2026-07-22-kimi-k3-preview

## 2-3 条核心摘要

- Kimi K3 的混合 KDA–MLA 架构迫使 vLLM 扩展核心 Prefix Cache 抽象：KDA 需要在精确 token 边界恢复 Matrix State 与 ShortConv State，而 MLA/full attention 仍按 token 保存 Paged KV。vLLM 将 physical block size、scheduler alignment 与 prefix-match unit 解耦，使大物理状态块内部仍可进行细粒度 Prefix 匹配。
- vLLM 对 K3 的支持覆盖整个热路径，而不只是 Attention Kernel：Prefill 集成 FlashKDA/FLA，Decode 融合 ShortConv、KDA Update、Gate 和 Norm；MLA 为 Prefill/Decode 采用不同 Fusion；AttnRes、SiTU MXFP4 MoE、Tool Calling、多模态以及 NVIDIA/AMD 路径也进入集成或验证阶段。
- 该文是权重发布前的 Preview，必须区分“已集成”“验证中”和“生产可用”：文章称 Non-disaggregated Serving 已工作，但 FlashKDA Backend Selection、数值验证、PD/Offload Prefix Cache、Expert Parallelism 和 Vendor Validation 仍处于最终验证阶段。

## 值得关注的论断

- 混合 Cache 的合法命中必须对应同一个逻辑 Prefix：MLA KV、KDA Matrix State 与 ShortConv State需对同一个 `num_computed_tokens` 有效；如果 KDA checkpoint 不存在，即使 MLA KV 匹配更长也不能直接从该边界恢复。
- Prefix Hash 可在大 Physical State Block 内使用更细粒度的 chained hash。命中后，Cached KDA State必须先 Copy-on-Write 到请求私有 Running State，避免继续生成时原地污染其他请求共享的 Prefix Snapshot；Same-step reuse也要等状态复制安全后再开放。
- Prefill 与 Decode 的最佳融合模式不同：Decode 可让 MLA Gate Projection 在 Side Stream 与主 Attention Path 并行；Prefill 中 Multi-stream 未必划算，因此把 Sigmoid 与 Elementwise Multiply 融入 Gate Projection Epilogue。
- 文章报告 MXFP4 SiTU MoE 已在 16-GPU `DP16+EP16` 配置上完成 Optimized Backend 选择与 Correctness Check，但没有给出端到端吞吐或延迟数字，不能解读为性能 Benchmark。

## 关键概念

- [[../concepts/递归状态 Prefix Caching]]
- [[../concepts/KDA]]
- [[../concepts/LatentMoE]]
- [[../concepts/Attention Residuals]]

## 相关实体

- [[../entities/vLLM]]
- [[../entities/vLLM Team]]
- [[../entities/Kimi K3]]
- [[../entities/FlashKDA]]
- [[../entities/Moonshot AI]]

## 与现有 wiki 的关系

- 为现有 [[../concepts/递归状态 Prefix Caching]] 增加 vLLM 一手实现：明确区分物理分配粒度、调度对齐边界和 Prefix Hash 粒度，并补充联合 `num_computed_tokens`、chained hash、Copy-on-Write 与 Same-step 可见性约束。
- 为 [[../entities/vLLM]]、[[../entities/Kimi K3]] 和 [[../entities/FlashKDA]] 补充 K3 Day-0 集成状态与 Prefill/Decode Kernel 分工。
- 未发现机制层直接冲突；但本文是 2026-07-22 Preview，部分 Release Branch 描述与其他 commit/recipe 的可见代码和默认开关可能不同，必须保留版本与验证状态。

## 待确认

- Release Branch 所称“Input Projection + Causal Convolution Fusion”的具体 Kernel 边界，与当前公开主分支中独立 Prefill Conv 调用的版本对应关系。
- PD/Offload Prefix Cache 已集成但仍在验证，与部分部署 Recipe 默认关闭 Decode Prefix Caching 之间的能力/默认配置边界。
- 后续正式发布版本的 FlashKDA Backend Selection、NVIDIA/AMD 支持矩阵以及端到端性能数据。
