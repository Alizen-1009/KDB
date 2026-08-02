---
type: entity
entity_type: 组织
topic: 推理服务
sources: 4
updated: 2026-04-23
---

# vLLM Team

## 一句话说明

`vLLM` 项目的官方维护团队，在官方博客和设计更新中常作为集体作者出现。

## 类型

- 组织 / 开源项目团队

## 核心信息

- 主要代表 `vLLM` 项目的官方工程实现与架构演进视角
- 在当前知识库里，`vLLM Team` 主要以来源作者身份出现，关联 `PagedAttention`、`MRV2` 等官方设计说明
- Wide-EP 官方博客进一步记录其在 DeepSeek/MLA 大规模 serving 上对 DPA+EP、DBO、EPLB、DeepEP 与 PD 分离的工程组合
- TileRT 合作博客展示 V1 Connector 作为跨引擎组合边界：无需修改 vLLM，即可让共享 Prefill pool 同时连接 native 与 specialized Decode pools
- Kimi K3 Preview展示团队将Hybrid Attention支持下沉到核心缓存抽象：解耦Physical State Block、Scheduler Alignment与Prefix-match Unit，并联合处理MLA KV、KDA Conv/Matrix State、Copy-on-Write和PD Cache Transfer。

## 相关概念

- [[PagedAttention]]
- [[持久批处理]]
- [[Wide Expert Parallelism]]
- [[Dual Batch Overlap]]
- [[Expert Parallel Load Balancing]]
- [[可插拔 Decode 引擎]]
- [[递归状态 Prefix Caching]]
- [[KDA]]

## 相关来源

- [[../sources/Model Runner V2 A Modular and Faster Core for vLLM]]
- [[../sources/vLLM Large Scale Serving DeepSeek @ 2.2k toksH200 with Wide-EP]]
- [[../sources/vLLM x TileRT Specialized Decode for Latency-Critical Serving]]
- [[../sources/A Preview of Production-Scale Kimi K3 Support on vLLM]]

## 冲突与备注

- 当前条目主要用于承接来源作者链接，后续若有更多官方博客或设计文档进入知识库，可再扩展
- K3来源是正式权重发布前Preview；文中的集成状态与验证状态应按后续release/commit继续校准。
