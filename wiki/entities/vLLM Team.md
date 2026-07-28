---
type: entity
entity_type: 组织
topic: 推理服务
sources: 3
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

## 相关概念

- [[PagedAttention]]
- [[持久批处理]]
- [[Wide Expert Parallelism]]
- [[Dual Batch Overlap]]
- [[Expert Parallel Load Balancing]]
- [[可插拔 Decode 引擎]]

## 相关来源

- [[../sources/Model Runner V2 A Modular and Faster Core for vLLM]]
- [[../sources/vLLM Large Scale Serving DeepSeek @ 2.2k toksH200 with Wide-EP]]
- [[../sources/vLLM x TileRT Specialized Decode for Latency-Critical Serving]]

## 冲突与备注

- 当前条目主要用于承接来源作者链接，后续若有更多官方博客或设计文档进入知识库，可再扩展
