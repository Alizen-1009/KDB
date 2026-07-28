---
type: entity
entity_type: 框架
topic: 推理服务
sources: 1
updated: 2026-07-26
---

# TileRT

## 一句话说明

`TileRT` 是面向延迟敏感推理的专用 Decode runtime，可通过 vLLM V1 Connector 与 stock vLLM Prefill 组合，在保留 vLLM serving 生态的同时优化单用户 token 生成速度。

## 类型

- 推理框架 / 专用 Decode 引擎

## 核心信息

- vLLM 负责 OpenAI-compatible API、调度、Chunked Prefill、Prefix Caching 和 Prefill；TileRT 只接管被路由到其 pool 的 Decode 流量。
- 集成使用 `KVConnectorBase_V1`、`MultiConnector` 与 `kv_connector_module_path`，不修改 vLLM 源码或内部 workers。
- Prefill state 通过 Mooncake 或 NIXL 以 RDMA one-sided writes 传入预注册 GPU buffers，再转换为 TileRT 原生布局并注入 live engine。
- TileRT 0.1.5 面向单用户低延迟：每个 Decode node 同时一个 in-flight request，支持 GLM-5/5.1、DeepSeek-V3.2，并可从首步启用 MTP。

## 相关概念

- [[可插拔 Decode 引擎]]
- [[PD分离]]
- [[KV Cache]]
- [[Speculative Decoding]]
- [[通信-计算重叠]]

## 相关来源

- [[../sources/vLLM x TileRT Specialized Decode for Latency-Critical Serving]]

## 冲突与备注

- TileRT 不是 vLLM、NVIDIA Dynamo 或 Ray Serve LLM 的直接同层替代：它在本文中是专用 Decode backend，而后两者偏集群 serving/编排层。
- 当前并发和模型覆盖是 0.1.5 的版本边界；精确性能与兼容性应按后续版本核实。
