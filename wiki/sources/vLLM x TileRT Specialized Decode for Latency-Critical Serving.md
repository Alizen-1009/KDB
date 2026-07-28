---
type: source
source_kind: 文章
topic: 推理服务
updated: 2026-07-26
---

# vLLM x TileRT Specialized Decode for Latency-Critical Serving

## 来源信息

- 标题：vLLM x TileRT: Specialized Decode for Latency-Critical Serving
- 作者：TileRT Team
- 日期：2026-07-14
- 类型：官方合作博客
- 原始文件：[[../../raw/articles/vLLM x TileRT Specialized Decode for Latency-Critical Serving.md]]
- 原始链接：https://vllm.ai/blog/2026-07-14-vllm-tilert-pd

## 2-3 条核心摘要

- PD 分离让 Decode 侧不仅能独立扩缩容，也能替换为不同推理引擎。该方案保留 stock vLLM 的 API、调度、Chunked Prefill、Prefix Caching 与 Prefill，只把延迟敏感流量的 Decode 交给 TileRT；普通流量仍进入 native vLLM Decode。
- 集成完全建立在 vLLM V1 公共扩展面上：TileRT 实现 `KVConnectorBase_V1`，在 `MultiConnector` 下组合，并通过 `kv_connector_module_path` 加载，不 fork、不 patch vLLM。Connector 只认领带 TileRT 标记的请求，对其余请求是 no-op，因此两个 Decode pools 可共享一个 Prefill 实例甚至同一个 forward batch。
- Prefill 后交接的不只有普通 KV：压缩 Attention KV、稀疏 Attention index cache、少量元数据和 MTP draft-layer KV 通过 Mooncake/NIXL 的 RDMA one-sided writes 进入预注册 GPU buffers；到达后转换为 TileRT 原生布局并注入运行中的 Decode engine。

## 值得关注的论断

- TileRT 与 native vLLM Decode 面向吞吐—延迟前沿的不同位置：前者追求单用户 token speed，适合 agent loop、交互式 coding 和实时语音；后者仍是高并发 batching、聚合吞吐和广模型/功能覆盖的默认选择。
- State extraction 在 Prefill forward window 内、cache blocks 回收前复制到 staging buffer，实际网络发送由后台 sender 完成，以便与下一轮 Prefill 重叠；这不是零成本，仍会消耗本地复制、buffer 和网络资源并可能产生背压。
- TileRT 要从首个 Decode step 启用 MTP，因此 Prefill 必须准备并传递 draft-layer KV。跨引擎交接要求模型权重、token 位置、dtype、KV/sparse index layout 与 speculative 配置一致。
- TileRT 0.1.5 每个 Decode node 同时只服务一个 in-flight request，当前支持 GLM-5/5.1 与 DeepSeek-V3.2；这些是版本限制，不是永久能力边界。

## 关键概念

- [[../concepts/可插拔 Decode 引擎]]
- [[../concepts/PD分离]]
- [[../concepts/KV Cache]]
- [[../concepts/Speculative Decoding]]
- [[../concepts/通信-计算重叠]]

## 相关实体

- [[../entities/TileRT]]
- [[../entities/vLLM]]
- [[../entities/vLLM Team]]

## 与现有 wiki 的关系

- 新增 TileRT 实体与可插拔 Decode 引擎概念，补足 PD 分离的跨引擎组合价值。
- 更新 PD 分离、vLLM、KV Cache、Speculative Decoding、通信-计算重叠和 vLLM Team 页面。
- 未发现直接冲突；该来源把现有“PD 资源隔离”扩展为“共享 serving/prefill 层下的异构 Decode backend”。

## 待确认

- TileRT 专用 Decode kernel 的内部优化机制不在本文范围内，需单独读取 repo/技术文档。
- Connector 的状态 layout、稀疏 index 格式、MTP draft KV 兼容和升级策略需按具体源码版本核实。
- 图中 GLM-5.1/B200 的 token speed 不从柱高转录；完整 benchmark 条件需读取原始评测资料。
