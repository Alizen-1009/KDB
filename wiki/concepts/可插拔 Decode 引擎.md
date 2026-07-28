---
type: concept
topic: 推理服务
sources: 1
updated: 2026-07-26
---

# 可插拔 Decode 引擎

## 定义

可插拔 Decode 引擎是 [[PD分离]] 带来的进一步模块化：Prefill、Serving API、缓存和调度可以由通用框架保留，而不同流量在 Prefill 完成后交给面向不同吞吐—延迟目标的 Decode backend。

```text
共享 Serving / Prefill Layer
├── 通用流量 -> Native Decode（高并发、聚合吞吐）
└── 延迟流量 -> Specialized Decode（单用户 token speed）
```

## 它解决什么问题

- 一个通用引擎很难同时在高并发吞吐、单用户低延迟、模型覆盖和功能成熟度上都取最优。
- 专用 Decode runtime 若独立重建 API、调度、Prefix Cache 和运维体系，接入成本很高。
- 公共 Connector 允许专用引擎只优化 Decode，同时复用成熟的 Serving/Prefill 生态。

## vLLM + TileRT 实例

- stock vLLM 提供 API、Router 之后的 Prefill、调度、Chunked Prefill 与 Prefix Caching。
- TileRT Connector 只认领携带目标 Decode node 标记的请求；其他请求走 native disaggregation path。
- `MultiConnector` 允许两个 Decode pools 共享同一个 Prefill 实例，甚至共享一次 forward batch。
- Connector 是纯 `kv_producer`，不接管 Prefill 调度和采样。

## 跨引擎状态交接

异构 PD 交接不仅要传 KV Cache，还要对齐 Decode 继续执行所需的全部状态：

- 压缩 Attention KV；
- 稀疏 Attention index cache；
- Token 位置和少量执行元数据；
- MTP draft-layer KV；
- Dtype、分页/物理布局和模型配置。

到达 Decode node 后，状态需要转换为目标引擎原生 layout，再注入 live engine。相较同构 PD，异构 PD 的兼容和版本耦合更强。

## 路由与背压

专用 Decode pool 容量可能很窄。例如 TileRT 0.1.5 每 node 仅允许一个 in-flight request，因此 Router 必须：

- 只把延迟敏感且模型受支持的请求发往专用 pool；
- 做 gated dispatch 和 back-pressure；
- 容量不足时排队或回退到 native pool；
- 保持客户端 API 不变，使切换只是路由决策。

## 与集群 Serving 框架的关系

TileRT Connector 解决“一个请求如何从 Prefill engine 交给专用 Decode engine”。NVIDIA Dynamo、Ray Serve LLM 等更偏集群级请求路由、autoscaling、worker pool 与 SLA 编排；它们可管理这类 backend，但不是同层替代。

## 关键权衡

- 优点：用专用 Decode 优化单用户延迟，又不放弃通用 serving 生态。
- 代价：状态传输、layout conversion、模型版本和 speculative state 对齐更复杂。
- State extraction 与后台 RDMA 可和下一轮 Prefill 重叠，但仍消耗 buffer、带宽和 GPU/网络资源。
- 多 Decode pools 会增加路由、容量规划、故障处理和一致性测试成本。

## 相关实体

- [[../entities/TileRT]]
- [[../entities/vLLM]]
- [[../entities/Nvidia Dynamo]]

## 相关来源

- [[../sources/vLLM x TileRT Specialized Decode for Latency-Critical Serving]]

## 相关概念

- [[PD分离]]
- [[KV Cache]]
- [[Speculative Decoding]]
- [[通信-计算重叠]]
- [[缓存感知路由]]

## 研究备注

- 后续需比较 native vLLM、TileRT 与其他 specialized decode engines 在相同 SLA、并发和输入输出长度下的吞吐—延迟前沿。
