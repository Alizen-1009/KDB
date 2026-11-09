---
type: entity
entity_type: 项目
topic: 推理服务
sources: 0
updated:
---

# Mooncake

## 一句话说明

以 KV cache 为中心、结合分布式缓存和 Prefill/Decode 分离的大模型服务项目。

## 类型

- 项目

## 核心信息

- 文章描述其利用集群资源建立 KV 池，并通过 Transfer Engine 在节点之间搬运 KV；调度和过载准入亦是系统层关注点。
- Transfer Engine 可作为其他组件的传输能力；PD 阶段交接与跨请求持久缓存复用是不同目的。

## 相关概念

- [[../concepts/PD分离]]
- [[../concepts/分层 KV Cache]]

## 相关来源

- [[../sources/KV Cache 研习社 · 深潜  LMCache、Mooncake、FlexKV——三条路线谁在改写 KV 的命运？]]

## 冲突与备注

- RDMA、GPU 直达、传输协议与吞吐依赖实际软硬件配置，不能将文章的零拷贝说法视为所有路径的保证。
- 真实负载与模拟极端场景的性能数据不可混为一谈，详见来源页。
