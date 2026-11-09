---
type: entity
entity_type: 项目
topic: KV Cache
sources: 1
updated: 2026-11-09
---

# LMCache

## 一句话说明

面向大模型推理的 KV cache 管理项目，支持将可复用缓存与引擎内生命周期解耦。

## 类型

- 项目

## 核心信息

- 文章描述其 MP 模式通过独立服务让多个推理副本共享主机内存缓存，以减少副本之间的缓存碎片和重复 prefill。
- 文章称其支持多种存储后端与 Mooncake Transfer Engine 集成；具体版本、后端及跨引擎数据兼容性待核实。

## 相关概念

- [[../concepts/分层 KV Cache]]
- [[../concepts/Prefix Caching]]

## 相关来源

- [[../sources/KV Cache 研习社 · 深潜  LMCache、Mooncake、FlexKV——三条路线谁在改写 KV 的命运？]]

## 冲突与备注

- 引擎无关不等于任意模型、引擎或版本的缓存布局均可直接互换；性能数字见来源中的实验边界。
