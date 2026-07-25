---
type: entity
entity_type: 框架
topic: 推理服务
sources: 1
updated: 2026-04-23
---

# Nvidia Dynamo

## 一句话说明

NVIDIA 面向推理服务优化的框架，用来整合缓存分层、KV offloading 与动态解耦等能力。

## 类型

- 项目 / 框架

## 核心信息

- 文中把 Dynamo 作为缓存分层和动态解耦的代表性系统提到。
- 通过 `KV Block Manager (KVBM)` 支持 KV 在不同存储层之间迁移。
- 在 `PD分离` 场景下，Dynamo 支持按请求条件在聚合路径与解耦路径之间动态选择。

## 相关概念

- [[KV Cache]]
- [[缓存感知路由]]
- [[PD分离]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]

## 冲突与备注

- Dynamo 的实现边界、商用形态和版本能力仍需结合官方资料补充
