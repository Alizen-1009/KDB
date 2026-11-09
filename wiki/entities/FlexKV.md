---
type: entity
entity_type: 项目
topic: KV Cache
sources: 0
updated:
---

# FlexKV

## 一句话说明

面向大模型推理的 KV cache 管理项目，文章将其重点描述为多级存储、异步搬运和跨节点复用。

## 类型

- 项目

## 核心信息

- 文章描述 GPU、CPU 内存、SSD 与远端存储的分层路径，及异步读写、预取和淘汰；具体可用层级和行为取决于配置与版本。
- 文章称其可通过推理框架集成，并与 Mooncake Transfer Engine 结合；接口与支持矩阵待核实。
- 文章声称有腾讯云生产环境验证；具体案例、负载、实验基线及指标尚未独立核实。

## 相关概念

- [[../concepts/分层 KV Cache]]
- [[../concepts/Prefix Caching]]

## 相关来源

- [[../sources/KV Cache 研习社 · 深潜  LMCache、Mooncake、FlexKV——三条路线谁在改写 KV 的命运？]]

## 冲突与备注

- 与 SGLang HiCache、vLLM KV offloading 存在功能重叠，不能以是否“工业可用”直接二分；各自实际采用规模及具体兼容性待一手证据核实。
