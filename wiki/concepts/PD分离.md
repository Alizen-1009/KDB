# PD分离

## 定义

将 Prefill 和 Decode 拆分到不同引擎或不同硬件资源上的推理架构，也常称为 `Prefill Decode Disaggregation`。

## 它解决什么问题

- 减少 prefill 与 decode 在同一套资源上互相争用
- 让不同阶段分别采用更适合自身负载的优化与并行策略

## 核心机制

- Prefill 引擎负责长输入处理、KV 构建和首 token
- Decode 引擎负责后续自回归生成
- 通过 NVLink 或 InfiniBand 等互联传递 KV Cache

## 关键权衡

- 获得更好的资源专用化和调优空间
- 代价是更高的系统复杂度、缓存传输成本和运维投入

## 相关实体

- [[../entities/Nvidia Dynamo]]
- [[../entities/TensorRT-LLM]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]

## 相关概念

- [[KV Cache]]
- [[缓存感知路由]]
- [[Tensor Parallelism]]

## 研究备注

- 条件聚合比全量解耦更实用，后续值得单独拆成子概念页
