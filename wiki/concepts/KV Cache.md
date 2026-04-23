# KV Cache

## 定义

在自回归 Transformer 推理中缓存历史 token 的 Key/Value 张量，以避免每生成一个新 token 时重算整段上下文注意力。

## 它解决什么问题

- 降低自回归生成阶段的重复计算成本
- 让长上下文推理在生产环境中具备可接受的时延和成本

## 核心机制

- 对已经完成 prefill 的 token 保存 K/V 表示
- 后续 decode 只为新 token 计算查询并与历史缓存交互
- 将单步注意力的重算模式从“重读整段序列”转向“复用历史状态”

## 推理阶段视角

- `Prefill`：对整个 prompt 并行编码，通常更容易接近 compute-bound
- `Generation / Decode`：逐 token 生成，attention 更容易变成 memory-bound
- 因此 KV Cache 一方面减少重算，另一方面也会把大量推理优化重新聚焦到显存容量和带宽上

## 关键权衡

- 计算复杂度下降，但显存占用显著上升
- 缓存越大越能减少重算，但也越容易触发显存压力和淘汰策略

## 分层理解

- `Shared KV Cache`：模型内部层间共享 K/V，目标是减少层级缓存占用与重复投影
- `Prefix Caching`：服务层跨请求复用公共前缀，目标是减少重复 prefill
- `缓存感知路由`：请求分发层优化，目标是让前两类缓存收益真正落地

这三者都和 KV Cache 有关，但作用层级分别是模型层、请求层和系统调度层。

## 相关实体

- [[../entities/vLLM]]
- [[../entities/TensorRT-LLM]]
- [[../entities/Nvidia Dynamo]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构]]
- [[../sources/斯坦福CS336 Lecture 10 - Inference systems and optimization]]
- [[../sources/美团一面：请介绍 vLLM PageAttention]]

## 相关概念

- [[Continuous Batching]]
- [[Prefix Caching]]
- [[PagedAttention]]
- [[Speculative Decoding]]
- [[Shared KV Cache]]

## 研究备注

- 后续可补充 KV Cache 的大小估算公式、不同模型结构下的存储开销与 offloading 策略
- 新增来源补强了一个更运行时的视角：KV cache 不只是“存历史 K/V”，还涉及逻辑块、物理块和映射表如何配合动态增长
