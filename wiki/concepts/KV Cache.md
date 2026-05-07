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
- 在 Gemma 4 `MTP Drafter` 中，KV cache 还可以跨 target model 与 drafter 复用：drafter 不必完整处理 prompt 建立自己的 KV，而是通过 cross-attention 使用目标模型已计算好的 KV cache

## 推理阶段视角

- `Prefill`：对整个 prompt 并行编码，通常更容易接近 compute-bound
- `Generation / Decode`：逐 token 生成，attention 更容易变成 memory-bound
- 因此 KV Cache 一方面减少重算，另一方面也会把大量推理优化重新聚焦到显存容量和带宽上

## 大小估算

- 如果忽略分页、对齐和实现细节，`KV Cache` 大小通常近似与 `batch size`、`sequence length`、`num_layers`、`num_kv_heads`、`head_dim` 和 `dtype bytes` 线性相关
- 一个常见的粗略公式是：`KV Cache bytes ≈ 2 * B * S * L * H_kv * D_head * bytes_per_elem`
- 前面的 `2` 对应同时保存 `K` 和 `V`

## 关键权衡

- 计算复杂度下降，但显存占用显著上升
- 缓存越大越能减少重算，但也越容易触发显存压力和淘汰策略
- 它把 decode 的主要成本从“重算整段序列”转成“读历史缓存并追加新状态”，因此布局、分页与缓存精度会直接影响吞吐

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
- [[../sources/多卡GPU监控与SM执行模型面试整理]]
- [[../sources/Gemma 4：Drafter 详解]]

## 相关概念

- [[Continuous Batching]]
- [[Prefix Caching]]
- [[PagedAttention]]
- [[Speculative Decoding]]
- [[Shared KV Cache]]
- [[MTP Drafter]]

## 研究备注

- 后续可补不同模型结构下的存储开销与 offloading 策略，以及 `FP8/量化 KV Cache` 的质量边界
- 新增来源补强了一个更运行时的视角：KV cache 不只是“存历史 K/V”，还涉及逻辑块、物理块和映射表如何配合动态增长
- 从硬件指标看，decode 阶段常因 KV cache 读写变成 memory-bound：这时可能出现 `DRAM Bandwidth` 较高、`Tensor Active` 不高，而不是单纯的低精度或 Tensor Core 退化问题。
- 需要区分两类“共享”：Gemma 4 目标模型内部的 `Shared KV Cache` 是层间共享；MTP drafter 里的 KV cache sharing 是 target model 与 drafter 之间的复用。
