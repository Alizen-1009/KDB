---
type: concept
topic: KV Cache
sources: 16
updated: 2026-06-21
---

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
- 在 `SGLang` 的 [[RadixAttention]] 语境中，KV cache 还会被保留在 radix tree 中，用于 program 分支、共享系统 prompt 或生成结果前缀的运行时复用
- 在 [[FlashMLA]] 语境中，KV cache 以 MLA latent cache 与 paged cache metadata 形式参与 decode kernel；优化重点不只是少存，还包括如何按 block table、变长序列和 Split-KV 高效读取历史状态
- 在 [[../entities/RTP-LLM]] 语境中，KV cache 被纳入 [[分层 KV Cache]] 与调度系统：Master 通过统一哈希映射跟踪跨 worker 前缀命中，并在 GPU、本地 CPU、远程 CPU、分布式存储之间按层级复用缓存
- 在 [[Decode Context Parallel]] 语境中，KV cache 会在既有 TP group 内进一步沿 context/token 维切分，用来减少长上下文 decode 下的重复缓存；`vllm并行策略之DCP` 进一步说明，vLLM 口径下采用 interleaved 存储，单个 request 的第 `n` 个 token KV 放到 `n % cp_world_size` 对应的 DCP rank。

## 与递归线性状态的区别

KV Cache 按 token/page 保存显式历史，已保存的每行 K/V 可以作为独立前缀记录复用。GDN/KDA 等线性注意力则用 [[线性注意力递归状态]] 压缩历史：Conv State 保存短卷积窗口，矩阵状态保存推进到当前边界后的长期聚合结果。后者大小不随上下文线性增长，但不能自然提供任意 token 边界回退，因此 Prefix Cache 需要 [[递归状态 Prefix Caching|递归状态 checkpoint]]。

混合 MLA/KDA 模型会同时维护 Token KV Pool 与按请求分配的递归状态槽；“统一缓存”通常指逻辑前缀树与生命周期协调，不表示两类状态具有相同物理布局或恢复粒度。

## 跨引擎 PD 状态交接

在 vLLM Prefill + TileRT Decode 这类 [[可插拔 Decode 引擎|异构 PD]] 中，交接对象不只普通 KV Cache，还包括压缩 KV、sparse-attention index cache、MTP draft-layer KV 和执行元数据。目标引擎需要把状态转换为自己的原生 layout 后注入 live engine。

来源使用 Mooncake/NIXL 以 RDMA one-sided writes 写入预注册 GPU buffers，避免 Host staging 和中间序列化；但 state extraction、本地 staging copy、layout conversion 与网络传输仍有实际成本。

## 推理阶段视角

- `Prefill`：对整个 prompt 并行编码，通常更容易接近 compute-bound
- `Generation / Decode`：逐 token 生成，attention 更容易变成 memory-bound
- 因此 KV Cache 一方面减少重算，另一方面也会把大量推理优化重新聚焦到显存容量和带宽上

## Attention 结构与算术强度

- 标准 `MHA` 在 decode 时每个 query head 都读取自己的历史 `K/V`，`QK` 和 `PV` 的 FLOPs 与 KV cache 读流量都随 `H * S * D_head` 线性增长；`FP16/BF16` 下粗略只有约 `1 FLOP/byte`，所以常见瓶颈是 HBM 带宽。
- `GQA/MQA` 减少 `KV heads`，让多个 query heads 共享同一份 `K/V`，本质上是在提高 KV cache 的复用率；它会提高 arithmetic intensity，但通常仍要结合 batch、context length 和 kernel 实现判断是否摆脱 memory-bound。
- [[MLA]] 把历史 `K/V` 缓存在更低维的 latent 表示里，显著减少每个历史 token 的 HBM 读取量；如果 attention score/value 路径仍保留较多按 head 计算，且 kernel 能有效复用 latent cache，就可能让 decode attention 的瓶颈从读 KV cache 转向实际计算吞吐。
- 在共享或压缩 KV 的结构中，位置编码还会影响 cache 语义：若直接对共享 KV 做 RoPE，V 也可能携带位置相位；[[MLA]] 通过额外小型 RoPE K cache 解耦，[[CSA-HCA|CSA/HCA]] 则需要处理压缩块位置标定和输出逆旋转。

## Prefill 计算量口径

设输入 hidden 为 `[B, S, D]`，query heads 为 `H_q`，`D_h = D / H_q`，KV heads 为 `H_kv`，`r = H_kv / H_q`。以下用 MACs 计数；若按 FLOPs 计数，矩阵乘加通常约乘 `2`。

- `MHA`：Q/K/V/O 线性层约 `4 * B * S * D^2` MACs；causal prefill 的 `QK^T + P @ V` 约 `B * D * S * (S + 1)` MACs，近似 `B * D * S^2`；单层 KV cache 写入约 `2 * B * S * D` 个元素。
- `GQA/MQA`：Q 与 O 线性层不变，K/V 线性层随 `H_kv` 缩小，总线性层约 `(2 + 2r) * B * S * D^2` MACs；但 `QK^T + P @ V` 仍需对每个 query head 计算，算术量仍近似 `B * D * S^2` MACs；单层 KV cache 写入降为 `2 * B * S * D * r` 个元素。
- `MLA`：若走 latent/absorbed 路径，设 KV latent rank 为 `C`，额外 RoPE key 维度为 `R`，attention core 约 `0.5 * B * S * (S + 1) * H_q * (2C + R)` MACs；cache 写入约 `B * S * (C + R)` 个元素。它在 prefill 下未必少算，核心收益更常体现在 decode 阶段少读历史 KV cache。

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
- `分层 KV Cache`：生产系统层缓存管理，目标是把可复用 KV 从单 GPU 扩展到跨节点和跨存储层级

这些机制都和 KV Cache 有关，但作用层级不同：模型结构、请求复用、路由调度和跨层级存储管理不能混为一谈。

## Layout 与访问模式

- 在 `PagedAttention` 代码走读中，K cache 与 V cache layout 不同，是因为两段计算的访存方向不同。
- QK 阶段需要对每个历史 token 读取 K 的 head_dim chunk，并与当前 query 做 dot product，因此 K cache 常排成便于 thread group 协作读取 16B chunk 的形式。
- PV 阶段需要对固定 head_dim 沿历史 token 维度做 `softmax(scores) @ V`，因此 V cache 更偏向让一个 block 内同一 head_dim 的多个 token 连续。
- 从 `PAv1` 的 CUDA 并行视角看，K cache layout 还服务于让 warp 内不同 thread group 处理不同历史 token，并由代表线程并行写入 shared memory 中的 logits；V cache layout 则更强调读路径，与 shared memory 中的 softmax 权重做局部 dot 后再归约。
- 这里的 `layout` 指逻辑张量维度如何映射到物理内存顺序；它不改变 attention 数学，只影响 kernel 的访存效率。
- DCP 的 interleaved KV cache 是另一层分布式 layout：它决定 token KV 被哪个 DCP rank 持有。该布局会影响 decode 读取、本地 partial attention、`lse` 合并，以及 prefill / prefix cache 写入时的 cache manager 逻辑。

## 相关实体

- [[../entities/vLLM]]
- [[../entities/TensorRT-LLM]]
- [[../entities/Nvidia Dynamo]]
- [[../entities/SGLang]]
- [[../entities/RTP-LLM]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构]]
- [[../sources/斯坦福CS336 Lecture 10 - Inference systems and optimization]]
- [[../sources/美团一面：请介绍 vLLM PageAttention]]
- [[../sources/多卡GPU监控与SM执行模型面试整理]]
- [[../sources/Gemma 4：Drafter 详解]]
- [[../sources/SGLang：LLM推理引擎发展新方向]]
- [[../sources/PageAttention代码走读]]
- [[../sources/vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现]]
- [[../sources/MLA与DP Attention面试整理]]
- [[../sources/DeepSeekV4中RoPE设计解析]]
- [[../sources/陈巍：DeepSeek 开源Day（1）-FlashMLA 深入分析（收录于：DeepSeek技术详解系列）]]
- [[../sources/RTP-LLM]]
- [[../sources/vllm并行策略之DCP(Decode Context Parallel)]]
- [[../sources/SGLang的KDA管理与Prefix Cache难题]]
- [[../sources/vLLM x TileRT Specialized Decode for Latency-Critical Serving]]

## 相关概念

- [[Continuous Batching]]
- [[Prefix Caching]]
- [[RadixAttention]]
- [[PagedAttention]]
- [[Speculative Decoding]]
- [[Shared KV Cache]]
- [[MTP Drafter]]
- [[CUDA Kernel]]
- [[MLA]]
- [[FlashMLA]]
- [[Decode Context Parallel]]
- [[DP Attention]]
- [[CSA-HCA|CSA/HCA]]
- [[分层 KV Cache]]
- [[线性注意力递归状态]]
- [[递归状态 Prefix Caching]]
- [[可插拔 Decode 引擎]]

## 研究备注

- 后续可补不同模型结构下的存储开销与 offloading 策略，以及 `FP8/量化 KV Cache` 的质量边界
- 新增来源补强了一个更运行时的视角：KV cache 不只是“存历史 K/V”，还涉及逻辑块、物理块和映射表如何配合动态增长
- 从硬件指标看，decode 阶段常因 KV cache 读写变成 memory-bound：这时可能出现 `DRAM Bandwidth` 较高、`Tensor Active` 不高，而不是单纯的低精度或 Tensor Core 退化问题。
- 需要区分两类“共享”：Gemma 4 目标模型内部的 `Shared KV Cache` 是层间共享；MTP drafter 里的 KV cache sharing 是 target model 与 drafter 之间的复用。
- SGLang 的 `RadixAttention` 又是另一类运行时共享：它不是模型结构内部共享，而是通过前缀树保存并复用请求或 program 分支的历史 KV。
- RTP-LLM 的来源补入了跨节点和多级存储视角，但具体缓存评分公式、同步周期与远程缓存成本应按论文/源码复核。
