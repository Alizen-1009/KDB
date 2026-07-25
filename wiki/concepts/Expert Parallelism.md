---
type: concept
topic: 并行与分布式
sources: 3
updated: 2026-06-12
---

# Expert Parallelism

## 定义

`Expert Parallelism` / `EP` 是 MoE 模型里的专家并行策略：把不同 expert 放到不同 GPU 或不同 rank 上，让 token 根据 router 结果被发送到对应 expert 所在设备计算，再把输出收回原 token 位置。

口语里的“大 EP”通常不是一个严格算法名，而是指 `EP size` 开得很大：expert 被分散到很多 GPU / rank 上，系统主要靠跨卡 token dispatch 来摊薄 expert 权重与计算压力。

## 它解决什么问题

- MoE 模型总 expert 参数很大，单卡或单节点难以完整承载。
- 每个 token 只激活少数 expert，因此适合把“专家权重”按 expert 维度分散，而不是让所有 GPU 都复制全部 expert。
- 对推理 serving，EP 更偏提升系统总吞吐；它把不同 token 路由到不同专家设备，减少单个设备上的 expert 权重压力。

## 核心机制

1. Router 在每个 token 上选择 `top-k` expert。
2. 系统按 expert id 把 token 分桶。
3. 如果 expert 不在本 rank，就通过跨 GPU / 跨节点通信把 token 发过去。
4. 每个 rank 对本地 expert 收到的 token 做 grouped GEMM / expert FFN。
5. expert 输出按原 token 顺序 gather 回来，并按 routing weight 合并。

多卡通信形态常接近 `all-to-all`：不是像 [[Tensor Parallelism]] 那样每层固定 all-reduce partial activation，而是把 token 送到拥有对应 expert 的 rank。

## 为什么需要 all-to-all

EP 下每个 rank 都可能产生要发给任意 expert 的 token，而这些 expert 又分布在 EP group 的任意 rank 上。因此通信不是“一份张量复制给大家”，也不是“大家把 partial result 求和”，而是每个 rank 都按路由结果向其它 rank 发送不同 token 子集，同时接收其它 rank 发来的本地 expert token。

可以把一次 MoE 层看成两次 token 重排：

1. `dispatch all-to-all`：把本 rank 的 token 按目标 expert / 目标 rank 发出去。
2. `expert compute`：每个 rank 只计算自己持有的 expert。
3. `combine all-to-all`：把 expert 输出送回 token 原来的 rank / 顺序，再按 routing weight 合并。

所以 `all-to-all` 的本质是“按 token 的动态路由做多对多交换”。它搬运的通常是 activation，不是 expert 权重；expert 权重常驻在持有该 expert 的 rank 上。

## 和 TP / DPA 的关系

- [[Tensor Parallelism]]：按张量维度切 dense 权重和矩阵乘，常用于 attention、dense MLP 或单个 expert 内部；优点是降低单请求时延，代价是每层同步通信重。
- `Expert Parallelism`：按 expert 维度切 MoE 权重；优点是分散大量 expert 参数，代价是 token dispatch / all-to-all、负载不均和小 batch GEMM。
- [[DP Attention]]：常用于 DeepSeek/MLA 类 serving 的 attention 侧，避免 KV cache 在 TP rank 间重复；实际系统中可能和 EP 组合：attention/KV cache 用 DPA，MoE expert 用 EP。

## EP size 和 TP size

`EP size` 不必等于 `TP size`。它们切的是不同维度：

- `TP size` 切单个 dense 矩阵或单个 expert 内部的张量维度。
- `EP size` 切 expert 集合，让不同 rank 持有不同 expert。
- 如果同时启用二者，常见是“expert 先按 EP 分组放置；每个 expert 内部再按 TP 切分”，或者 attention / dense 层走 TP，MoE 层走 EP。

部署里经常看到 `EP size = TP size`，通常是因为单机单副本、只开一个并行组时配置最简单：同一组 GPU 既承担 TP 通信，也承担 MoE expert 分布。但这不是数学要求，也不是所有 serving 框架的固定规则。实际选择更受这些因素影响：总 GPU 数、是否启用 DP / DPA、expert 数是否能被 EP size 整除、节点内外网络拓扑、decode batch 大小、是否允许每个 expert 内部再 TP。

## 与 AFD 的组合

在 [[Attention-FFN 分离]] 中，EP 自然位于 FFN 服务内部：Attention 侧先通过 AFD connector 把 hidden states 交给 FFN 侧，FFN ranks 再按 expert id 执行 dispatch All-to-All、local expert compute 和 combine All-to-All，最后通过 connector 把 FFN output 返回 Attention 侧。

因此 AFD 与 EP 不是替代关系：AFD 决定 Attention/FFN 的服务边界和容量配比，EP 决定 FFN 侧的 expert 如何跨 rank 放置。组合后需要同时计算 A/F 激活往返、EP All-to-All、rank 映射和负载不均的成本。

## 关键权衡

- 优点：降低每张卡承载的 expert 权重，适合超大 MoE；在 batch 足够大时能提升总体吞吐。
- 代价：跨卡 token dispatch 成为核心开销；负载不均会导致热门 expert 拖慢整个 step。
- 大 EP 的风险：EP size 越大，单个 rank 收到的 token 可能越碎，跨节点 all-to-all 成本和 tail effect 更明显。
- 常见配套优化：expert load balancing、EPLB、grouped GEMM、dispatch/gather overlap、topology-aware placement。

## 面试口径

一句话：`EP` 是“专家放在不同卡上，token 去找专家”；`TP` 是“同一层矩阵被多卡一起算”；`DPA` 是“attention/KV cache 按数据副本组织，避免重复 cache”。

如果别人说“大 EP”，可以理解成：MoE 的 expert 被铺到很多卡上，系统把瓶颈从“单卡放不下/算不动 expert”转成“跨卡 token 路由、all-to-all 和负载均衡能不能扛住”。

关于 `EP size` 和 `TP size`：可以先回答“不必相等”。若面试语境是单机 MoE serving，`EP=TP` 常见；若有 DP/DPA、跨节点部署或 hybrid MoE parallel，二者经常会分开设计。

## 相关概念

- [[MoE]]
- [[Tensor Parallelism]]
- [[DP Attention]]
- [[集合通信]]
- [[Tail Effect]]
- [[Attention-FFN 分离]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/MLA与DP Attention面试整理]]
- [[../sources/vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署]]

## 研究备注

- 后续可结合 SGLang / vLLM / TensorRT-LLM 的具体参数名补充 `EP size`、`EPLB`、跨节点 expert placement 和 all-to-all overlap 的实现差异。
