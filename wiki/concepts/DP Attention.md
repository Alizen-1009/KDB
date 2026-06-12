# DP Attention

## 定义

`DP Attention` / `Data Parallelism Attention` 是推理系统里的并行策略：把 attention 计算按 data-parallel replica 组织，让不同 replica 处理不同请求或 batch，从而避免在某些模型结构下用 [[Tensor Parallelism]] 切 attention 时造成低效的 `KV Cache` 复制。

## 它解决什么问题

- 对 [[MLA]] / DeepSeek 类模型，`KV Cache` 已经被压缩成较小 latent 表示，attention 的 KV 维度不像传统 `MHA` 那样自然适合按 head 做 TP 切分。
- 如果仍按普通 TP 组织 attention，多个 GPU 可能各自保留相同或高度重复的 latent KV cache，导致显存被重复占用，限制 batch size 和吞吐。
- `DP Attention` 的目标是让 attention 侧更像多个独立 serving 副本：每个副本维护自己的请求与 KV cache，减少无意义复制，并让系统用更大的有效 batch 提升吞吐。

## 和 MLA 的关系

- [[MLA]] 是模型结构：通过低秩 KV 联合压缩减少每个 token 的缓存量。
- `DP Attention` 是系统并行策略：决定多 GPU serving 时 attention/KV cache 如何分布和调度。
- 二者经常一起出现，因为 MLA 模型的 KV head / latent cache 结构会削弱传统 TP attention 的收益，使 DP-style attention 更有吸引力。

## 关键权衡

- 优点：减少 attention 侧 KV cache 重复，提升可承载 batch size，尤其适合长上下文和 DeepSeek/MLA 类模型 serving。
- 代价：模型权重或 attention 相关权重可能需要在多个 DP replica 中复制；如果模型本身太大，还需要和 [[Expert Parallelism]]、MoE 并行或其他权重切分策略组合。
- 它不是 attention 数学本身的改动，不改变模型输出语义；收益主要来自显存布局、并行组织和调度效率。
- `DP Attention` 更偏多并发吞吐优化，而不是单请求 latency 优化；如果请求数少、batch 填不满、router/通信/调度开销较高，开启后不一定加速，甚至可能让单请求变慢。

## `dp attention = 8` 的含义

- 更准确地说，它表示有 `8` 个 attention data-parallel replica / 分片可以分别承载请求流，而不是每个 decode step 固定处理 `8` 个请求。
- 每个 replica 内部仍然可以通过 [[Continuous Batching]] 处理多个 active requests；因此系统一次处理的请求数可能小于、等于或大于 `8`。
- 如果只有一个长请求，它通常只落到其中一个 DP attention replica；不会因为 `dp attention = 8` 就自动把这一个请求切成 8 份。这个场景更接近 [[Decode Context Parallel]]。
- 如果有很多请求，router / scheduler 会把请求分散到 8 个 replica 上，让每个 replica 维护自己的 KV cache 和 batch。

## 和 DCP 的关系

- [[Decode Context Parallel]] 和 `DP Attention` 都在解决普通 TP 下 attention/KV cache 组织不理想的问题，但切分粒度不同。
- `DP Attention` 是请求/batch 级：不同 DP replica 处理不同请求，各自维护自己的 KV cache，常和 [[Expert Parallelism]] 组合服务 MoE。
- `DCP` 是单请求 context 级：同一个请求的历史 KV cache 被切到多个 GPU 上，attention 时需要合并各 shard 的 softmax 统计。
- 如果目标是提高整体 QPS、让不同请求分摊到不同 attention replica，优先想到 `DP Attention`；如果单个长上下文请求/长会话的 KV cache 在一个 TP group 内仍然重复或放不下，才更像 `DCP` 的问题。

## 相关实体

- [[../entities/SGLang]]
- [[../entities/DeepSeek-AI]]

## 相关概念

- [[MLA]]
- [[Tensor Parallelism]]
- [[Decode Context Parallel]]
- [[Expert Parallelism]]
- [[KV Cache]]
- [[Continuous Batching]]

## 相关来源

- [[../sources/MLA与DP Attention面试整理]]

## 研究备注

- `DP Attention` 在 SGLang 文档中也称 `DPA`，常以 `--dp-size` 与 `--enable-dp-attention` 这类 serving 配置出现；具体收益依赖模型结构、并行拓扑、batch、上下文长度和 attention backend。
- 后续可继续整理 DeepSeek/MLA serving 中 `DPA + EP` 的具体组合方式。
