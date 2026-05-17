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
- 代价：模型权重或 attention 相关权重可能需要在多个 DP replica 中复制；如果模型本身太大，还需要和 `Expert Parallelism`、MoE 并行或其他权重切分策略组合。
- 它不是 attention 数学本身的改动，不改变模型输出语义；收益主要来自显存布局、并行组织和调度效率。

## 相关实体

- [[../entities/SGLang]]
- [[../entities/DeepSeek-AI]]

## 相关概念

- [[MLA]]
- [[Tensor Parallelism]]
- [[KV Cache]]
- [[Continuous Batching]]

## 相关来源

- [[../sources/MLA与DP Attention面试整理]]

## 研究备注

- `DP Attention` 在 SGLang 文档中也称 `DPA`，常以 `--dp-size` 与 `--enable-dp-attention` 这类 serving 配置出现；具体收益依赖模型结构、并行拓扑、batch、上下文长度和 attention backend。
- 后续可补 `Expert Parallelism` 独立页面，并整理 DeepSeek/MLA serving 中 `DPA + EP` 的组合方式。
