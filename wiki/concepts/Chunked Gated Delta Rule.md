---
type: concept
topic: 注意力机制
sources: 1
updated: 2026-06-12
---

# Chunked Gated Delta Rule

## 定义

`Chunked Gated Delta Rule` 是 `Gated Delta Rule` 的分块并行实现。这里的 `chunk` 指把输入序列沿时间/token 维切成固定长度的小块，在每个块内部用矩阵化形式并行计算状态更新，再把块与块之间的最终 state 串起来。

## 为什么叫 Chunk

`Gated Delta Rule` 本身是递推形式。若状态采用工程常见的 K-last 布局 `S [d_v,d_k]`：

```text
S_bar = alpha_t * S_{t-1}
delta_v = beta_t * (v_t - S_bar @ k_t)
S_t = S_bar + delta_v @ k_t^T
o_t = S_t @ (q_t * scale)
```

若采用数学布局 `S [d_k,d_v]`，则全部转置后写成 `prediction=k_t^T S_bar`、`S_t=S_bar+k_t delta_v^T`。详见 [[../../output/reports/GDN公式与逐步计算|GDN公式与逐步计算]]。

如果按 token 一个一个算，就是 recurrent path，依赖链很长：

```text
S0 -> S1 -> S2 -> S3 -> ... -> SL
```

`chunk` 版本把长度 `L` 的序列切成多个块：

```text
[1..C] [C+1..2C] [2C+1..3C] ...
```

块内仍然保持因果依赖，但会把一段 token 的更新改写成更适合 GPU 的矩阵运算；块间只传递每个 chunk 的最终状态。

## 持久状态：Conv State 与矩阵状态

GDN 层跨 token 持久化的通常不只有 `final_state` 矩阵，还包括短 causal depthwise convolution 的 Conv State：

- Conv State 保存最近 `kernel_size-1` 个卷积输入特征，供下一 token 计算局部卷积；
- Recurrent Matrix State 把全部历史压缩成每 head 固定大小的 key-value 关联记忆，由当前 `k/v/gate` 更新、由 `q` 读取。

Decode 逐 token 原地推进两种状态；chunk 版本在 prefill/training 中把块内更新矩阵化、块间传递最终状态。可恢复前缀边界必须同时保存两者，详见 [[线性注意力递归状态]] 与 [[递归状态 Prefix Caching]]。

## 和 Chunked Prefill 的区别

- [[Chunked Prefill]] 的 chunk 是 serving 调度粒度：长 prompt 分多轮下发，方便和 decode 交错。
- `chunk_gated_delta_rule` 的 chunk 是算子 / kernel 内部的计算粒度：把线性递推状态更新按时间维分块，以获得并行度和更好的硬件利用率。

## 适用直觉

- `recurrent` 版本适合逐 token decode，因为每步只更新一次 state。
- `chunk` 版本适合 prefill / training，因为要一次处理一长段序列，分块后可以把更多工作变成矩阵乘和块级 scan。

## Qwen3Next 形状口径

以用户提供的 `Qwen3NextForCausalLM` 配置为例，`linear_attention` 层对应 `Qwen3NextGatedDeltaNet`：

- 模型输入/输出：`hidden_states [B, T, 8192] -> output [B, T, 8192]`
- GDN 头配置：`linear_num_key_heads=16`，`linear_num_value_heads=128`，`linear_key_head_dim=128`，`linear_value_head_dim=128`
- 投影后：`q [B,T,16,128]`、`k [B,T,16,128]`、`v [B,T,128,128]`、`z [B,T,128,128]`、`beta [B,T,128]`、`g [B,T,128]`
- 因为 `128 / 16 = 8`，Qwen3Next 会把 `q/k` repeat 到 value-head 粒度，送入 GDN kernel 时为 `q/k/v [B,T,128,128]`
- `chunk_gated_delta_rule` 输出 `o [B,T,128,128]`，再经 gated RMSNorm、flatten 为 `[B,T,16384]`，最后 `out_proj` 回 `[B,T,8192]`
- 若输出 cache matrix state，`final_state` 形状为 `[N,128,128,128]`，其中 `N` 是 batch 中的序列数；bf16 下约 `4 MB / linear-attention layer / sequence`。该数值只计算矩阵状态，不包含 Conv State 和 runtime bookkeeping。

这个配置的 `layer_types` 是每 4 层 3 个 `linear_attention` 加 1 个 `full_attention`，共 60 层时约 45 个 GDN 层、15 个 full attention 层。

## 相关概念

- [[Chunked Prefill]]
- [[KV Cache]]
- [[CUDA Kernel]]
- [[线性注意力递归状态]]
- [[递归状态 Prefix Caching]]

## 相关实体

- [[../entities/SGLang]]

## 相关来源

- [[../sources/SGLang的KDA管理与Prefix Cache难题]]

## 研究备注

- 上述 GDN state 语义可用于理解递归线性注意力；KDA 的精确矩阵公式不能直接由 GDN 公式替代，需要一手技术资料核实。
