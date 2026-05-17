# MLA 与 DP Attention 面试整理

## 来源信息

- 类型：对话整理 / 面试专题笔记
- 整理日期：2026-05-11
- 主题：`MHA / GQA / MLA` 的 decode 算术强度、`MLA` 计算流程、矩阵吸收与 `DP Attention`
- 主要参考：
  - DeepSeek-V2 paper: https://arxiv.org/abs/2405.04434
  - DeepSeek-V3 inference code: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
  - SGLang DP/DPA guide: https://sgl-project.github.io/advanced_features/dp_dpa_smg_guide.html

## 一句话回答

`MLA` 是模型结构层的 KV cache 压缩：把完整 per-head `K/V` 缓存在低维 latent 表示里，并通过矩阵吸收避免推理时显式还原完整 `K/V`。`DP Attention` 是 serving 系统层的并行策略：在多 GPU 部署时重新组织 attention/KV cache 的分布，避免 DeepSeek/MLA 类模型在普通 TP attention 下重复保存 KV cache。

## Decode Attention 的算术强度

面试里说的“强度”一般是 `arithmetic intensity`：

```text
arithmetic intensity = FLOPs / HBM bytes
```

在 decode 阶段，标准 `MHA` 每生成一个新 token，要读取完整历史 `K/V cache`：

```text
QK FLOPs ≈ 2 * H * S * D
PV FLOPs ≈ 2 * H * S * D
总 FLOPs ≈ 4 * H * S * D

读 KV cache bytes ≈ 2 * H * S * D * bytes_per_elem
```

在 `FP16/BF16` 下，`bytes_per_elem = 2`，所以粗略：

```text
intensity ≈ 1 FLOP/byte
```

这个量级很低，因此标准 `MHA` 的 decode attention 常常是 memory-bound。注意，这个判断主要针对 attention 读历史 KV cache 的路径，不等价于整个 decoder block 都 memory-bound；`QKV projection`、`O projection`、`MLP`、`MoE expert GEMM` 等大矩阵乘法在合适 batch/shape 下仍可能 compute-bound。

`GQA/MQA` 通过减少 `KV heads`，让多个 query heads 共享同一份 `K/V`，提高 KV cache 复用率。粗略看：

```text
intensity ≈ H_q / H_kv FLOP/byte
```

`MLA` 进一步把历史 `K/V` 缓存成低维 latent，显著降低每个历史 token 的 HBM 读取量；如果 kernel 能有效复用 latent cache，FLOPs 相对 bytes 会变大，decode attention 就可能从 memory-bound 往 compute-bound 移动。

## MLA 的基本计算流程

设第 `t` 个 token 在某层的输入为 `h_t`，hidden size 为 `d`，head 数为 `n_h`，普通 head dim 为 `d_h`，latent KV 维度为 `d_c`。

标准 MHA 会直接产生完整 `Q/K/V`：

```text
q_t = W_Q h_t
k_t = W_K h_t
v_t = W_V h_t
```

MLA 的核心是低秩 KV 联合压缩：

```text
c_t^KV = W_DKV h_t
k_t^C = W_UK c_t^KV
v_t^C = W_UV c_t^KV
```

其中：

- `c_t^KV` 是低维 latent KV，维度为 `d_c`
- `W_DKV` 是 down-projection
- `W_UK / W_UV` 是从 latent 还原 key/value 的 up-projection

推理时真正缓存的不是完整 `k_t^C / v_t^C`，而是：

```text
c_t^KV
```

为了兼容 RoPE，DeepSeek MLA 又把 Q/K 拆成 content 部分和 RoPE 部分：

```text
q_i = [q_i^C ; q_i^R]
k_i = [k_i^C ; k^R]
```

其中 `q_i^R / k^R` 承载 RoPE。推理 cache 需要保存：

```text
c_t^KV
k_t^R
```

所以每层每 token 的 KV cache 近似是：

```text
d_c + d_h^R
```

DeepSeek-V2 论文中的配置示例为 `d_c = 512`、`d_h^R = 64`。

## 为什么 RoPE 要 decouple

如果直接对 `k_t^C = W_UK c_t^KV` 做 RoPE，会出现：

```text
RoPE_j(W_UK c_j^KV)
```

`RoPE_j` 和位置 `j` 有关，是 position-sensitive 的矩阵。它会夹在 `W_UK` 与 query 侧计算之间，不能随意交换顺序。这样 `W_UK` 就无法被吸收到 query projection 里，推理时可能需要为历史 token 重新展开完整 key，破坏 MLA 的效率。

因此 MLA 把 RoPE 维度单独拆出去：

```text
score_j = (q_i^C)^T k_{j,i}^C + (q_i^R)^T k_j^R
```

也就是：

```text
content score + positional score
```

## 矩阵吸收：吸收 W_UK

content score 原本是：

```text
(q_i^C)^T k_{j,i}^C
= (q_i^C)^T W_UK_i c_j^KV
```

利用矩阵乘法结合律：

```text
(q_i^C)^T W_UK_i c_j^KV
= (W_UK_i^T q_i^C)^T c_j^KV
```

所以推理时可以先算：

```text
q_i^latent = W_UK_i^T q_i^C
```

再和缓存的 latent KV 做 dot product：

```text
content_score_j = (q_i^latent)^T c_j^KV
```

这就是 `W_UK` 被吸收到 query 侧。

## 矩阵吸收：吸收 W_UV

value 路径原本是：

```text
o_i = sum_j p_j v_{j,i}^C
    = sum_j p_j W_UV_i c_j^KV
```

由于 `W_UV_i` 是线性映射：

```text
o_i = W_UV_i (sum_j p_j c_j^KV)
```

所以可以先在 latent 空间聚合：

```text
z_i = sum_j p_j c_j^KV
```

再做：

```text
o_i = W_UV_i z_i
```

最终还有 output projection：

```text
u = W_O concat(o_1, ..., o_h)
```

工程实现里可以把 `W_UV` 与 `W_O` 融合或重排到 output 侧，因此论文说 `W_UV` 可以 absorbed into `W_O`。

## Decode 时的 MLA 步骤

```text
1. Query path:
   c_t^Q = W_DQ h_t
   q_t = W_UQ c_t^Q
   拆分 q_nope 与 q_rope
   q_rope 应用 RoPE

2. KV path:
   c_t^KV = W_DKV h_t
   k_t^R = RoPE(W_KR h_t)

3. 写 cache:
   kv_cache[t] = c_t^KV
   pe_cache[t] = k_t^R

4. score:
   q_nope_abs = W_UK^T q_nope
   score = q_nope_abs @ kv_cache_history
         + q_rope @ pe_cache_history

5. softmax:
   p = softmax(score)

6. value 聚合:
   z = p @ kv_cache_history

7. output:
   o = W_UV z
   y = W_O concat(o_heads)
```

在 DeepSeek-V3 的推理代码里，非 naive 路径大体体现为：

```text
q_nope = q_nope @ W_UK
scores = q_nope @ kv_cache + q_pe @ pe_cache
x = scores @ kv_cache
x = x @ W_UV
x = W_O x
```

这里的重点不是代码逐行等价，而是数据流：不显式展开完整历史 `K/V`，而是在 latent cache 上完成 attention 的主要读写。

## Latency 与瓶颈迁移

MLA 的 latency 收益主要来自 decode 长上下文：

```text
MHA 每步读流量 ∝ seq_len * 2 * n_heads * head_dim
MLA 每步读流量 ∝ seq_len * (d_c + d_h^R)
```

这会显著降低 HBM traffic，尤其在长上下文、大 batch、KV cache 压力大的 serving 场景里有效。

但 MLA 不是免费优化。它会增加或重排计算：

```text
q_nope_abs = W_UK^T q_nope
score = q_nope_abs @ latent_cache
z = p @ latent_cache
z 再经过 W_UV / W_O
```

因此它的典型效果是：

```text
HBM bytes 大幅下降
FLOPs 不一定同比例下降，部分路径甚至增加
```

这就是 MLA 可能把 decode attention 从 memory-bound 推向 compute-bound 的原因。

## DP Attention 解决什么问题

`DP Attention` / `Data Parallelism Attention` 是 serving 系统里的并行策略，不是新的 attention 公式。

它主要解决 DeepSeek/MLA 类模型在普通 Tensor Parallel attention 下的 KV cache 复制问题。SGLang 文档给出的典型问题是：DeepSeek/MLA 模型的 KV cache 已经是 latent 表示，而且 KV head 很少；如果继续用普通 TP 组织 attention，多张 GPU 可能保存重复 KV cache，导致：

```text
KV cache 重复保存
显存浪费
batch size 受限
decode throughput 受限
```

DP Attention 的思路是让 attention 侧按 data parallel replica 工作：

```text
不同 replica 处理不同 batch/request
每个 replica 维护自己的 KV cache
不要让同一批请求的 KV cache 在所有 TP rank 上重复一份
```

所以：

```text
MLA: 模型结构层压缩每个 token 的 KV cache
DP Attention: 系统并行层避免多卡部署把 KV cache 重复 N 份
```

两者常一起出现，但层级不同：

- `MLA`：改变 attention 的参数化与 KV cache 形态
- `DP Attention`：改变多 GPU serving 时 attention/KV cache 的并行组织

对 DeepSeek 这类 MoE 模型，工程上 DP Attention 往往还会和 `Expert Parallelism` 组合：attention 侧用 DPA 避免 KV cache 重复，expert 侧用 EP 分摊大量 expert 权重。

## 面试回答模板

可以这样回答：

> 标准 MHA 的 decode attention 通常 memory-bound，因为每生成一个 token 只做一小段 query 计算，却要从 HBM 读取完整历史 K/V cache，FP16 下粗略只有约 `1 FLOP/byte`。GQA/MQA 通过减少 KV heads，让多个 query heads 共享 KV，提高 KV 复用率。MLA 更进一步，把 KV cache 压缩成低维 latent，并把 RoPE 部分 decouple 出来；推理时通过矩阵吸收，把 `W_UK` 吸收到 query 侧，把 `W_UV` 吸收到 output 侧，因此不用显式还原完整历史 K/V。这样 HBM 读流量显著下降，decode attention 的瓶颈可能从 memory-bound 迁移到 compute-bound。
>
> DP Attention 则是系统层优化，不改变 attention 数学。它主要针对 MLA/DeepSeek 类模型在普通 TP attention 下 KV cache 容易重复保存的问题。DPA 让 attention 侧按 data parallel replica 处理不同 batch，每个 replica 维护自己的 KV cache，从而减少显存浪费、提高可承载 batch size 和 decode throughput。它和 MLA 的关系是：MLA 减少单 token KV cache，DPA 避免多卡部署把这份 cache 复制浪费回来。

## 容易说错的点

- 不要说“整个 decoder 都 memory-bound”；更准确是 decode attention 读历史 KV cache 的路径常 memory-bound。
- 不要说“MLA 就是少算”；MLA 更准确是少读 KV cache，用更多或重排后的计算换更少 HBM traffic。
- 不要把 `matrix absorption` 理解成训练出的新矩阵；它主要来自矩阵乘法结合律和推理实现重排。
- 不要把 `DP Attention` 当成新的 attention 结构；它是 serving 并行策略。
- 不要认为 DPA 只和 MLA 绑定；它最适合 MLA 类模型，但部分框架也支持标准 attention 模型，只是收益场景不同。
