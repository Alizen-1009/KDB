---
type: concept
topic: 注意力机制
updated: 2026-08-02
sources: 0
---

# KDA

## 定义

`KDA`（Kimi Delta Attention）是 [[../entities/Kimi K3|Kimi K3]] 使用的递归线性注意力。它保留 Delta Rule 的“只写入 value 预测误差”机制，并把 GDN 常见的 per-head scalar decay 扩展为 **per-head、per-key-channel decay vector**。

## 与 GDN 共用的 Delta Rule

采用状态布局 `S [d_k,d_v]`：

```text
S_bar      = decay(S_{t-1})
prediction = k_t^T S_bar
delta_v    = beta_t * (v_t - prediction)
S_t        = S_bar + k_t delta_v^T
o_t        = S_t^T q_t
```

区别主要在 `decay(S)`：

- GDN：`S_bar = alpha_t * S`，一个 head 内所有 key channels 共用 scalar `alpha_t`；
- KDA：`S_bar = Diag(alpha_t) * S`，`alpha_t [d_k]` 给每个 key channel 独立 retention。

KDA 的完整更新为：

```text
S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1}
    + beta_t k_t v_t^T
```

等价于上述“衰减→预测→误差→rank-1写入”。

## Kimi K3 的 decay 参数化

当前 token 先生成 channel-wise decay logits：

```text
z_t^h = W_alpha_up W_alpha_down x_t + b_alpha^h    # [d_k]
```

Kimi K3 不使用 GDN/Kimi Linear 的无下界 negative-softplus decay，而采用：

```text
g_t^h     = g_min * sigmoid(exp(A_h) * z_t^h)
alpha_t^h = exp(g_t^h)
g_min     = -5
```

因此：

```text
g_t,j ∈ (-5, 0)
alpha_t,j ∈ (exp(-5), 1)
```

`A_h` 是每个 head 的可学习 log-scale；`z_t^h` 是每 token、每 channel 的动态信号。这里没有 GDN 同形的 `dt_bias`：`b_alpha^h` 位于 channel-wise logit projection 中，语义不能直接替换成 GDN 的 positive-step bias。

## 为什么给 log-decay 加下界

Chunkwise KDA 会使用累计 retention `Gamma` 并出现 `1/Gamma` rescaling。若单步 `g` 可趋于负无穷，chunk 内累计 decay 的倒数可能溢出。

K3 固定 `g_min=-5`、二级tile长度16，因此：

```text
sum(g) ∈ (-80, 0)
1 / Gamma < exp(80)
```

该范围仍可由 BF16 动态范围容纳。这样对角与非对角 causal tiles 都能走 dense Tensor Core GEMM，消除 Kimi Linear 中较慢的 position-pair diagonal path。

## Full-rank 输出门

K3 还把 Kimi Linear 的低秩 output gate 改为 input-dependent full-rank gate：

```text
y_t = W_o [sigmoid(W_g x_t) * RMSNorm(o_t)]
```

它允许每个 token 按输出 channel 调节递归状态读取结果。

## 工程权衡

- Channel-wise decay 比 scalar GDN 更有表达力，不同 key channels 可有不同记忆长度。
- 代价是 gate、累计 decay、chunkwise变换和kernel layout更复杂。
- Lower bound 既是数值稳定设计，也是硬件设计：它使16-token tile可统一映射到Tensor Core矩阵乘。
- KDA仍是固定大小递归状态，不能像KV Cache一样任意回退历史token；serving还需保存Conv State和Matrix State checkpoint。

## 相关概念

- [[线性注意力递归状态]]
- [[Chunked Gated Delta Rule]]
- [[混合注意力]]
- [[MLA]]

## 官方资料

- [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)，§2.1.1
- [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)

## 待核实

- Kimi K3 官方开源仓库当前主要提供权重与技术报告；具体 serving engine 的 state layout、dtype、chunk size和融合边界需绑定实现版本。
