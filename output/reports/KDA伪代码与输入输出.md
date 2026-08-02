# KDA 伪代码与输入输出

> 本文给出 Kimi K3 Technical Report §2.1.1 对应的 **KDA（Kimi Delta Attention）语义参考伪代码**。它强调公式、张量形状与状态更新，不代表生产 FlashKDA kernel 的具体并行实现。

相关页面：[[../../wiki/concepts/KDA|KDA]]、[[../../wiki/concepts/线性注意力递归状态|线性注意力递归状态]]、[[../../output/reports/KDA相对GDN的改进|KDA相对GDN的改进]]。

---

## 1. 层级输入与输出

### 输入

```text
x: [B, T, d]
```

- `B`：batch size；
- `T`：当前处理的 token 数；
- `d`：模型 hidden size。

若从历史前缀继续执行，还需要：

```text
initial_matrix_state: [B, H, K, V]
initial_conv_state_q: [B, H*K, conv_kernel-1]
initial_conv_state_k: [B, H*K, conv_kernel-1]
initial_conv_state_v: [B, H*V, conv_kernel-1]
```

其中：

- `H`：KDA heads；
- `K=d_k`：每个 head 的 key/query dimension；
- `V=d_v`：每个 head 的 value dimension；
- `matrix_state[b,h]` 是 `[K,V]` 的 key-to-value fast-weight matrix。

### 输出

```text
y: [B, T, d]
final_matrix_state: [B, H, K, V]
final_conv_state_q/k/v: 最近 conv_kernel-1 个卷积输入
```

`y` 可以继续进入 residual、AttnRes、MoE 或下一层。

---

## 2. 参数形状

用 `R` 表示 decay gate 的低秩维度：

```text
W_q:          [d, H*K]
W_k:          [d, H*K]
W_v:          [d, H*V]
W_beta:       [d, H]

W_alpha_down: [d, R]
W_alpha_up:   [R, H*K]
b_alpha:      [H, K]
A:            [H]

g_min:        scalar，Kimi K3固定为-5

W_gate:       [d, H*V]      # full-rank output gate
W_o:          [H*V, d]
```

具体 checkpoint 可能按转置形式存储；这里只表达数学输入输出维度。

---

## 3. 核心递推公式

对 token `t`、head `h`：

$$
\bar S_t^h
=\operatorname{Diag}(\boldsymbol\alpha_t^h)S_{t-1}^h
$$

$$
\hat v_t^h
=(k_t^h)^\top\bar S_t^h
$$

$$
\Delta v_t^h
=\beta_t^h(v_t^h-\hat v_t^h)
$$

$$
S_t^h
=\bar S_t^h+k_t^h(\Delta v_t^h)^\top
$$

$$
\tilde o_t^h
=(S_t^h)^\top q_t^h
$$

Kimi K3 channel-wise retention：

$$
z_t^h
=W_\alpha^\uparrow W_\alpha^\downarrow x_t+b_\alpha^h
$$

$$
g_t^h
=g_{\min}\sigma(\exp(A_h)z_t^h)
$$

$$
\boldsymbol\alpha_t^h=\exp(g_t^h)
$$

其中：

```text
g_t:     [B,H,K]，每个key channel一个log-decay
alpha_t: [B,H,K]，每个key channel一个retention
beta_t:  [B,H]，每个head一个delta write gate
```

---

## 4. 单 Token、单 Head 最小伪代码

状态布局采用：

```text
state: [K, V]
```

```python
def kda_one_head_step(q, k, v, alpha, beta, state):
    """
    q:     [K]
    k:     [K]
    v:     [V]
    alpha: [K]      channel-wise retention
    beta:  scalar   delta write strength
    state: [K, V]

    returns:
      output:    [V]
      new_state: [K, V]
    """

    # 1. 每个key channel独立遗忘。
    # alpha[:, None]: [K,1]，按state的K维逐行缩放。
    state_bar = alpha[:, None] * state                 # [K,V]

    # 2. 旧状态对当前key对应value的预测。
    prediction = qkv_contract(k, state_bar)            # [V]
    # 等价：prediction[v] = sum_k k[k] * state_bar[k,v]

    # 3. Delta Rule只写入预测误差。
    error = v - prediction                             # [V]
    delta_v = beta * error                             # [V]

    # 4. 沿当前key方向执行rank-1 update。
    new_state = state_bar + outer(k, delta_v)          # [K,V]

    # 5. 用query读取更新后的状态（read-after-write）。
    output = qkv_contract(q, new_state)                # [V]

    return output, new_state
```

其中：

```python
def qkv_contract(vector_k, state_kv):
    return einsum("k,kv->v", vector_k, state_kv)


def outer(vector_k, vector_v):
    return einsum("k,v->kv", vector_k, vector_v)
```

---

## 5. Batch + Multi-Head PyTorch 风格伪代码

```python
import torch
import torch.nn.functional as F


def kda_recurrent_core(
    q,          # [B,T,H,K]
    k,          # [B,T,H,K]
    v,          # [B,T,H,V]
    alpha,      # [B,T,H,K]
    beta,       # [B,T,H]
    state,      # [B,H,K,V]
):
    B, T, H, K = q.shape
    V = v.shape[-1]

    outputs = []

    for t in range(T):
        q_t = q[:, t]               # [B,H,K]
        k_t = k[:, t]               # [B,H,K]
        v_t = v[:, t]               # [B,H,V]
        alpha_t = alpha[:, t]       # [B,H,K]
        beta_t = beta[:, t]         # [B,H]

        # Step 1: channel-wise forget
        state_bar = alpha_t[..., None] * state          # [B,H,K,V]

        # Step 2: predict value; sum over K
        prediction = torch.einsum(
            "bhk,bhkv->bhv",
            k_t,
            state_bar,
        )                                                # [B,H,V]

        # Step 3: gated delta correction
        delta_v = beta_t[..., None] * (v_t - prediction) # [B,H,V]

        # Step 4: rank-1 state write
        state = state_bar + torch.einsum(
            "bhk,bhv->bhkv",
            k_t,
            delta_v,
        )                                                # [B,H,K,V]

        # Step 5: read after write
        o_t = torch.einsum(
            "bhk,bhkv->bhv",
            q_t,
            state,
        )                                                # [B,H,V]

        outputs.append(o_t)

    output = torch.stack(outputs, dim=1)                 # [B,T,H,V]
    final_state = state                                  # [B,H,K,V]
    return output, final_state
```

---

## 6. 完整 KDA 层伪代码

```python
def kda_layer_forward(
    x,                          # [B,T,d]
    params,
    initial_matrix_state=None,  # [B,H,K,V]
    initial_conv_state=None,
):
    B, T, d = x.shape
    H, K, V = params.H, params.K, params.V

    # ------------------------------------------------------------
    # Step 1: 输入投影
    # ------------------------------------------------------------
    q_raw = linear(x, params.W_q)             # [B,T,H*K]
    k_raw = linear(x, params.W_k)             # [B,T,H*K]
    v_raw = linear(x, params.W_v)             # [B,T,H*V]

    beta_logits = linear(x, params.W_beta)     # [B,T,H]

    decay_low = linear(x, params.W_alpha_down) # [B,T,R]
    decay_logits = linear(
        decay_low,
        params.W_alpha_up,
    )                                          # [B,T,H*K]

    output_gate_logits = linear(x, params.W_gate)  # [B,T,H*V]

    # ------------------------------------------------------------
    # Step 2: ShortConv + Swish
    # Conv函数同时返回供下一次decode使用的Conv State。
    # ------------------------------------------------------------
    conv_q0 = None if initial_conv_state is None else initial_conv_state["q"]
    conv_k0 = None if initial_conv_state is None else initial_conv_state["k"]
    conv_v0 = None if initial_conv_state is None else initial_conv_state["v"]

    q_conv, conv_q = short_causal_conv(
        q_raw,
        conv_q0,
        params.conv_q,
    )
    k_conv, conv_k = short_causal_conv(
        k_raw,
        conv_k0,
        params.conv_k,
    )
    v_conv, conv_v = short_causal_conv(
        v_raw,
        conv_v0,
        params.conv_v,
    )

    q = swish(q_conv).reshape(B, T, H, K)
    k = swish(k_conv).reshape(B, T, H, K)
    v = swish(v_conv).reshape(B, T, H, V)

    # ------------------------------------------------------------
    # Step 3: Q/K L2 Normalization
    # ------------------------------------------------------------
    q = l2_normalize(q, dim=-1)                # [B,T,H,K]
    k = l2_normalize(k, dim=-1)                # [B,T,H,K]

    # ------------------------------------------------------------
    # Step 4: Delta写入门 beta
    # ------------------------------------------------------------
    beta = sigmoid(beta_logits)                 # [B,T,H]

    # ------------------------------------------------------------
    # Step 5: Kimi K3 lower-bounded channel-wise decay
    # ------------------------------------------------------------
    z = decay_logits.reshape(B, T, H, K)
    z = z + params.b_alpha[None, None, :, :]   # [B,T,H,K]

    head_scale = exp(params.A)[None, None, :, None]  # [1,1,H,1]

    g = params.g_min * sigmoid(head_scale * z)  # [B,T,H,K]
    # g_min = -5，因此 g属于(-5,0)

    alpha = exp(g)                              # [B,T,H,K]

    # ------------------------------------------------------------
    # Step 6: 初始化长期矩阵状态
    # ------------------------------------------------------------
    if initial_matrix_state is None:
        state = zeros(B, H, K, V, dtype=float32)
    else:
        state = initial_matrix_state

    # ------------------------------------------------------------
    # Step 7: Gated Delta recurrence
    # 生产Prefill会改写成chunk/UT/GEMM；这里用逐token参考语义。
    # ------------------------------------------------------------
    recurrent_output, final_state = kda_recurrent_core(
        q=q,
        k=k,
        v=v,
        alpha=alpha,
        beta=beta,
        state=state,
    )                                           # [B,T,H,V]

    # ------------------------------------------------------------
    # Step 8: Head-wise RMSNorm + Full-rank output gate
    # ------------------------------------------------------------
    recurrent_output = headwise_rmsnorm(
        recurrent_output,
    )                                           # [B,T,H,V]

    output_gate = sigmoid(
        output_gate_logits.reshape(B, T, H, V),
    )                                           # [B,T,H,V]

    gated_output = output_gate * recurrent_output

    # ------------------------------------------------------------
    # Step 9: 拼接heads并回投影
    # ------------------------------------------------------------
    gated_output = gated_output.reshape(B, T, H * V)
    y = linear(gated_output, params.W_o)         # [B,T,d]

    final_conv_state = {
        "q": conv_q,
        "k": conv_k,
        "v": conv_v,
    }

    return y, final_state, final_conv_state
```

---

## 7. 每一步形状总表

| 步骤 | Tensor | 形状 |
| --- | --- | --- |
| 输入 | `x` | `[B,T,d]` |
| Q投影 | `q_raw` | `[B,T,H*K]` |
| K投影 | `k_raw` | `[B,T,H*K]` |
| V投影 | `v_raw` | `[B,T,H*V]` |
| ShortConv/reshape | `q,k` | `[B,T,H,K]` |
| ShortConv/reshape | `v` | `[B,T,H,V]` |
| Write gate | `beta` | `[B,T,H]` |
| Decay logits | `z` | `[B,T,H,K]` |
| Log decay | `g` | `[B,T,H,K]` |
| Retention | `alpha` | `[B,T,H,K]` |
| 进入状态 | `S_{t-1}` | `[B,H,K,V]` |
| 预测value | `prediction` | `[B,H,V]` |
| Delta | `delta_v` | `[B,H,V]` |
| 新状态 | `S_t` | `[B,H,K,V]` |
| Head输出 | `o_t` | `[B,H,V]` |
| 序列输出 | `o` | `[B,T,H,V]` |
| 拼接heads | `o_flat` | `[B,T,H*V]` |
| 层输出 | `y` | `[B,T,d]` |

---

## 8. K-last 工程布局版本

某些 kernel 将状态存成：

```text
state: [B,H,V,K]
```

此时核心递推改写为：

```python
# alpha按K维缩放
state_bar = alpha_t[..., None, :] * state       # [B,H,V,K]

# state @ k
prediction = torch.einsum(
    "bhvk,bhk->bhv",
    state_bar,
    k_t,
)

delta_v = beta_t[..., None] * (v_t - prediction)

# delta_v outer k
state = state_bar + torch.einsum(
    "bhv,bhk->bhvk",
    delta_v,
    k_t,
)

output = torch.einsum(
    "bhvk,bhk->bhv",
    state,
    q_t,
)
```

这与 `[K,V]` 布局完全等价，只是避免kernel中显式转置。

---

## 9. Prefill 与 Decode

### Decode

`T=1` 或很小时，直接运行 recurrent step：

```text
读取Conv State与Matrix State
→ 处理当前token
→ 原地写回新状态
```

### Prefill

长 `T` 若逐token执行会形成串行链。生产 FlashKDA 使用 chunkwise/UT transform：

```text
块内：多个token转成三角矩阵与GEMM并行
块间：final state顺序传递
```

优化版的输出与 `kda_recurrent_core` 应在数值容差内一致。本文伪代码是 correctness reference，不是性能实现。

---

## 10. 与 GDN 伪代码的唯一核心差异

GDN：

```python
alpha = alpha_scalar[..., None, None]
state_bar = alpha * state
```

KDA：

```python
alpha = alpha_vector[..., None]   # [B,H,K,1]
state_bar = alpha * state
```

后面的：

```text
prediction
error
delta_v
outer-product update
query read
```

仍是同一套 Delta Rule。

---

## 11. 官方公式对应关系

官方报告 Eq. 1：

$$
S_t
=(I-\beta_tk_tk_t^\top)
\operatorname{Diag}(\alpha_t)S_{t-1}
+\beta_tk_tv_t^\top
$$

代码拆解：

```python
state_bar = alpha[:, None] * state
prediction = k.T @ state_bar
delta_v = beta * (v - prediction)
state = state_bar + outer(k, delta_v)
```

两者严格等价。

官方报告 Eq. 6：

$$
y_t=W_o[\sigma(W_gx_t)\odot\operatorname{RMSNorm}(\tilde o_t)]
$$

代码对应：

```python
output_gate = sigmoid(linear(x, W_gate))
y = linear(output_gate * rmsnorm(recurrent_output), W_o)
```

---

## 12. 使用边界

- 这是单层forward语义，不包含residual/AttnRes/MoE。
- 实际K3可能融合Q/K/V projection、ShortConv、gate和norm。
- `A` 在官方公式中是per-head log-scale；不要直接替换成GDN的`A_log + dt_bias`参数化。
- `b_alpha`是channel-wise decay-logit bias，不是GDN positive-step `dt_bias`。
- state dtype可能是FP32、BF16或混合累加；需按backend确认。
- 变长batch需要`cu_seqlens`和request state slots，未在最小伪代码中展开。

## 官方来源

- [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)，§2.1.1、Eq. 1–6
- [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)
