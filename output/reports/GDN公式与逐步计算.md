# GDN 公式与逐步计算

> GDN 在本文中指 **Gated DeltaNet / Gated Delta Network**。它用固定大小的 fast-weight matrix 压缩历史，不是显式保存所有历史 K/V。

相关页面：[[../../wiki/concepts/线性注意力递归状态|线性注意力递归状态]]、[[../../wiki/concepts/Chunked Gated Delta Rule|Chunked Gated Delta Rule]]、[[../../wiki/concepts/KV Cache|KV Cache]]。

---

## 1. 一句话直觉

GDN 把每个 head 的历史压缩成一个矩阵：

$$
S_t \in \mathbb{R}^{d_k \times d_v}
$$

可以把 $S_t$ 理解成一个在线学习的线性映射：

$$
k \mapsto v
$$

当前 token 到来时：

1. 先按遗忘门衰减旧状态；
2. 用当前 key 查询状态对 value 的预测；
3. 计算目标 value 与预测之间的误差；
4. 只沿当前 key 方向把误差写回状态；
5. 用 query 从更新后的状态读取输出。

核心不是“把 $k_tv_t^\top$ 一直累加”，而是：

> **只有旧状态预测错的部分才被写入。**

---

## 2. 符号与形状

以下先讨论一个 batch、一个 head、一个 token。令：

| 符号                 | 形状          | 含义                         |
| ------------------ | ----------- | -------------------------- |
| $x_t$              | $[d]$       | 当前 hidden state            |
| $q_t$              | $[d_k]$     | 读取状态的 query                |
| $k_t$              | $[d_k]$     | 写入方向与检索键                   |
| $v_t$              | $[d_v]$     | 希望状态记住的目标 value            |
| $z_t$              | $[d_v]$     | 输出门分支                      |
| $g_t$              | scalar      | log-space遗忘门，通常 $g_t\le0$  |
| $\alpha_t=e^{g_t}$ | scalar      | real-space状态保留率，通常 $(0,1]$ |
| $\beta_t$          | scalar      | delta写入强度，通常 $(0,1)$       |
| $S_{t-1}$          | $[d_k,d_v]$ | 上一时刻的长期矩阵状态                |
| $o_t$              | $[d_v]$     | 当前head输出                   |

工程实现也常把状态转置保存为：

$$
S_t^{\text{impl}}\in\mathbb{R}^{d_v\times d_k}
$$

此时公式中的 $k_t^\top S$ 会写成 `state @ k`，外积 $k\Delta v^\top$ 会写成 `delta_v[:, None] * k[None, :]`。两种布局数学等价。

---

## 3. 从 hidden state 生成各分支

GDN 层通常先做输入投影：

$$
q_t=W_qx_t,\qquad
k_t=W_kx_t,\qquad
v_t=W_vx_t
$$

同时生成：

$$
z_t=W_zx_t,\qquad
a_t=W_ax_t,\qquad b_t=W_bx_t
$$

其中：

- $a_t$ 用于生成遗忘/衰减门；
- $b_t$ 用于生成 delta 写入门；
- $z_t$ 用于最终输出 gating。

许多 GDN 实现还会对 $q/k/v$ 分支做短 causal depthwise convolution：

$$
\tilde q_t=\operatorname{SiLU}(\operatorname{DWConv}(q_{t-K+1:t}))
$$

$\tilde k_t,\tilde v_t$ 同理。为了简化记号，下文仍写成 $q_t,k_t,v_t$。

这也是为什么 serving cache 除了矩阵状态 $S_t$，还需要保存最近 $K-1$ 个卷积输入组成的 Conv State。

---

## 4. Q/K 归一化

常见实现对每个 head 的 query/key 做 L2 normalization：

$$
q_t\leftarrow\frac{q_t}{\lVert q_t\rVert_2+\epsilon},\qquad
k_t\leftarrow\frac{k_t}{\lVert k_t\rVert_2+\epsilon}
$$

输出读取时还可使用缩放：

$$
s=\frac{1}{\sqrt{d_k}}
$$

因此实际读取 query 为 $sq_t$。

归一化使 key 方向和 delta 更新尺度更稳定；具体模型或 kernel 是否在算子内部完成归一化，应以实现为准。

---

## 5. 生成两个门

### 5.1 遗忘门

FLA/vLLM常见参数化为：

$$
\Delta_t=\operatorname{softplus}(a_t+\operatorname{dt\_bias})>0
$$

$$
A=-\exp(A_{\log})<0
$$

$$
g_t=A\Delta_t=-\exp(A_{\log})\Delta_t\le0
$$

$$
\alpha_t=\exp(g_t)\in(0,1]
$$

#### `A_log [H]` 是什么

`A_log` 是每个 value/state head 一个的**可学习基础衰减速率参数**。模型不直接学习负数 $A$，而是学习：

$$
A_{\log,h}=\log\lambda_h
$$

再构造：

$$
A_h=-\exp(A_{\log,h})=-\lambda_h<0
$$

这样天然保证 $A_h$ 为负、$\alpha_t\le1$，避免状态因为正指数而无约束增长。

- `A_log` 越大 $\Rightarrow\lambda_h$ 越大 $\Rightarrow$ 衰减更快、记忆时间尺度更短；
- `A_log` 越小 $\Rightarrow\lambda_h$ 越小 $\Rightarrow$ 衰减更慢、记忆时间尺度更长。

不同 heads 可以学到不同长短的时间尺度。

#### `dt_bias [H]` 是什么

`dt_bias` 是每个 value/state head 一个的**可学习步长偏置**。token-dependent投影 $a_t$ 给出当前token对步长的调整，bias提供每个head的基线：

$$
\Delta_{t,h}=\operatorname{softplus}(a_{t,h}+\operatorname{dt\_bias}_h)
$$

Softplus保证 $\Delta_{t,h}>0$。FLA初始化时通常先采样一个较小的目标步长 $\Delta_h$，再用inverse-softplus设置`dt_bias`，让模型初始就处于合理衰减范围。

`dt_bias`不是位置bias，也不是Attention score bias；它控制遗忘动力学中的有效step size。

#### 两者如何配合

最终保留率为：

$$
\boxed{
\alpha_{t,h}
=\exp\left[-\exp(A_{\log,h})
\operatorname{softplus}(a_{t,h}+\operatorname{dt\_bias}_h)\right]
}
$$

可以理解成：

```text
A_log：这个head天生忘得多快（长期时间尺度）
dt_bias：这个head的默认步长
 a_t：当前token临时把步长调大或调小
alpha：当前token最终保留多少旧状态
```

例如 $\exp(A_{\log})=2$、$\Delta_t=0.1$：

$$
\alpha_t=e^{-2\times0.1}=e^{-0.2}\approx0.819
$$

即该token后保留约81.9%的旧状态。

很多 kernel 的接口直接接收 log-space $g_t$，在内部计算 `exp(g_t)`；另一些接口直接接收 real-space $\alpha_t$。调用时要区分。

### 5.2 Delta写入门

$$
\beta_t=\sigma(b_t)\in(0,1)
$$

解释：

- $\beta_t\approx0$：几乎不写当前信息；
- $\beta_t\approx1$：完整写入当前预测误差。

$\alpha_t$ 控制“旧记忆保留多少”，$\beta_t$ 控制“当前误差改多少”，两者职责不同。

---

## 6. Gated Delta Rule：逐步递推

以下采用状态布局：

$$
S_t\in\mathbb{R}^{d_k\times d_v}
$$

### Step 1：衰减旧状态

$$
\bar S_t=\alpha_tS_{t-1}
$$

形状不变：

```text
[d_k, d_v] -> [d_k, d_v]
```

这一步给所有旧记忆施加统一的 per-head decay。

### Step 2：让旧状态预测当前 value

$$
\hat v_t=k_t^\top\bar S_t
$$

形状：

```text
[1, d_k] @ [d_k, d_v] -> [d_v]
```

$\hat v_t$ 表示：“如果把当前 key 输入旧的关联记忆，它认为对应 value 应该是什么？”

### Step 3：计算预测误差

$$
e_t=v_t-\hat v_t
$$

如果旧状态已经正确记住 $k_t\mapsto v_t$，则 $e_t\approx0$，无需重复覆盖。

### Step 4：用 $\beta$ 控制修正量

$$
\Delta v_t=\beta_te_t
=\beta_t(v_t-\hat v_t)
$$

### Step 5：沿当前 key 方向做 rank-1 update

$$
S_t=\bar S_t+k_t\Delta v_t^\top
$$

形状：

```text
[d_k, 1] @ [1, d_v] -> [d_k, d_v]
```

完整地写：

$$
\boxed{
S_t=
\alpha_tS_{t-1}
+\beta_tk_t\left(v_t-k_t^\top\alpha_tS_{t-1}\right)^\top
}
$$

这就是 gated delta rule 的核心。

### Step 6：从更新后的状态读取输出

FLA reference采用 read-after-write：

$$
o_t=(sq_t)^\top S_t
$$

形状：

```text
[1, d_k] @ [d_k, d_v] -> [d_v]
```

合在一起：

$$
\boxed{
\begin{aligned}
\bar S_t &= \alpha_tS_{t-1}\\
\hat v_t &= k_t^\top\bar S_t\\
\Delta v_t &= \beta_t(v_t-\hat v_t)\\
S_t &= \bar S_t+k_t\Delta v_t^\top\\
o_t &= (sq_t)^\top S_t
\end{aligned}
}
$$

---

## 7. 转置布局下的工程写法

若 kernel 保存：

```text
state: [d_v, d_k]
```

则等价伪代码是：

```python
alpha = exp(g_t)                         # scalar
state = alpha * state                    # [V, K]

prediction = state @ k_t                 # [V]
error = v_t - prediction                 # [V]
delta_v = beta_t * error                 # [V]

state = state + outer(delta_v, k_t)      # [V, K]
output = state @ (q_t * scale)            # [V]
```

这与 FLA fused recurrent reference 的计算顺序一致。

还可以不显式构造 `state_out` 就计算输出：

$$
o_t
=\bar S_t^\top(sq_t)
+\Delta v_t\left(k_t^\top sq_t\right)
$$

该恒等式有利于 decode kernel 减少中间状态的寄存器存活时间。

---

## 8. 一个二维数值例子

令：

$$
S_{t-1}=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},\quad
k_t=
\begin{bmatrix}
1\\0
\end{bmatrix},\quad
v_t=
\begin{bmatrix}
3\\2
\end{bmatrix}
$$

并设：

$$
\alpha_t=0.5,\qquad\beta_t=0.25
$$

### 1. 衰减

$$
\bar S_t=0.5S_{t-1}
=
\begin{bmatrix}
0.5&0\\
0&0.5
\end{bmatrix}
$$

### 2. 预测value

$$
\hat v_t=k_t^\top\bar S_t
=[1,0]
\begin{bmatrix}
0.5&0\\
0&0.5
\end{bmatrix}
=[0.5,0]
$$

### 3. 误差

$$
e_t=[3,2]-[0.5,0]=[2.5,2]
$$

### 4. 写入门

$$
\Delta v_t=0.25[2.5,2]=[0.625,0.5]
$$

### 5. 外积更新

$$
k_t\Delta v_t^\top
=
\begin{bmatrix}1\\0\end{bmatrix}
[0.625,0.5]
=
\begin{bmatrix}
0.625&0.5\\
0&0
\end{bmatrix}
$$

$$
S_t=
\begin{bmatrix}
1.125&0.5\\
0&0.5
\end{bmatrix}
$$

若：

$$
q_t=[1,1]^\top,\qquad s=1
$$

则：

$$
o_t=q_t^\top S_t
=[1,1]
\begin{bmatrix}
1.125&0.5\\
0&0.5
\end{bmatrix}
=[1.125,1.0]
$$

---

## 9. 多head与GVA

包含 batch/head 后，常见逻辑形状为：

```text
q, k: [B, T, H_k, d_k]
v:    [B, T, H_v, d_v]
g:    [B, T, H_v]
beta: [B, T, H_v]
state:[B, H_v, d_k, d_v]
```

若 $H_v>H_k$，可以使用 Grouped Value Attention：多个 value heads共享同一个 q/k head。

映射关系：

$$
h_k=\left\lfloor\frac{h_v}{H_v/H_k}\right\rfloor
$$

每个 value head仍有独立的 $v,g,\beta,S$，但复用对应的 $q/k$。

例如：

```text
H_k = 16
H_v = 128
```

则每个 q/k head服务8个 value heads。

---

## 10. 输出门与回投影

递推得到每个 value head 的 $o_t$ 后，通常还会使用 $z_t$ 做 gated normalization：

$$
\tilde o_t
=\operatorname{RMSNorm}(o_t)\odot\phi(z_t)
$$

$\phi$ 常为 SiLU 或 sigmoid，具体取决于模型。

随后拼接所有 value heads：

$$
\operatorname{concat}(\tilde o_t^{(1)},\ldots,\tilde o_t^{(H_v)})
\in\mathbb{R}^{H_vd_v}
$$

最后回投影：

$$
y_t=W_o\operatorname{concat}(\tilde o_t)
$$

得到与 Transformer hidden size 相同的输出。

---

## 11. 完整前向流程

```text
hidden x_t
  │
  ├─ Wq/Wk/Wv ─> short causal conv + SiLU ─> q_t, k_t, v_t
  ├─ Wz ───────────────────────────────────> z_t
  ├─ Wa ─> softplus + A_log ───────────────> g_t ─> alpha_t=exp(g_t)
  └─ Wb ─> sigmoid ────────────────────────> beta_t

q_t, k_t ─> L2Norm

S_{t-1}
  └─ × alpha_t ─> S_bar
                    │
k_t ────────────────┴─> prediction = k_t^T S_bar
v_t ─────────────────> error = v_t - prediction
beta_t ──────────────> delta_v = beta_t * error

S_t = S_bar + k_t delta_v^T

o_t = (scale * q_t)^T S_t
  │
  └─ RMSNorm + output gate z_t
       └─ flatten heads
            └─ W_o
                 └─ y_t
```

---

## 12. Prefill与Decode如何使用同一公式

### Decode / fused recurrent

每次只有1个或少量新token，直接按上面的递推逐步原地更新：

```text
S_0 -> token 1 -> S_1 -> token 2 -> S_2
```

适合 `q_len≈1` 的Decode。

### Prefill / chunked gated delta rule

长Prompt若逐token执行，会形成长串行链。Chunk版本把序列切成块：

```text
chunk 0 | chunk 1 | chunk 2 | ...
```

块内把多个delta update改写成三角矩阵与GEMM，块间仍传递final state：

```text
S_in(0) -> chunk0 -> S_out(0)=S_in(1)
                         -> chunk1 -> S_out(1)
```

Chunk版本与逐token版本应保持同一数学语义；差别在并行重排，而不是换了模型公式。

---

## 13. 与标准线性注意力和普通DeltaNet对比

### 标准线性注意力

$$
S_t=S_{t-1}+k_tv_t^\top
$$

它无条件累加，每次都写完整value。

### Delta rule

$$
S_t=S_{t-1}+\beta_tk_t(v_t-k_t^\top S_{t-1})^\top
$$

只写预测误差。

### Gated delta rule

$$
S_t=\alpha_tS_{t-1}
+\beta_tk_t(v_t-k_t^\top\alpha_tS_{t-1})^\top
$$

在delta rule基础上增加可学习遗忘。

---

## 14. 复杂度与状态

每 token、每 head 的主状态更新量级为：

$$
O(d_kd_v)
$$

长度 $T$ 的总复杂度：

$$
O(THd_kd_v)
$$

而状态显存与上下文长度无关：

$$
O(Hd_kd_v)
$$

相比 Softmax Attention 的显式 KV history，GDN 长上下文状态不会随 $T$ 线性增长；代价是历史被压缩进矩阵，不能像 Paged KV Cache 一样随意访问或回退任意token。

一个可继续执行的请求状态通常包括：

```text
Conv State
+ Recurrent Matrix State S_t
```

Prefix Cache、请求迁移和投机解码回滚都必须同时处理二者。

---

## 15. 最小PyTorch参考伪代码

```python
import torch
import torch.nn.functional as F


def gdn_step(
    q,                 # [H, K]
    k,                 # [H, K]
    v,                 # [H, V]
    a,                 # [H]
    b,                 # [H]
    state,             # [H, K, V]
    A_log,             # [H]
    dt_bias,           # [H]
):
    # 1. Q/K normalization
    q = F.normalize(q.float(), dim=-1)
    k = F.normalize(k.float(), dim=-1)

    # 2. Gates
    dt = F.softplus(a.float() + dt_bias.float())
    g = -torch.exp(A_log.float()) * dt       # log decay <= 0
    alpha = torch.exp(g)                     # retention in (0, 1]
    beta = torch.sigmoid(b.float())           # write gate in (0, 1)

    # 3. Forget
    state_bar = alpha[:, None, None] * state.float()

    # 4. Predict current value from old memory
    prediction = torch.einsum("hk,hkv->hv", k, state_bar)

    # 5. Delta correction
    delta_v = beta[:, None] * (v.float() - prediction)

    # 6. Rank-1 write
    state_new = state_bar + torch.einsum("hk,hv->hkv", k, delta_v)

    # 7. Read after write
    scale = q.shape[-1] ** -0.5
    output = torch.einsum("hk,hkv->hv", q * scale, state_new)

    return output, state_new
```

该代码只展示核心矩阵递推，不包含：input projections、short convolution、GVA head映射、output gate、RMSNorm、out projection、varlen batch与cache slot管理。

---

## 16. 常见误区

1. **$g$ 与 $\alpha$混用**：FLA常传log-space $g$，实际衰减率是 $e^g$；某些backend直接传real-space gate。
2. **$\alpha$ 与 $\beta$职责混淆**：$\alpha$遗忘旧状态，$\beta$控制当前delta写入。
3. **状态转置看成公式不同**：`[K,V]`和`[V,K]`只是layout差异。
4. **只保存矩阵状态**：启用short conv时还需要Conv State。
5. **把Chunk当成调度Chunked Prefill**：前者是kernel数学并行，后者是serving调度粒度。
6. **把KDA公式直接当GDN**：二者都使用递归状态，但KDA的严格更新和gate结构不同。

---

## 来源

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- [FLA recurrent Gated Delta Rule reference](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py)
- [FLA GatedDeltaNet layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py)
- 本地生产形状与验证reference：`/Users/alizen/Dev/Qwen3.7-GDN-OPT/bench/fused_recurrent_gated_delta_rule/reference.py`

## 待核实边界

不同模型可能采用不同output gate、Q/K normalization、head sharing、negative-eigenvalue扩展、state dtype和read/write顺序。本文核心递推严格对齐当前FLA recurrent reference；映射到具体Qwen/vLLM版本时，应再核对该版本的gate参数化与cache layout。
