# KDA 最小 Decode 伪代码（T=1）

> 假设上游已经从当前 `x_t` 和 Conv State 计算好 `q/k/v/alpha/beta`。本文件只保留 KDA 核心状态递推。

## 输入与输出

```text
输入：
q:      [B,H,K]       已做L2Norm
k:      [B,H,K]       已做L2Norm
v:      [B,H,V]
alpha:  [B,H,K]       real-space retention，已经exp(g)
beta:   [B,H]         write gate，已经sigmoid(b)
state:  [B,H,K,V]     上一个token的矩阵状态

输出：
o:         [B,H,V]
new_state: [B,H,K,V]
```

其中 `T=1`，因此没有 sequence 维，也不需要 token loop。

## 最小伪代码

```python
import torch


def kda_decode_step(q, k, v, alpha, beta, state):
    """
    q, k:   [B,H,K]
    v:      [B,H,V]
    alpha:  [B,H,K]
    beta:   [B,H]
    state:  [B,H,K,V]

    returns:
      output:    [B,H,V]
      new_state: [B,H,K,V]
    """

    # 1. 每个key channel独立衰减
    state_bar = alpha[..., None] * state

    # 2. 旧状态预测当前value
    prediction = torch.einsum(
        "bhk,bhkv->bhv",
        k,
        state_bar,
    )

    # 3. 只写入预测误差
    delta_v = beta[..., None] * (v - prediction)

    # 4. 沿当前key方向更新状态
    new_state = state_bar + torch.einsum(
        "bhk,bhv->bhkv",
        k,
        delta_v,
    )

    # 5. 用当前query读取更新后的状态
    output = torch.einsum(
        "bhk,bhkv->bhv",
        q,
        new_state,
    )

    return output, new_state
```

## 五行数学

$$
\bar S_t=\operatorname{Diag}(\alpha_t)S_{t-1}
$$

$$
\hat v_t=k_t^\top\bar S_t
$$

$$
\Delta v_t=\beta_t(v_t-\hat v_t)
$$

$$
S_t=\bar S_t+k_t\Delta v_t^\top
$$

$$
o_t=S_t^\top q_t
$$

## 如果上游传入的是 `g` 和 `b`

若上游没有提前完成激活：

```python
alpha = torch.exp(g)       # g: [B,H,K]，K3中范围(-5,0)
beta = torch.sigmoid(b)    # b: [B,H]

output, new_state = kda_decode_step(
    q, k, v, alpha, beta, state
)
```

## 完整层还差什么

核心递推返回的是每个head的 `output [B,H,V]`。完整KDA层还会在外面执行：

```python
output = headwise_rmsnorm(output)
output = output * torch.sigmoid(output_gate)  # [B,H,V]
y = out_proj(output.flatten(1))               # [B,d]
```

`q/k/v/alpha/beta/output_gate` 如何从 `x_t` 得到，以及ShortConv如何更新Conv State，都可以放在该核心函数外面。

## 一句话

```text
上游：x_t + Conv State -> q,k,v,alpha,beta
核心：q,k,v,alpha,beta + Matrix State -> output,new_state
下游：RMSNorm + output gate + out_proj -> y_t
```
