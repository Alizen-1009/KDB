# KDA 相对 GDN 的改进

## 核心结论

KDA 与 GDN 的 state update 骨架相同，都是 Gated Delta Rule：

```text
衰减旧状态
→ 用当前key预测value
→ 计算预测误差
→ 沿key方向写入误差
→ 用query读取更新后的状态
```

KDA 的核心变化不是放弃 Delta Rule，而是把 GDN 的 **per-head scalar decay** 升级为 **per-head、per-key-channel decay**。Kimi K3 又在 Kimi Linear KDA 基础上加入 lower-bounded log-decay 和 full-rank output gate，使它更稳定并更适合 Tensor Core chunk kernel。

相关页面：[[../../wiki/concepts/KDA|KDA]]、[[../../wiki/concepts/线性注意力递归状态|线性注意力递归状态]]、[[../../wiki/concepts/Chunked Gated Delta Rule|Chunked Gated Delta Rule]]。

---

## 1. GDN 的标量遗忘门

GDN 每个 state/value head 产生一个 scalar retention：

$$
\Delta_{t,h}=\operatorname{softplus}(a_{t,h}+\operatorname{dt\_bias}_h)
$$

$$
g_{t,h}=-\exp(A_{\log,h})\Delta_{t,h}
$$

$$
\alpha_{t,h}=\exp(g_{t,h})
$$

对同一 head 的整个状态矩阵：

$$
\bar S_t=\alpha_{t,h}S_{t-1}
$$

也就是说，一个 head 内所有 $d_k$ 个 key channels 以同样速度遗忘。

---

## 2. KDA 的 channel-wise遗忘门

KDA 将 retention 改为向量：

$$
\boldsymbol\alpha_t
\in(0,1)^{d_k}
$$

状态衰减变成：

$$
\bar S_t
=\operatorname{Diag}(\boldsymbol\alpha_t)S_{t-1}
$$

因此第 $j$ 个 key channel 的状态行有自己的 retention：

$$
\bar S_t[j,:]
=\alpha_{t,j}S_{t-1}[j,:]
$$

直觉：

```text
GDN：一个head只有一个遗忘速度
KDA：一个head内每个key channel都能有不同遗忘速度
```

这让同一个 head 同时承载短期和长期信息，而不必完全依赖不同 heads分工。

---

## 3. KDA 的Delta Rule公式

官方 Kimi K3 技术报告写为：

$$
S_t=
\left(I-\beta_tk_tk_t^\top\right)
\operatorname{Diag}(\boldsymbol\alpha_t)S_{t-1}
+\beta_tk_tv_t^\top
$$

$$
\tilde o_t=S_t^\top q_t
$$

令：

$$
\bar S_t=\operatorname{Diag}(\boldsymbol\alpha_t)S_{t-1}
$$

展开：

$$
S_t
=\bar S_t-\beta_tk_tk_t^\top\bar S_t+\beta_tk_tv_t^\top
$$

整理后：

$$
\boxed{
S_t
=\bar S_t
+\beta_tk_t\left(v_t-k_t^\top\bar S_t\right)^\top
}
$$

所以逐步计算仍然是：

$$
\hat v_t=k_t^\top\bar S_t
$$

$$
e_t=v_t-\hat v_t
$$

$$
S_t=\bar S_t+\beta_tk_te_t^\top
$$

KDA 改的是旧状态如何衰减，不是Delta correction本身。

---

## 4. KDA 如何生成channel-wise decay

KDA 对每个head生成 $d_k$ 维 decay logit：

$$
z_t^h
=W_{\alpha}^{\uparrow}
W_{\alpha}^{\downarrow}x_t
+b_{\alpha}^h
\in\mathbb{R}^{d_k}
$$

这是低秩投影加head-specific、channel-wise bias。它比直接用完整 $d\to H d_k$ 投影更省参数和计算。

同时：

$$
\beta_t^h=\sigma(W_\beta^hx_t)
$$

仍是每head scalar write strength。

Q/K/V路径为：

$$
q_t^h,k_t^h
=\operatorname{L2Norm}
\left[
\operatorname{Swish}
\left(
\operatorname{ShortConv}(W_{q/k}^hx_t)
\right)
\right]
$$

$$
v_t^h
=\operatorname{Swish}
\left(
\operatorname{ShortConv}(W_v^hx_t)
\right)
$$

---

## 5. Kimi Linear KDA 的问题

Kimi Linear 沿用类似 GDN/Mamba-2 的 negative-softplus log-decay：

$$
g_t^h
=-\exp(A_h)\operatorname{softplus}(z_t^h)
\in(-\infty,0)^{d_k}
$$

$$
\boldsymbol\alpha_t^h=\exp(g_t^h)
$$

相比 GDN：

- GDN 的 $a_t$ 通常是 scalar per head；
- KDA 的 $z_t$ 是 vector per head/channel；
- 两者都可能让 $g$ 非常负。

在逐token recurrence里，极小 $\alpha$ 本身可以表达强遗忘；但chunkwise并行要计算累计retention：

$$
\Gamma_{i\to j}
=\prod_{r=i}^{j}\boldsymbol\alpha_r
$$

并出现：

$$
1/\Gamma
$$

如果 $g$ 无下界，$1/\Gamma$ 可能迅速溢出。

---

## 6. Kimi K3 改进一：Lower-bounded log-decay

Kimi K3 改成：

$$
\boxed{
g_t^h
=g_{\min}\operatorname{Sigmoid}
\left(\exp(A_h)z_t^h\right)
}
$$

其中：

$$
g_{\min}=-5
$$

因此：

$$
g_t^h\in(-5,0)^{d_k}
$$

$$
\boldsymbol\alpha_t^h
=\exp(g_t^h)
\in(e^{-5},1)^{d_k}
$$

注意它和GDN参数不完全同形：

```text
GDN:
g = -exp(A_log) * softplus(a + dt_bias)

Kimi K3 KDA:
z = W_up W_down x + b_alpha
g = g_min * sigmoid(exp(A) * z)
```

- KDA的 $A_h$ 是head-wise log-scale；
- $b_\alpha^h$ 是channel-wise decay-logit bias；
- 没有与GDN完全同义的 `dt_bias` positive-step参数。

---

## 7. Lower bound 为什么能加速kernel

Chunkwise KDA将大chunk进一步切成16-token secondary tiles。

每步：

$$
g\in(-5,0)
$$

16步累计：

$$
\sum_{r=1}^{16}g_r\in(-80,0)
$$

因此：

$$
\Gamma=\exp\left(\sum g\right)
$$

$$
1/\Gamma<e^{80}
$$

这个范围仍在BF16动态范围内。

Kimi Linear为了处理无界rescaling：

- 非对角tiles可以用dense Tensor Core GEMM；
- 对角tiles仍需要显式position-pair计算。

Kimi K3将log-decay限制在有限范围后：

> 对角和非对角causal tiles都可以使用dense Tensor Core矩阵乘。

从而消除position-pair diagonal path。这个改动同时解决：

- 数值溢出风险；
- 对角tile无法高效走Tensor Core的问题；
- kernel路径不统一的问题。

所以lower bound不是单纯的模型正则，而是明确的模型–kernel协同设计。

---

## 8. Kimi K3 改进二：Full-rank输出门

Kimi Linear的output gate使用低秩参数化。Kimi K3改成input-dependent full-rank projection：

$$
\boxed{
y_t
=W_o\left[
\sigma(W_gx_t)
\odot\operatorname{RMSNorm}(\tilde o_t)
\right]
}
$$

其中：

$$
W_gx_t\in\mathbb{R}^{H d_v}
$$

即每个token可以独立调节每个输出channel，而不是只通过低秩gate控制。

主要作用偏表达能力：

- channel级控制从递归状态读出的内容；
- 降低低秩gate的信息瓶颈；
- 与K3的Gated MLA输出门保持一致设计。

代价是gate projection参数和计算更多。

---

## 9. 对比表

| 维度 | GDN | Kimi Linear KDA | Kimi K3 KDA |
| --- | --- | --- | --- |
| Delta correction | 有 | 有 | 有 |
| Forget gate粒度 | scalar/head | vector/head/channel | vector/head/channel |
| Log-decay | `-exp(A_log) softplus(a+dt_bias)` | `-exp(A) softplus(z)` | `g_min sigmoid(exp(A)z)` |
| Log-decay范围 | $(-\infty,0)$ | $(-\infty,0)$ | $(-5,0)$ |
| Retention范围 | $(0,1)$ | $(0,1)$ | $(e^{-5},1)$ |
| Chunk对角tile | 实现相关 | position-pair路径 | dense Tensor Core GEMM |
| Output gate | 模型相关，常有full output branch | 低秩 | full-rank |
| 主要优势 | 简单、scalar gate | channel-wise记忆 | channel-wise记忆 + 数值/硬件友好 |

---

## 10. KDA的代价

KDA并不是无成本增强：

1. scalar decay变成 $d_k$ 维vector，gate带宽和计算增加；
2. state每行/列需要不同decay，fused recurrent kernel更复杂；
3. chunkwise需要UT transform、累计decay和三角矩阵处理；
4. output full-rank gate增加projection成本；
5. 固定矩阵状态仍会压缩历史，不能完全代替周期性global attention。

因此K3采用：

```text
3层 KDA + 1层 Gated MLA
```

KDA承担高效局部/长期递归混合，周期性MLA提供不受压缩状态限制的全局token-to-token交互。

---

## 11. 一句话总结

> GDN是“每个head一个遗忘速度的DeltaNet”；KDA把遗忘门细化到每个key channel。Kimi K3进一步把无下界negative-softplus gate改成`[-5,0]`有界log-decay，使16-token chunk的累计rescaling落在BF16范围内，从而把原本特殊处理的对角tile也变成Tensor Core GEMM；同时用full-rank output gate提升channel级表达能力。

## 官方来源

- [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)，§2.1.1、Eq. 1–6
- [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464)

## 待核实边界

Kimi K3报告给出了算法与系统设计，但具体开源serving backend的state layout、UT transform实现、state dtype和chunk dispatch仍需按代码版本核实。不能把FlashKDA某个kernel的workspace和tile细节当作KDA数学定义本身。
