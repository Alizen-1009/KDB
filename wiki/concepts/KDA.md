---
type: concept
topic: 注意力机制
updated: 2026-08-02
sources: 3
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

## 部署与Prefix Cache

Kimi K3混合部署同时维护两类cache：KDA的固定大小`Conv State + Matrix State`，以及Gated MLA随序列增长的Paged KV Cache。Prefix只有在同一token边界同时存在MLA KV和所有KDA cache groups的状态checkpoint时才能复用。

生产系统将两类cache放入统一byte-sized paged pool，但解耦physical page、prefix hash block与KDA checkpoint粒度：报告示例使用6144-token物理page、内部512-token hash blocks，KDA仅在稀疏hash endpoints保存checkpoint。命中时checkpoint先复制为请求私有running state，不能原地修改共享快照。完整流程见 [[../../output/reports/Kimi K3的KDA部署与Prefix Cache|Kimi K3的KDA部署与Prefix Cache]]。

投机解码回滚与Prefix Cache不同：K3不为每个draft位置保存完整状态，而缓存较小的projected inputs，在验证后片上重放accepted tokens并重建正确状态。

vLLM的K3集成进一步将Physical State Block、Scheduler Alignment与Prefix Hash Unit解耦：大状态块内部仍可注册细粒度Prefix Endpoint，但只有MLA KV、Matrix State和ShortConv State对同一个`num_computed_tokens`有效时才算命中。命中checkpoint必须Copy-on-Write为请求私有Running State。

## GLM-5.3-Flash 配置案例

[[../entities/GLM-5.3-Flash]] 在 `45` 个文本层中使用 `34` 个 KDA 层与 `11` 个 DSA 层，约为 `3:1`；DSA 位于 layers `3/7/.../43`，最后的 layer `44` 是 KDA。其 KDA 配置为：

```text
num_heads              = 64
head_dim               = 128
short_conv_kernel_size = 4
gate_lower_bound       = -5.0
```

这里 KDA 用固定大小 recurrent state 低成本聚合历史，周期性的 [[DeepSeek Sparse Attention|DSA]] 保留显式 top-k token 检索能力。该配置案例不改变本页从 Kimi K3 资料整理出的 KDA 机制定义。

## 工程权衡

- Channel-wise decay 比 scalar GDN 更有表达力，不同 key channels 可有不同记忆长度。
- 代价是 gate、累计 decay、chunkwise变换和kernel layout更复杂。
- Lower bound 既是数值稳定设计，也是硬件设计：它使16-token tile可统一映射到Tensor Core矩阵乘。
- KDA仍是固定大小递归状态，不能像KV Cache一样任意回退历史token；serving还需保存Conv State和Matrix State checkpoint。

## 投影与Decode融合

Kimi K3技术报告§5.4.2明确的Decode融合范围是ShortConv、Input Norm、Gating、KDA Recurrence与Output Norm；它还通过缓存`projected inputs`来Replay投机接受tokens，未明确声称Input Linear与整个Decode Core同kernel。开源vLLM则进一步明确实现：共享输入`x`的Q/K/V、Full-rank Output Gate、Decay低秩入口和Beta合并为一次Merged Column-Parallel GEMM；Decay第二级Projection仍单独执行。Q/K/V保持Packed Layout，Decode融合ShortConv、Q/K Norm、Gate、KDA Recurrence、Output Gate与RMSNorm；Input Projection和最终`o_proj`仍是独立GEMM。Prefill因FlashKDA需要dense Q/K/V，融合边界与Decode不同。Shape、报告/源码证据边界见 [[../../output/reports/KDA投影融合优化|KDA投影融合优化]]。

## FlashKDA并行

逐token递推可写成仿射transition `S_t=M_t S_{t-1}+B_t`。仿射变换可结合，因此segments可用prefix scan组合；chunk内部通过UT transform改写为causal lower-triangular GEMM。[[../entities/FlashKDA|FlashKDA]]进一步用CUTLASS/Tensor Core实现，并重叠块内token计算与块间状态传播。完整推导见 [[../../output/reports/FlashKDA为什么能并行|FlashKDA为什么能并行]]。

## CAKE KDA 全融合 Prefill

[[../entities/CAKE KDA]] 在 B200/SM100a 上提供另一种 prefill 调度：不把 chunk preparation 与 recurrence 拆成 K1/K2 两个 kernels，而是在单 CTA 内由五组 producer 预先准备五个 32-token chunks，再由 consumer 严格按顺序推进 FP32 recurrent state。固定 exponent anchor 让 chunk 32 的 BF16 Q/K 因子保持在可用范围内，并在 `Mqk` 中抵消；state 跨 chunks 常驻 [[Tensor Memory|TMEM]]，chunk-local 中间量通过五级 SMEM ring 和 lifetime aliasing 留在片上。

这与 [[../entities/FlashKDA|FlashKDA]] 两阶段方案形成互补：FlashKDA 的 K1 可沿 chunks×heads 提供更高 preparation 并行度，但需要 global workspace；CAKE 消除 workspace/HBM 往返，却让整体 grid 更直接受 batch×heads 限制。小 batch 或少 heads 时，两阶段或 shape-aware dispatch 仍可能更合适。

## 伪代码

Decode `T=1` 且上游已算好 `q/k/v/alpha/beta` 时，优先看 [[../../output/reports/KDA最小Decode伪代码|KDA最小Decode伪代码]]；完整输入投影、Conv State、batch/sequence与K-last布局再看 [[../../output/reports/KDA伪代码与输入输出|KDA伪代码与输入输出]]。

## 相关实体

- [[../entities/Kimi K3]]
- [[../entities/GLM-5.3-Flash]]

## 相关概念

- [[线性注意力递归状态]]
- [[Chunked Gated Delta Rule]]
- [[混合注意力]]
- [[MLA]]
- [[Tensor Memory]]

## 相关来源

- [[../sources/A Preview of Production-Scale Kimi K3 Support on vLLM]]
- [[../sources/REMINDER FF-KDA & CAKE KDA Highlights]]
- [[../sources/glm-5-architecture-evolution]]

## 官方资料

- [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)，§2.1.1
- [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3)

## 待核实

- Kimi K3 官方开源仓库当前主要提供权重与技术报告；具体 serving engine 的 state layout、dtype、chunk size和融合边界需绑定实现版本。
