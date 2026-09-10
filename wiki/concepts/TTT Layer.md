---
type: concept
topic: 模型架构
sources: 2
updated: 2026-08-26
---

# TTT Layer

## 定义

`Test-Time Training Layer`（TTT Layer）是一类把序列模型隐藏状态定义为**学习器内部状态**的 RNN-like 层。参数化实现中，隐藏状态是小模型 `f` 随序列在线更新的权重 `W_t`；每个 token 既用于更新状态，也用于从更新后的状态读取输出。

## 它解决什么问题

- 普通 RNN 将全部历史压进固定大小向量，单 token 成本稳定，但长上下文表达能力受隐藏状态限制。
- Transformer self-attention 通过增长的 [[KV Cache]] 保留显式历史，表达能力强，但训练总算术量随序列长度二次增长，decode 单步需要读取越来越长的历史。
- TTT 希望让固定大小状态本身成为更具表达力的线性模型或 MLP，在状态容量、长上下文质量和每 token 渐近成本之间取得新折中。

## 核心机制

### 状态、更新与输出

令 `f(·; W_t)` 为内循环 learner，`W_t` 是时间 `t` 的隐藏状态：

```text
更新：W_t = W_{t-1} - η ∇_W l(W_{t-1}; x_t)
输出：z_t = f(θ_Q x_t; W_t)
```

原论文使用可学习的多视图重建损失：

```text
l(W; x_t) = || f(θ_K x_t; W) - θ_V x_t ||²
```

- `θ_K x_t`：training view，决定用什么信息训练内部 learner；
- `θ_V x_t`：label view，决定希望状态重建/记住什么；
- `θ_Q x_t`：test view，决定当前输出如何查询状态。

### 内循环与外循环

- **内循环**：在每条输入序列上更新 `W_0 → W_1 → ... → W_T`；即使测试序列也会推进这组状态。`W_t` 在外循环视角下是 activation / hidden state，不是跨样本共享参数。
- **外循环**：使用语言模型 next-token prediction 目标训练 `θ_K / θ_V / θ_Q`、网络其余参数、初始化 `θ_init=W_0` 和 token-dependent learning rate 等，使内循环学会一种对下游语言建模有用的自监督任务。

因此，“test-time training”不表示在 serving 时用标签微调整个 LLM；它指 TTT 层在前向过程中，用当前无标签序列更新其局部隐藏状态 learner。

## 两种主要实现

### TTT-Linear

- 内部 learner 为 `f(x)=Wx`，隐藏状态是方阵。
- 计算和状态流量相对更规整，更容易通过 dual form 映射到 matmul。
- 特定简化条件下可退化为 [[线性注意力递归状态|linear attention fast-weight state]]。

### TTT-MLP

- 内部 learner 是带 LayerNorm、残差与 GELU 的两层 MLP，隐藏维度通常扩到输入维度的 `4×`。
- 隐藏状态表达力更强，但每 token 的权重更新、内存 I/O 和 wall-clock 成本显著更高。

两者在论文实验中默认使用受 Mamba 启发、带时间卷积和门控的 backbone；因此比较结果既包含 TTT 层差异，也包含 backbone 选择影响。

## 并行化与系统实现

### Mini-batch TTT

严格 online GD 中，第 `t` 个梯度依赖 `W_{t-1}`，形成串行链。Mini-batch TTT 让同一小批量内的梯度都相对于上一批末尾状态计算，因此批内可并行、批间仍推进状态。

- 小 `b`：更接近 online learning，通常 perplexity 更好，但并行度低。
- 大 `b`：并行度高，但状态适应更粗；`b=T` 的 batch GD 退化到更简单的累计形式。
- 原论文主要实验使用 `b=16`，该值是特定实验折中，不是通用最优值。

### Dual form

Primal form 会显式产生多个 `d×d` 梯度或权重矩阵，内存 I/O 很大。Dual form 直接推导小批量输出与批末状态，将外积/累计重写为矩阵乘法，避免物化所有中间矩阵。

Dual form 不改变数学输出，但会用额外算术量换更高加速器利用率。论文报告的实现收益绑定 JAX/TPU 或特定 GPU inference kernel，不能只按 FLOPs 推断生产速度。

## 与 Attention 的理论联系

### TTT-Linear 与 Linear Attention

原论文定理 1 要求：

- `f(x)=Wx`；
- batch GD，所有梯度都在 `W_0` 计算；
- `η=1/2`；
- `W_0=0`；
- 使用 `θ_K / θ_V` 重建与 `θ_Q` 输出视图。

此时：

```text
W_t = Σ_{s≤t} (θ_V x_s)(θ_K x_s)^T
z_t = W_t (θ_Q x_t)
```

它等价于不含 softmax 的最简 linear attention。实际 TTT-Linear 还包含 mini-batch、LayerNorm、残差、可学习初始化和学习率，因此不能直接说“TTT-Linear 就是线性注意力”。

### 非参数 Learner 与 Self-Attention

原论文定理 2 使用保存历史样本列表的 Nadaraya-Watson learner 和指数核，可得到与 softmax self-attention 相同的输出。该 learner 的状态随序列增长，不是固定大小 `W_t`；这个定理提供统一抽象，不产生新的高效 attention kernel。

## 来源 Benchmark

原论文与两篇解读文章记录：

- TTT / Transformer：`125M / 350M / 760M / 1.3B`；Mamba 对应 `130M / 370M / 790M / 1.4B`。
- Pile 使用 `2k / 8k`；Books3 使用 `1k–32k` 上下文。
- Pile 2k：TTT-Linear、Mamba、Transformer 大致接近；TTT-MLP 的额外 FLOPs 抵消部分 perplexity 优势。
- Pile 8k 与 Books3 32k：来源报告 TTT-Linear / TTT-MLP 相对 Mamba 的优势扩大。
- Transformer 原始 perplexity 常保持竞争力，但长上下文 FLOP 成本更高；TTT-MLP 的内存 I/O 与实际延迟仍是主要限制。

这些结论只覆盖到约 `1.3B` 与 `32k`，主要质量指标是 perplexity，不能外推到更大模型、百万上下文或生产 serving。

## 关键权衡

- 固定大小状态避免 KV 历史随 token 线性增长，但状态可能是每层大矩阵或 MLP 权重，并不一定小。
- 测试时状态更新增加反向/梯度计算、状态写入和回滚复杂度；它不是“免费记忆”。
- 状态把历史压缩进权重后，不能像 token KV 一样自然支持任意位置回退、逐 token 复用或精确删除。
- 更强 learner 可能提高长上下文表达力，也会增加 FLOPs、HBM 流量、kernel 复杂度和数值稳定性风险。
- Mini-batch size、学习率、初始化、视图投影、LayerNorm 与 backbone 都会影响结果，调参维度明显多于简单递推状态。

## 相关实体

- [[../entities/TTT-LM]]

## 相关来源

- [[../sources/【LLM2】Standford TTT模型(Learn at Test Time)]]
- [[../sources/一文通透TTT：Learning to “Learn at Test Time”，让RNN的隐藏层变成可学习的函数，把T]]

## 相关概念

- [[线性注意力递归状态]]
- [[KV Cache]]
- [[Chunked Gated Delta Rule]]
- [[重计算]]
- [[Benchmarking]]

## 研究备注

- 关键目标、模型规模和定理边界已按 [原论文](https://arxiv.org/abs/2407.04620) 核对；原始资料中的 `250M` 和“内循环 next-token prediction”均按二手来源误记处理。
- TTT 与 GDN/KDA 都可用固定大小矩阵表达历史，但更新目标不同：TTT 显式对可学习自监督损失求梯度；GDN/KDA 使用结构化 gated/delta recurrence，不能直接等同。
- 仍需核对最新 TTT-LM repo、checkpoint、训练 kernel、推理框架支持、状态 checkpoint 与 speculative rollback 的版本化实现。
