---
title: "谭邵杰的计算机奇妙之旅"
source: "https://shaojiemike.top/artificial-intelligence/2026/03/17/RL-training-inference-mismatch/#%E8%AE%AD%E6%8E%A8%E4%B8%80%E8%87%B4%E6%80%A7"
author:
published: 2026-03-17
created: 2026-08-28
description: "谭邵杰的计算机奇妙之旅"
tags:
  - "clippings"
---
## RL: Training Inference Mismatch

> [!abstract] 导言
> - 25年，RL训练崩溃归因于训推不一致；
> - 为此提出了很多方法，TIS，Router Replay，FP16训推，batch一致性...
> - 如何判断 模型当前训推不一致，并找到不一致实现处，是实践的要点。

## 基本概念

### KL 散度

#### Reverse KL Divergence

GRPO 使用的是 **Reverse KL** （采样自当前策略 π\_θ，参考策略为 π\_ref）：

$$
D_{K L} \left(\right. \pi_{\theta} \parallel \pi_{r e f} \left.\right) = \mathbb{E}_{x \sim \pi_{\theta}} \left[log \frac{\pi_{\theta} \left(x\right)}{\pi_{r e f} \left(x\right)}\right]
$$

在 LLM 场景下，按 token 级别计算：

```js
KL = Σ_t π_θ(o_t | q, o_<t) · log[π_θ(o_t | q, o_<t) / π_ref(o_t | q, o_<t)]
```

#### 三种 KL 估计方法

由于直接计算 KL 的期望成本高，实践中采用蒙特卡洛近似。常见有三种估计器 \[\[49\]\]\[\[50\]\]：

| 估计器 | 公式 | 特性 | 适用场景 |
| --- | --- | --- | --- |
| **k1** | `-log(r)` ，其中 `r = π_ref/π_θ` | 无偏但 **方差极大** ，梯度不含 π\_ref | PPO 中作为 reward shaping， **不适合作为独立 KL loss** |
| **k2** | `0.5 * (log(r))²` | **有偏但方差低** ，梯度等价于 Reverse KL | ✅ GRPO 推荐（VeRL/TRL 默认） |
| **k3** | `(r - 1) - log(r)` | 无偏、方差低，但梯度等价于 **Forward KL** | 需注意：采样分布不匹配时可能不稳定 |

关键代码逻辑（VeRL/TRL 实现）：

```python
# 假设已获取 per-token logps
log_ratio = ref_logps - actor_logps  # log(π_ref/π_θ)

# k1: 原始 KL (高方差)
kl_k1 = -log_ratio

# k2: 平方近似 (低方差，推荐)
kl_k2 = 0.5 * (log_ratio ** 2)

# k3: Bregman 形式 (无偏，但梯度对应 Forward KL)
kl_k3 = (log_ratio.exp() - 1) - log_ratio

# 最终 loss 加入
loss = policy_loss + beta * kl_loss_type(actor_logps, ref_logps)
```

> 🔍 **为什么 k2 更推荐？**  
> \- k2 的梯度： `∇_θ [0.5*(log r)²] = (log r) · ∇_θ log π_θ` ，恰好匹配 Reverse KL 的实用梯度形式 \[\[49\]\]  
> \- k3 虽然无偏，但其梯度对应 Forward KL，在 π\_θ 与 π\_ref 差距较大时，重要性采样权重 `π_ref/π_θ` 可能爆炸，导致训练不稳定

监控指标的计算流程（每步 RL）

```js
1️⃣ 采样阶段：
   - 对每个 query，用当前策略 π_θ 生成 G 个 completions
   - 同时用参考策略 π_ref 计算相同 tokens 的 log-probs

2️⃣ 计算 per-token KL：
   log_ratio = log(π_ref) - log(π_θ)
   kl_token = kl_loss_type(log_ratio)  # k1/k2/k3 选一

3️⃣ 聚合为标量指标：
   - 按 completion 平均：kl_per_seq = mean(kl_token over tokens)
   - 按 batch 平均：kl_monitor = mean(kl_per_seq over all completions)

4️⃣ 用于：
   - 📊 监控：TensorBoard/W&B 记录 kl 曲线，判断策略漂移
   - ⚖️ 正则：loss += β * kl_monitor（β=0.001~0.04，依任务调整）
```

---

四、实践建议

1. **配置选择** （以 VeRL 为例）\[\[13\]\]\[\[41\]\]：
	```yaml
	actor_rollout_ref:
	  actor:
	    use_kl_loss: true          # 启用 KL 正则（GRPO 必须）
	    kl_loss_coef: 0.001        # β 系数，数学任务可增至 0.04
	    kl_loss_type: "k2"         # 推荐 k2；若用 k3+ 可开启 straight-through 梯度修正
	```
2. **监控阈值参考** ：
3. KL < 0.01：策略变化过小，可能学习缓慢
4. KL ∈ \[0.01, 0.1\]：健康更新区间
5. KL > 0.2：策略漂移过大，需检查 β 或奖励设计
6. **调试技巧** ：
7. 同时记录 `kl_k2` 和 `kl_k3` ，若二者差异显著，说明 π\_θ 与 π\_ref 已偏离较大
8. 若 KL 持续上升且 reward 不增，考虑定期重置 reference model（DeepSeek-R1 实践）\[\[50\]\]

### log p

`log_p` 是模型对 **每个位置实际生成的 token** 计算出的 **对数概率（Log Probability）** ，属于标量值，而 hidden\_state 是 Transformer 层输出的高维稠密向量。两者在计算链路、数据形态和用途上完全不同。

---

🔍 `log_p` 的完整计算链路

```js
Input Tokens 
   ↓ [Embedding]
Hidden States (layer 0) 
   ↓ [Transformer Blocks × N]
Final Hidden States: h ∈ ℝ^{B×L×D}  ← 这才是你问的 hidden_state
   ↓ [LM Head: Linear + Bias]
Logits: z ∈ ℝ^{B×L×V}            ← 词表大小 V 的未归一化得分
   ↓ [Log Softmax]
Log Probabilities: log_p_all ∈ ℝ^{B×L×V}
   ↓ [Gather 实际 token ID]
log_p ∈ ℝ^{B×L}                  ← 你问的 log_p（每个位置一个标量）
```

📐 维度与形态对比

| 概念 | 形状 | 数据类型 | 物理含义 |
| --- | --- | --- | --- |
| `hidden_state` | `[B, L, D]` | float32/16 | Transformer 输出的上下文表征向量 |
| `logits` | `[B, L, V]` | float32/16 | 词表每个 token 的原始得分 |
| `log_p` | `[B, L]` | float32/16 | **当前策略下，每个位置真实 token 的对数概率** |

> 💡 `D` 通常为 4096/7680 等， `V` 为词表大小（如 32k/128k），而 `log_p` 已坍缩到 `[B, L]` ， **只保留实际生成 token 的概率信息** 。

---

💻 代码级直观实现（PyTorch）

```python
import torch
import torch.nn.functional as F

# 假设 model 已加载，input_ids 为 [B, L]
outputs = model(input_ids=input_ids, return_dict=True)
logits = outputs.logits  # [B, L, V]

# 1. 计算全词表 log softmax
log_probs_all = F.log_softmax(logits, dim=-1)  # [B, L, V]

# 2. 提取实际 token 对应的 log_p
# input_ids.unsqueeze(-1) -> [B, L, 1]
token_log_p = log_probs_all.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)  # [B, L]

# 3. 通常只保留 response 部分（prompt 和 padding 用 mask 过滤）
response_mask = (input_ids >= tokenizer.vocab_size)  # 示例：假设 response token ID 较大
valid_log_p = token_log_p * response_mask  # 后续计算 KL/Loss 时会 mask 掉无效位置
```

---

⚠️ 常见误区澄清

| 误区 | 正确理解 |
| --- | --- |
| “ `log_p` 就是 hidden\_state” | hidden\_state 是向量表征； `log_p` 是经过 LM Head + LogSoftmax + Gather 后的标量概率 |
| “推理时不需要 `log_p` ” | 推理（生成）时模型会隐式计算它用于采样；RL 训练时需 **显式保存** 用于 loss |
| “ `log_p` 越大越好” | 仅表示模型对该 token 更自信；RL 中需与 reward/advantage 结合，盲目最大化会导致 mode collapse |
| “KL 直接用 hidden\_state 算” | KL 是 **概率分布距离** ，必须基于 `log_p` ；hidden\_state 是特征空间，无法直接算分布散度 |

---

📌 总结

- `log_p` = **每个位置真实 token 的对数概率** ，形状 `[B, L]`
- 由 `hidden_state → logits → log_softmax → gather` 得到， **不是 hidden\_state 本身**
- 在 GRPO 中用于：KL 正则、importance ratio、策略梯度、loss mask
- 训练时需同时保存 `actor_logp` 和 `ref_logp` ，推理时通常不显式输出但底层会计算

## 训推一致性

LogP Diff vs KL 散度：本质区别与阈值含义

| 维度 | 🔹 LogP Diff (训推一致性) | 🔹 KL 散度 (RL 正则/监控) |
| --- | --- | --- |
| **比较对象** | **同一模型** ，train mode vs infer mode | **两个策略** ，actor π\_θ vs reference π\_ref |
| **核心目标** | 验证数值计算一致性（精度对齐） | 控制策略更新幅度（防止分布崩溃） |
| **数学形式** | `δ = \\|log_p^train - log_p^infer\\|` | `KL = 𝔼[log(π_θ/π_ref)]` 或其近似 |
| **是否取绝对值** | ✅ 是，关注 **偏差大小** | ❌ 否，保留 **方向信息** （谁更自信） |
| **是否加权** | ❌ 所有 token 平等对待 | ✅ 高概率 token 贡献更大（p·log(p/q)） |
| **量纲/单位** | log 空间的相对误差（无单位比值） | 信息论距离（nat，无单位但数值意义不同） |
| **典型阈值** | rel\_diff < **0.01 (1%)** | KL < **0.01~0.1** (依任务/β系数调整) |

---

```python
# 在 trainer 的 logging 阶段
def compute_consistency_and_kl_metrics(batch):
    metrics = {}

    # 🔹 训推一致性（仅调试阶段启用）
    if config.debug_consistency:
        with torch.no_grad():
            logp_train = model_train(input_ids).logps      # train mode
            logp_infer = model_infer(input_ids).logps      # eval mode
        rel_diff = (logp_train - logp_infer).abs() / (logp_train.abs() + 1e-8)
        metrics["debug/consistency_rel_diff"] = verl_F.masked_mean(
            rel_diff, batch["response_mask"]
        ).item()

    # 🔹 KL 散度（训练必选）
    log_ratio = batch["ref_logps"] - batch["actor_logps"]
    kl_token = 0.5 * (log_ratio ** 2)  # k2
    metrics["actor/kl_loss"] = agg_loss(
        kl_token, batch["response_mask"], config.loss_agg_mode
    ).item()

    return metrics
```

## 训推一致性指标

你的理解里有一个关键偏差： **RL 训练里计算 log p 时，训练端不是只输入 response，而是输入 `prompt + rollout 生成的 response`** 。response 在这里有双重身份：

- 作为 **label/action** ：要计算每个生成 token 的 log probability；
- 作为 **teacher forcing 的上下文** ：第 $t$ 个 response token 之后的 token，需要以前面已经生成的 response token 为条件。

所以训练和推理对齐的不是“API 传入的张量长得一样”，而是对齐同一个条件概率：

$$
log \pi_{\theta} \left(y_{t} \mid x , y_{< t}\right)
$$

其中：

- $x$ ：prompt；
- $y_{t}$ ：第 $t$ 个生成出来的 response token；
- $y_{< t}$ ：它之前已经生成的 response tokens。

推理时逐步得到这些概率；训练时用一次 causal forward 并行算出这些概率。 **数学上是等价的** ，只要 token、mask、position id、模型权重、logits 处理方式一致。

---

### log p 到底对齐什么

设 prompt 为：

```js
x = [x1, x2]
```

rollout 生成的 response 为：

```js
y = [y1, y2, y3]
```

推理阶段并不是永远只输入 prompt。实际过程是：

```js
prefill([x1, x2])      -> log p(y1 | x1, x2)
decode(y1)             -> log p(y2 | x1, x2, y1)
decode(y2)             -> log p(y3 | x1, x2, y1, y2)
```

训练阶段会把同一条轨迹拼起来：

```js
input = [x1, x2, y1, y2, y3]
```

然后 causal LM 的 logits 对齐关系是：

```js
logits at x2 -> predict y1
logits at y1 -> predict y2
logits at y2 -> predict y3
```

因此训练端算的是：

$$
log \pi_{\theta} \left(y_{1} \mid x_{1} , x_{2}\right)
$$
 
$$
log \pi_{\theta} \left(y_{2} \mid x_{1} , x_{2} , y_{1}\right)
$$
 
$$
log \pi_{\theta} \left(y_{3} \mid x_{1} , x_{2} , y_{1} , y_{2}\right)
$$

这和推理阶段逐步 decode 得到的 log p 是同一个东西。

需要注意的是， **prompt token 通常不参与 RL loss** ，但它们必须作为上下文参与 attention。也就是说：

- attention mask 不能把 prompt 屏蔽掉；
- loss mask / response mask 只是在计算 loss 时忽略 prompt 部分。

如果训练端真的只输入 response，那么算出来的是：

$$
log \pi_{\theta} \left(y_{t} \mid y_{< t}\right)
$$

这当然无法和推理阶段的：

$$
log \pi_{\theta} \left(y_{t} \mid x , y_{< t}\right)
$$

对齐。这种流程就是错的。

---

### 为什么一次训练前向可以等价

decoder-only causal LM 的序列概率分解为：

$$
\pi_{\theta} \left(y \mid x\right) = \prod_{t = 1}^{T} \pi_{\theta} \left(y_{t} \mid x , y_{< t}\right)
$$

推理阶段是一个因子一个因子地采样：

```js
p(y1 | x)
p(y2 | x, y1)
p(y3 | x, y1, y2)
...
```

训练阶段用 teacher forcing，把完整序列：

```js
[prompt, response]
```

一次送入模型。由于 causal mask 的存在，每个位置只能看见自己左边的 token，不能看见未来 token，所以它可以并行计算所有条件概率。

以 token 序列：

```js
s = [x1, x2, y1, y2, y3]
```

为例，模型输出 logits：

```js
z0, z1, z2, z3, z4
```

其中：

```js
z1 -> predict y1
z2 -> predict y2
z3 -> predict y3
```

最后一个 logits `z4` 是用来预测 `y3` 后面的下一个 token 的，通常不参与当前 response 的 log p 计算。

一个简化的 PyTorch 对齐逻辑如下：

```python
# prompt_ids: [m]
# response_ids: [n]
# ids: [1, m+n]
ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0)

logits = model(ids).logits          # [1, m+n, vocab]

# logits[:, i] predicts ids[:, i+1]
logp_next = logits[:, :-1].log_softmax(dim=-1)
targets = ids[:, 1:]

token_logp = logp_next.gather(
    dim=-1,
    index=targets.unsqueeze(-1)
).squeeze(-1)

# response 的第一个 token 在 full sequence 中的位置是 m
# 它由 logits 的位置 m-1 预测
resp_logp = token_logp[:, m - 1 : m - 1 + len(response_ids)]
```

这个 `resp_logp` 就应该和推理引擎在 rollout 时记录的 output token logprobs 对齐。

如果你使用 HuggingFace CausalLM 的 `labels` ，常见做法是：

```python
labels = ids.clone()
labels[:, :prompt_len] = -100
```

因为 HF CausalLM 内部通常会做 shift：用 `logits[:, :-1]` 预测 `labels[:, 1:]` 。所以把原始 labels 中 prompt 部分置为 `-100` 后，第一个 response token 仍然会由最后一个 prompt token 的 logits 来预测。

---

### prefill + decode 和 full forward 的关系

prefill + decode 不是另一种概率模型，它只是带 KV cache 的增量计算。

推理：

```js
prefill(prompt) 生成 prompt 的 KV cache
使用最后一个 prompt 位置的 logits 采样 y1
decode(y1) 更新 KV cache，得到 y2 的 logits
decode(y2) 更新 KV cache，得到 y3 的 logits
...
```

训练 full forward：

```js
一次性输入 [prompt, y1, y2, y3, ...]
用 causal mask 同时算出所有位置的 logits
```

KV cache 的作用只是避免重复计算历史 token 的 key/value。理论上：

```js
full forward 的 prefix hidden states
```

和：

```js
prefill + decode 逐步得到的 hidden states
```

应该一致。

因此，只要以下条件一致，二者的 per-token log p 应该接近相等：

- 模型权重一致；
- tokenizer 和 token ids 一致；
- attention mask 一致；
- position ids 一致；
- RoPE scaling、YaRN、NTK 等位置编码配置一致；
- logits 处理方式一致；
- dtype 和 kernel 数值误差在可接受范围内。

实际工程中不一定 bitwise 相等，尤其是 bf16、FlashAttention、PagedAttention、tensor parallel、FP8 KV cache 等场景，但差异应该很小。若出现系统性大偏差，就说明存在训推不一致。

---

### RL 崩溃为什么和 log p 不一致有关

以 PPO / GRPO 类算法为例，训练时通常需要 old logprob 和 new logprob：

$$
r_{t} = exp \left(log \pi_{\theta_{\text{new}}} \left(y_{t} \mid x , y_{< t}\right) - log \pi_{\theta_{\text{old}}} \left(y_{t} \mid x , y_{< t}\right)\right)
$$

其中：

- `old_logprob` ：rollout 时生成该 token 的策略概率；
- `new_logprob` ：训练时当前 actor 对同一个 token 的概率；
- $r_{t}$ ：importance ratio。

如果刚同步完 actor，且还没有做 optimizer step，那么理论上：

```js
new_logprob ≈ old_logprob
r_t ≈ 1
```

如果此时就出现较大偏差，比如：

```js
new_logprob - old_logprob
```

大面积偏离 0，那么 PPO ratio 会被错误放大或缩小，clip、KL、advantage 加权都会失真，训练就可能崩溃。

所以这里说的“训推不一致”，通常不是指“训练输入 response，推理输入 prompt”这种概念差异，而是指：

```js
rollout/inference engine 记录的 log p
```

和：

```js
training engine 对同一批 prompt+response 重算的 log p
```

不一致。

---

### 常见的不一致来源

**1\. token 序列不一致**

这是最常见的问题。

需要确保训练端使用的不是重新 detokenize 再 retokenize 的文本，而是 rollout 时真实生成的 token ids：

```js
prompt_token_ids + generated_response_token_ids
```

常见坑包括：

- BOS 是否自动添加；
- chat template 是否一致；
- 是否使用 `add_generation_prompt=True` ；
- assistant header 是否被当成 prompt 还是 response；
- EOS token 是否包含在 response 中；
- stop string 和 stop token 的处理是否一致；
- 训练端是否重新拼文本后再 tokenize，导致边界 token 变化。

**2\. shift 和 mask 错位**

response 第一个 token 的 logprob 来自最后一个 prompt token 的 logits。

如果 prompt 长度为 $m$ ，response 长度为 $n$ ，那么 response logprob 对应：

```js
logits[m-1 : m-1+n]
```

而不是：

```js
logits[m : m+n]
```

常见错误是 off-by-one。

**3\. logits processor 不一致**

推理时可能使用：

- temperature；
- top-p；
- top-k；
- repetition penalty；
- min length；
- bad words；
- forced EOS；
- stop token suppression。

如果推理记录的是处理后的 logprob，而训练端用的是原始 logits 的 logprob，就会不一致。

例如 temperature 为 $\tau$ 时，采样分布是：

$$
\text{softmax} \left(z / \tau\right)
$$

而不是：

$$
\text{softmax} \left(z\right)
$$

所以要么两边都用 raw logits，要么两边都应用相同的 temperature。调试时建议先关闭所有 processor：

```js
temperature = 1
top_p = 1
top_k disabled
repetition_penalty = 1
```

**4\. padding、position id、packing 不一致**

训练端常用 padding / sequence packing，推理端常用动态 batching / paged attention。需要确保：

- pad token 不参与 attention；
- prompt 不被 loss mask 误当成 attention mask 屏蔽；
- position ids 计算一致；
- packed sequence 之间不能互相 attention；
- RoPE scaling 配置一致；
- left padding / right padding 不导致 position id 差异。

**5\. 模型权重版本不一致**

RL 系统通常有两个模型副本：

```js
rollout engine actor
training engine actor
```

需要确认：

- 权重是否已经同步；
- LoRA adapter 是否一致；
- rollout engine 是否加载了 merge 后权重；
- tensor parallel 切分是否一致；
- 是否存在异步 rollout 的 stale policy；
- 量化权重、FP8 KV cache 是否引入较大误差。

**6\. 数值 kernel 差异**

full forward 和 decode 可能使用不同 kernel：

- FlashAttention；
- PagedAttention；
- fused RMSNorm；
- fused softmax；
- bf16/fp16/fp32；
- tensor parallel all-reduce；
- vocab parallel softmax。

小误差正常，但大面积系统性误差不正常。

经验上，fp32/eager 模式可以非常接近；bf16、不同 attention kernel 下可能有 $10^{- 3}$ 到 $10^{- 2}$ 量级差异。这个不是硬标准，但如果偏差明显更大，或者 ratio 明显偏离 1，就要排查。

---

### 建议的排查流程

可以按下面顺序做最小化一致性测试：

1. 固定一批 prompt。
2. rollout engine 生成 response，并保存：
3. `prompt_token_ids`
4. `response_token_ids`
5. `old_logprobs`
6. `attention_mask`
7. `position_ids`
8. 关闭 temperature、top-p、top-k、repetition penalty 等 logits processor。
9. 确认训练端和推理端使用同一份权重。
10. 训练端用：
```js
input_ids = prompt_token_ids + response_token_ids
```

重算 per-token logprob。

1. 对齐 response 部分：
```js
train_logprob[i] 对齐 infer_logprob[i]
```

其中：

```js
train_logprob[i]
=
log p(response_ids[i] | prompt_ids, response_ids[:i])
```
1. 比较：
```python
delta = train_logprob - infer_logprob
ratio = torch.exp(delta)
```

期望：

```js
delta ≈ 0
ratio ≈ 1
```

如果不满足，优先检查：

```js
token ids -> chat template -> shift -> mask -> logits processor -> position ids -> 权重同步 -> dtype/kernel
```

一句话总结：

**推理是逐步生成同一条 response，训练是对这条固定 response 做 teacher-forcing 评分；二者对齐的是每个 token 的条件 log probability，而不是表面上的输入 API。prefill+decode 与一次 full forward 在 causal mask 下理论等价，工程上的不一致才是需要排查的核心。**