---
type: concept
topic: 训练与 Scaling
sources: 0
updated:
---

# RL 训推不一致

## 定义

`RL 训推不一致` 指在线 PPO / GRPO 等训练闭环中，Rollout 推理引擎与 Trainer 在模型策略版本和条件上下文按预期一致时，对同一批已生成 token 重算出的 per-token 条件 logprob 出现超出预期的系统性偏差。

对 prompt `x` 与 response `y`，真正需要对齐的是：

$$
\log \pi_\theta(y_t\mid x,y_{<t})
$$

不是“推理 API 每次输入一个 token、训练 API 一次输入整段序列”这种表面执行形式。

## 为什么 Prefill + Decode 与 Full Forward 理论等价

Decoder-only causal LM 把序列概率分解为：

$$
\pi_\theta(y\mid x)=\prod_t\pi_\theta(y_t\mid x,y_{<t})
$$

Rollout 先 prefill prompt，再逐 token decode；Trainer 把同一组 `prompt_token_ids + response_token_ids` 做 teacher forcing full forward。只要每个位置使用相同的历史 token、权重、mask、position 和 logits 定义，两条路径计算的是同一组条件概率。

设 prompt 长度为 `m`、response 长度为 `n`：

```text
完整序列：       [x_0 ... x_{m-1}, y_0 ... y_{n-1}]
预测 y_0：       logits[m-1]
预测 y_1：       logits[m]
response logp：  logits[m-1 : m-1+n]
```

Prompt 通常被 `loss/response mask` 排除，但不能被 `attention mask` 排除；后续 response token 仍必须以完整 prompt 和已生成 response prefix 为条件。

## 为什么它会破坏 RL 更新

Rollout 保存旧策略概率，Trainer 重算当前策略概率：

$$
r_t=\exp\left(\log\pi_{new}(y_t\mid x,y_{<t})-\log\pi_{old}(y_t\mid x,y_{<t})\right)
$$

如果 actor 刚同步、还没有 optimizer update，通常期望：

```text
delta_t = train_logp_t - rollout_logp_t ≈ 0
ratio_t = exp(delta_t) ≈ 1
```

若系统实现已让 `delta_t` 大面积偏移，importance ratio、PPO clipping、KL 监控和 advantage 加权就会把运行时差异误当成策略更新，严重时导致训练不稳定。

这不同于“当前 actor 与 reference policy 的 KL”：前者比较同一预期策略在两条执行路径上的实现一致性，后者比较两个策略分布的距离。

## 常见不一致来源

### 1. Token 与模板

- Rollout 后把文本 detokenize，再由 Trainer retokenize
- BOS/EOS、assistant header、`add_generation_prompt` 或 stop token 归属不同
- response 截断边界与实际生成 token IDs 不一致

诊断时必须保存并复用 Rollout 真实的 `prompt_token_ids` 与 `response_token_ids`。

### 2. Shift 与 Mask

- response 首 token 错误地对齐到 `logits[m]`，而不是 `logits[m-1]`
- 把 prompt 的 loss mask 误用于 attention
- padding、packing 或 sequence boundary 让不同样本互相可见

### 3. Logits 与采样分布

- 一侧比较 raw model logprob，另一侧比较 temperature、top-k/top-p、repetition penalty 或 token suppression 之后的采样分布
- vocab-parallel gather、log-softmax dtype 或 token gather 实现不同

调试时应先关闭所有 processor，明确比较 raw logits 分布还是实际行为策略分布，再逐项恢复。

### 4. Position 与缓存

- left/right padding 导致 position IDs 不同
- RoPE scaling、YaRN、NTK 等配置不同
- Prefill、PagedAttention、KV Cache dtype 或 cache 写入边界不同

### 5. 权重与策略版本

- Trainer 权重尚未完整同步到 Rollout Mesh
- LoRA merge/adapter、量化权重或 TP/EP shard 版本不一致
- 异步 Rollout 使用 stale policy，却没有携带清晰的 policy version

[[跨 Mesh 权重重分片]] 只负责传输和布局转换；同步 barrier、原子版本切换与轨迹版本标记仍由 RL runtime 负责。

### 6. 数值与 Kernel

- Trainer full forward 与 Rollout decode 使用不同 Attention、RMSNorm、softmax 或 reduction kernel
- BF16/FP16/FP8、FP8 KV Cache、TP all-reduce 顺序和动态 batch 形态不同
- 小幅非 bitwise 差异可能正常；有方向、随位置累积或集中爆发的偏差更值得优先排查

## 推荐排查流程

1. 固定少量 prompt，并保存 Rollout 的 token IDs、per-token old logprob、采样参数、position/mask 信息与 policy version。
2. 使用同一份权重；关闭 dropout、temperature、top-k/top-p、repetition penalty 与其它 processor。
3. Trainer 直接拼接原始 token IDs，用正确 causal shift 重算 response logprob。
4. 同时查看 `delta` 的均值、绝对值分位数、最大值、正负方向、随 token 位置的变化，以及 `exp(delta)`；不要只看一个平均相对误差。
5. 若仍不一致，先切到高精度/eager/单卡等简单路径，再按 tokenizer → shift/mask → position → 权重同步 → dtype/kernel → 并行与动态 batching 的顺序恢复复杂度。
6. 在数值基线通过后，再验证真实采样 processor、量化、Paged KV、TP/EP 和异步 Rollout。

## 与确定性推理的边界

[[确定性推理]] 关注同一推理路径能否复现；RL 训推一致性关注两个不同执行栈是否实现同一条件概率。两边各自可复现，并不保证它们彼此一致；反之，两边存在可接受的微小浮点差异，也不代表 RL 闭环一定失效。

## 关键权衡

- 更严格的一致性模式通常需要关闭部分高性能 kernel、量化或动态调度，只适合作为分层诊断基线，不一定适合长期生产 Rollout。
- 允许多大误差取决于 dtype、模型、序列长度、后端、token 分布和 RL loss 对 ratio 的敏感性，不存在脱离条件的统一阈值。
- 对异步 RL，old/new policy 本来可以不同；关键是区分算法预期的 policy staleness 与实现错误造成的同版本偏差。

## 相关来源

- [[../sources/RLTraining Inference Mismatch]]

## 相关概念

- [[跨 Mesh 权重重分片]]
- [[混合精度训练与推理]]
- [[确定性推理]]
- [[Continuous Batching]]
- [[KV Cache]]

## 研究备注

- 原文给出的 KL estimator 梯度解释、固定 KL/β 阈值与 reference reset 建议尚未核验，不作为本概念的定义依据。
- 后续可结合 VeRL、TRL 或具体 Rollout backend 的官方 consistency test，补充 raw/post-processed logprob 语义、可接受误差分布和自动二分诊断工具。
