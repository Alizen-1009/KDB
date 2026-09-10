---
type: concept
topic: 投机解码
sources: 10
updated: 2026-07-08
---

# Speculative Decoding

## 定义

一种先用较便宜的机制生成候选 token，再由 target model 并行验证并接受/拒绝的推理加速方法；常见实现既包括 `draft model`，也包括 `辅助层` 与 `数据匹配` 路线。

## 它解决什么问题

- 降低逐 token 自回归生成的高延迟
- 把 target model 的一部分顺序生成开销转化为更适合并行检查的工作

## 核心机制

- 先由较便宜的机制连续猜出若干 token
- target model 对这些候选做并行检查
- 按接受/拒绝规则保留一部分候选，并在需要时回退到 target 分布

## 验证与接受规则

验证阶段不是让 drafter / MTP 自己判断对错，而是把候选 token 接到当前上下文后，让 target model 一次 forward 计算这些位置的 logits：

```text
prefix = x_1 ... x_t
draft  = y_1 ... y_k

target input: x_1 ... x_t y_1 ... y_k

target logits at position t     -> score y_1
target logits at position t + 1 -> score y_2
...
target logits at position t+k-1 -> score y_k
```

- `greedy / argmax 验证`：若 target model 在对应位置的 argmax 等于 draft token，则接受；遇到第一个不一致的位置就停止，丢弃后续 draft，并用 target model 在失败位置给出的 token 继续。
- `严格 speculative sampling`：若 draft token 来自 proposal distribution `q`，target distribution 为 `p`，则按 `min(1, p(y_i) / q(y_i))` 的概率接受；拒绝时从归一化后的正差分布 `(p - q)_+` 采样替代 token。这个规则用于保持采样分布与 target model 一致。
- 如果所有 draft token 都被接受，通常还可以从 target model 在最后一个 draft 后的位置额外采样/选择一个 bonus token，从而保证本轮至少多前进。
- 从 kernel / runtime 形状看，verify 阶段常把普通 decode 的 `qlen=1` 变成 `qlen=draft_len` 的小 chunk decode；历史 prefix 仍来自 KV cache，新增 draft token 的 KV 写入 speculative slots，接受后保留，拒绝后回滚或覆盖。

## 常见路线

- `草稿模型`：用一个更小的 draft model 先生成候选，再由大模型校验
- `辅助层 / 多头预测`：在主模型尾部增加额外 heads 或模块来生成候选，如 `Medusa`、`EAGLE`、[[Multi-Token Prediction|MTP]]
- `数据匹配预测`：利用 prompt 或历史数据中的高频模式直接猜测后续 token，如 `ngram`、`suffix decoding`
- [[Multi-Token Prediction|MTP]]：既可以是训练辅助目标，也可以在推理时作为候选 token 生成机制；只有进入“候选生成 + target 验证”的执行路径时，才构成 speculative decoding。
- [[并行投机解码]]：用 diffusion-style / masked-filling drafter 一次并行预测多个未来位置，减少 sequential drafter 的自回归链；[[DFlash]] 用 target hidden context 增强并行草稿，[[DSpark]] 再根据 confidence 与硬件 `SPS(B)` 自适应截断真正进入 target verification 的前缀。
- `MTP Drafter`：Gemma 4 的具体实现案例，drafter 会利用目标模型 activation、共享 KV cache，并在 E2B/E4B 上用 clustered/sparse LM Head 降低 logits 计算
- `RTP-LLM` 的模块化框架：把 `ProposeExecutor / ScoreExecutor / SpeculativeSampler / SpeculativeUpdater` 拆开，支持朴素 draft、Prompt Lookup、Eagle、MTP 等路线

## 关键权衡

- 能显著改善吞吐和单 token 生成效率
- 效果依赖猜测机制质量、接受率和系统实现开销
- 如果候选经常在第一次校验就失败，总计算量可能反而高于普通 decode
- 不同路线的代价结构差异很大：`draft model` 更吃额外模型协同，`辅助层` 更吃训练耦合，`数据匹配` 更依赖场景重复率
- 并行 drafter 能缩短 proposal 串行链，但固定长块可能把低接受率后缀也送去验证；自适应验证长度可减少浪费，却依赖 confidence 校准、硬件性能模型和额外调度开销。

## 框架实现影响

- 不会改写推理系统“每轮完成一次前向”的基本调度逻辑
- 会改变 `KV Cache` 的管理方式：需要为 speculative token 预留位置，并在候选未被采纳时支持回退或覆盖
- 对运行时输入准备、采样和异步调度提出更高要求，这也是 `vLLM MRV2` 强调 speculative decoding 兼容性的原因之一
- 在 Gemma 4 的 `MTP Drafter` 语境里，target model 仍是最终验证者；连续接受的草稿 token 可直接输出，遇到第一个拒绝 token 后，后续草稿被丢弃并由 target model 给出替代 token
- 在 `SGLang` 的黑盒 API 场景中，文章提到另一种解释器级 speculative execution：第一次 API 调用忽略 stop 条件多生成若干 token，后续原语若能匹配这些额外输出，就可以减少一次 API 调用的输入成本和延迟
- RTP-LLM 来源强调 C++ 级别的模块化调用可减少 Python/C++ 边界开销；该说法需要结合具体框架版本和 profiler 结果核实
- 对 GDN/KDA 等递归线性注意力，draft token 不仅会写临时 KV，还会推进 Conv State 与矩阵状态。Rejected token 不能留在主状态中；来源描述 SGLang 先写暂存状态，验证后按实际接受长度提交，精确实现依赖版本。
- 在 vLLM Prefill + TileRT Decode 的跨引擎 PD 中，TileRT 要从首个 Decode step 启用 MTP，因此 Prefill 侧必须生成并传递 draft-layer KV；目标引擎还需对齐 token position、dtype、layout 和 speculative config。
- DFlash/DSpark 来源称 vLLM 可通过 `method=dflash/dspark` 部署并行 drafter。DSpark 的 batch 级 prefix scheduler 还会让每个请求使用不同 verify length，因此 runtime 不只管理 KV 回滚，还要联合估计 acceptance benefit 与 target verify batch shape 的硬件效率；正式支持范围需绑定 vLLM commit 核实。
- [[../sources/DSpark：结合半自回归生成与置信度调度的投机解码技术]] 称 DSpark 已用于 DeepSeek-V4 Flash / Pro 预览版线上流量，并在相同系统吞吐下提高单用户生成速度。该案例说明生产调度目标是 throughput-latency Pareto frontier，而不是单请求 acceptance length 最大化；具体提升数字仍需绑定官方 workload 与 baseline。

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/vLLM]]
- [[../entities/Gemma 4]]
- [[../entities/SGLang]]
- [[../entities/RTP-LLM]]
- [[../entities/DeepSeek V4]]
- [[../entities/DeepSeek-AI]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 10 - Inference systems and optimization]]
- [[../sources/Model Runner V2 A Modular and Faster Core for vLLM]]
- [[../sources/LLM提速利器：投机推理的原理与常见方案]]
- [[../sources/Gemma 4：Drafter 详解]]
- [[../sources/SGLang：LLM推理引擎发展新方向]]
- [[../sources/RTP-LLM]]
- [[../sources/SGLang的KDA管理与Prefix Cache难题]]
- [[../sources/vLLM x TileRT Specialized Decode for Latency-Critical Serving]]
- [[../sources/并行投机解码(DFlashDSpark)的快速理解与vLLM实测]]
- [[../sources/DSpark：结合半自回归生成与置信度调度的投机解码技术]]

## 相关概念

- [[KV Cache]]
- [[Continuous Batching]]
- [[Multi-Token Prediction]]
- [[MTP Drafter]]
- [[LLM Programs]]
- [[线性注意力递归状态]]
- [[递归状态 Prefix Caching]]
- [[可插拔 Decode 引擎]]
- [[并行投机解码]]
- [[DFlash]]
- [[DSpark]]

## 研究备注

- 现有来源已经把 speculative decoding 从“单一 draft-target 机制”扩展成了一个方案族；后续若频繁引用 `Medusa / EAGLE`，可再拆独立概念页
- 不同接受规则（阈值比较、拒绝采样、校准）对精确采样分布、收益和实现复杂度的影响，当前 wiki 仍写得偏粗，后续可继续细化
- Gemma 4 的例子提醒：`drafter` 不一定是完全独立的小模型，也可以和 target model 深度耦合，复用 activation/KV cache 来换取更高接受率和更低延迟
- SGLang 的 API speculative execution 与常规 draft-target speculative decoding 不是同一层机制；前者更偏程序解释器和黑盒 API 调用复用，失败时可能额外消耗 token，触发条件仍待官方资料核实
- RTP-LLM 中 DeepSeek-V3/MTP、Prompt Lookup 等结果应按任务重复率、接受率、并发和框架版本拆开看；不要只用单个吞吐倍数概括 speculative decoding 的整体收益。
- 并行投机来源提醒：只看 `num_speculative_tokens` 不足以解释收益，还应报告实际 acceptance length、proposal/verify 各自耗时、总 verify batch、并发和硬件效率曲线。正确验证路径原则上不应被解读为提高 target 模型质量；任务准确率变化应先排查数值非确定性和评测噪声。
- DSpark 的线上来源进一步提醒：离线平均接受长度、单用户 tok/s、系统总吞吐与 SLA 下的 Pareto frontier 是不同指标；生产结果不能只用一个 acceptance 或 speedup 数字概括。
