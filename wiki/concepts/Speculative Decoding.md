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

## 常见路线

- `草稿模型`：用一个更小的 draft model 先生成候选，再由大模型校验
- `辅助层 / 多头预测`：在主模型尾部增加额外 heads 或模块来生成候选，如 `Medusa`、`EAGLE`、`MTP`
- `数据匹配预测`：利用 prompt 或历史数据中的高频模式直接猜测后续 token，如 `ngram`、`suffix decoding`
- `MTP Drafter`：Gemma 4 的具体实现案例，drafter 会利用目标模型 activation、共享 KV cache，并在 E2B/E4B 上用 clustered/sparse LM Head 降低 logits 计算

## 关键权衡

- 能显著改善吞吐和单 token 生成效率
- 效果依赖猜测机制质量、接受率和系统实现开销
- 如果候选经常在第一次校验就失败，总计算量可能反而高于普通 decode
- 不同路线的代价结构差异很大：`draft model` 更吃额外模型协同，`辅助层` 更吃训练耦合，`数据匹配` 更依赖场景重复率

## 框架实现影响

- 不会改写推理系统“每轮完成一次前向”的基本调度逻辑
- 会改变 `KV Cache` 的管理方式：需要为 speculative token 预留位置，并在候选未被采纳时支持回退或覆盖
- 对运行时输入准备、采样和异步调度提出更高要求，这也是 `vLLM MRV2` 强调 speculative decoding 兼容性的原因之一
- 在 Gemma 4 的 `MTP Drafter` 语境里，target model 仍是最终验证者；连续接受的草稿 token 可直接输出，遇到第一个拒绝 token 后，后续草稿被丢弃并由 target model 给出替代 token
- 在 `SGLang` 的黑盒 API 场景中，文章提到另一种解释器级 speculative execution：第一次 API 调用忽略 stop 条件多生成若干 token，后续原语若能匹配这些额外输出，就可以减少一次 API 调用的输入成本和延迟

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/vLLM]]
- [[../entities/Gemma 4]]
- [[../entities/SGLang]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 10 - Inference systems and optimization]]
- [[../sources/Model Runner V2 A Modular and Faster Core for vLLM]]
- [[../sources/LLM提速利器：投机推理的原理与常见方案]]
- [[../sources/Gemma 4：Drafter 详解]]
- [[../sources/SGLang：LLM推理引擎发展新方向]]

## 相关概念

- [[KV Cache]]
- [[Continuous Batching]]
- [[MTP Drafter]]
- [[LLM Programs]]

## 研究备注

- 现有来源已经把 speculative decoding 从“单一 draft-target 机制”扩展成了一个方案族；后续若频繁引用 `Medusa / EAGLE / MTP`，可再拆独立概念页
- 不同接受规则（阈值比较、拒绝采样、校准）对精确采样分布、收益和实现复杂度的影响，当前 wiki 仍写得偏粗，后续可继续细化
- Gemma 4 的例子提醒：`drafter` 不一定是完全独立的小模型，也可以和 target model 深度耦合，复用 activation/KV cache 来换取更高接受率和更低延迟
- SGLang 的 API speculative execution 与常规 draft-target speculative decoding 不是同一层机制；前者更偏程序解释器和黑盒 API 调用复用，失败时可能额外消耗 token，触发条件仍待官方资料核实
