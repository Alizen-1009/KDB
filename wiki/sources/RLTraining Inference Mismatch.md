---
type: source
source_kind: 文章
topic: 训练与 Scaling
updated:
---

# RLTraining Inference Mismatch

## 来源信息

- 标题：RL: Training Inference Mismatch
- 作者：未标注（来源站点为“谭邵杰的计算机奇妙之旅”）
- 日期：2026-03-17
- 类型：文章 / 实践整理
- 原始文件：`raw/articles/RLTraining Inference Mismatch.md`
- 原始链接：https://shaojiemike.top/artificial-intelligence/2026/03/17/RL-training-inference-mismatch/

## 2-3 条核心摘要

- 文章把 RL 训推一致性的核心定义为：Rollout 推理引擎与 Trainer 对同一条 `prompt + response` 轨迹，应给出接近一致的 per-token 条件对数概率 `log πθ(y_t | x, y_<t)`。推理采用 `prefill + decode`，训练采用 causal teacher forcing full forward；两者计算调度不同，但理论概率分解相同。
- PPO / GRPO 等方法依赖 Rollout 保存的 `old_logprob` 与 Trainer 重算的 `new_logprob` 构造 importance ratio。若权重刚同步、尚未执行 optimizer step，两者仍出现系统性差异，ratio、clipping、KL 与 advantage 加权都会被错误扭曲。
- 文章给出分层排查顺序：首先核对原始 token IDs、chat template、causal shift 与 mask，再核对 logits processor、position IDs、权重版本/LoRA/量化配置，最后隔离 dtype、Attention backend、Paged KV、TP reduction 等数值 kernel 差异。

## 值得关注的论断

- Prompt token 通常不进入 RL loss，但必须作为 response 的条件上下文参与 Attention；`loss/response mask` 与 `attention mask` 是两个不同层级。
- 第一个 response token 的 logprob 来自最后一个 prompt 位置的 logits。设 prompt 长度为 `m`、response 长度为 `n`，正确切片从 `logits[m-1]` 开始，off-by-one 是高频错误。
- Rollout 与 Trainer 一致性应优先比较同一批 token 的 `delta = train_logprob - rollout_logprob` 和 `exp(delta)`；固定 KL 阈值或单个平均相对误差不足以替代 token 级分布、尾部与系统性偏差检查。

## 关键概念

- [[../concepts/RL 训推不一致]]
- [[../concepts/跨 Mesh 权重重分片]]
- [[../concepts/混合精度训练与推理]]
- [[../concepts/确定性推理]]

## 相关实体

- 暂无独立实体页；原文举例涉及 VeRL、TRL、PPO 与 GRPO，但不足以据此建立框架或算法实体页。

## 与现有 wiki 的关系

- 新建 [[../concepts/RL 训推不一致]]，沉淀条件 logprob 对齐、importance ratio 失真机制和分层排查流程。
- 更新 [[../concepts/跨 Mesh 权重重分片]]，补充权重传输完成不等于策略版本语义自动一致，以及异步 Rollout 的 policy staleness 边界。
- 更新 [[../concepts/混合精度训练与推理]] 与 [[../concepts/确定性推理]]，区分可接受的低精度数值漂移、系统性训推偏差和 bitwise 可复现性。
- 未发现与现有 wiki 的直接事实冲突；该来源主要连接了训练闭环、Rollout serving 和数值执行路径。

## 待确认

- 原文前部 PyTorch 示例直接用同位置 logits gather 同位置 token，缺少 causal shift；后文 `logits[:, :-1]` 对齐 `targets=ids[:, 1:]` 的版本才是正确口径。
- 示例 `response_mask = input_ids >= tokenizer.vocab_size` 对合法 token IDs 基本恒为假，不能作为真实 response mask 实现。
- “GRPO 必须启用 KL”、固定 KL 健康阈值、固定 β 范围、定期重置 reference model，以及 `k2/k3` 梯度归属等说法需要绑定论文、框架版本、采样分布与 detach 语义核实，本次不提升为知识库通用结论。
- 原文所述 `10^-3～10^-2` kernel 数值差异仅作为经验线索；一致性阈值应由 dtype、模型、序列长度、后端和 RL 算法敏感性共同校准。
