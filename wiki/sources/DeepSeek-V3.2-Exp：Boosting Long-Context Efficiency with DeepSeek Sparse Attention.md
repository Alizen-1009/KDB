---
type: source
source_kind: 论文
topic: 注意力机制
updated: 2026-08-27
---

# DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention

## 来源信息

- 标题：DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention
- 作者：DeepSeek-AI
- 日期：原始 PDF 未注明
- 类型：论文 / 技术报告
- 原始文件：`raw/papers/DeepSeek_V3_2.pdf`
- 模型：[deepseek-ai/DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)
- 推理实现：[Hugging Face inference 目录](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference)

## 2-3 条核心摘要

- [[../entities/DeepSeek-V3.2-Exp|DeepSeek-V3.2-Exp]] 从已扩展到 `128K` 上下文的 DeepSeek-V3.1-Terminus checkpoint 继续训练而来；论文称二者唯一的架构差异是加入 [[../concepts/DeepSeek Sparse Attention|DeepSeek Sparse Attention（DSA）]]。
- DSA 用 lightning indexer 为 query 与历史 token 计算重要性，再只让 top-k latent KV entries 进入核心 attention。它基于 [[../concepts/MLA|MLA]] 的 MQA 模式，使每个 latent KV entry 在所有 query heads 间共享；核心 attention 从 `O(L²)` 降为 `O(Lk)`，但 token-level indexer 仍是 `O(L²)`。
- DSA 不是推理期直接剪枝：模型先用约 `2.1B` tokens 对 indexer 做 dense warm-up，再用约 `943.7B` tokens 联合训练 sparse model 与 indexer，最后沿用 DeepSeek-V3.1-Terminus 的 post-training pipeline、算法和数据。

## 值得关注的论断

- Dense warm-up 固定主模型、只用 attention-distribution KL loss 训练 indexer；Sparse Training 中 indexer input 从主模型图中 detach，indexer 只接收 KL loss，主模型只接收 LM loss。这个优化隔离避免 indexer objective 直接改写主模型表示。
- 表 1 的 `14` 项能力评测未显示整体显著退化，但不是每项持平：GPQA、HLE 和 HMMT 2025 的下降被作者归因于 V3.2-Exp 生成的 reasoning tokens 更少；在输出长度接近的中间 checkpoint 上差距缩小。模型比较因此需要控制生成长度。
- Figure 3 的推理成本来自 H800 集群实际服务 benchmark，并按 `$2/GPU-hour` 换算；短序列 prefill 使用专门的 masked MHA 模式模拟 DSA。图中没有精确数据表，不能从曲线估算并固化美元数值。

## 架构与训练细节

### Lightning Indexer 与 Token Selection

- Index score 采用少量 indexer heads 的加权 ReLU query-key 相似度：`I_t,s = Σ_j w_t,j · ReLU(q_t,j · k_s)`；论文称 indexer 可使用 FP8。
- 每个 query 选择 index score 最高的 `2048` 个历史 latent KV entries，再执行核心 attention。
- DSA 在 MLA 的 MQA mode 下实现：每个 latent KV entry 由同一 query token 的所有 query heads 共享，以满足 kernel 复用需要。

### Continued Pre-Training

- **Dense Warm-up**：`1000` steps，LR `1e-3`，每步 `16×128K`，约 `2.1B` tokens；保持 dense attention，只训练 indexer。
- **Sparse Training**：`15000` steps，LR `7.3e-6`，每步 `480×128K`，约 `943.7B` tokens；每个 query 选择 `k=2048`，主模型和 indexer 按各自 loss 联合推进。
- 两阶段数据分布都与 DeepSeek-V3.1-Terminus 的 `128K` 长上下文扩展数据完全对齐。

### Post-Training 与评测

- Post-training 使用 specialist distillation，再把 reasoning、agent 与 human alignment 合并为单个 mixed RL 阶段，算法仍为 GRPO；论文强调与 V3.1-Terminus 使用相同 post-training pipeline、算法和数据，以尽量隔离 DSA 影响。
- 能力表包含 general、search agent、code、code agent 和 math 共 `14` 项；有升有降，不能概括为逐项不损失。
- 作者明确把该模型标为 experimental，并称仍在开展更大规模真实场景验证。

## 关键概念

- [[../concepts/DeepSeek Sparse Attention|DeepSeek Sparse Attention]]
- [[../concepts/Qwen Sparse Attention|Qwen Sparse Attention]]
- [[../concepts/MLA|MLA]]
- [[../concepts/KV Cache|KV Cache]]
- [[../concepts/RoPE|RoPE]]
- [[../concepts/Benchmarking|Benchmarking]]

## 相关实体

- [[../entities/DeepSeek-V3.2-Exp|DeepSeek-V3.2-Exp]]
- [[../entities/DeepSeek-AI|DeepSeek-AI]]

## 与现有 wiki 的关系

- 创建 [[../concepts/DeepSeek Sparse Attention|DeepSeek Sparse Attention]]，补齐 DSA 的 token-level indexer、MLA MQA instantiation、训练与复杂度边界。
- 更新 [[../concepts/Qwen Sparse Attention|Qwen Sparse Attention]]：区分 DSA 的 token-level `O(L²)` indexer 与 QSA 的 micro-block compression。
- 更新 [[../concepts/MLA|MLA]]、[[../concepts/KV Cache|KV Cache]] 与 [[../concepts/RoPE|RoPE]]：记录 DSA 如何选择 latent KV、共享 MQA entry 和使用 partial RoPE。
- 更新 [[../concepts/Benchmarking|Benchmarking]]：记录 reasoning length 与 H800 服务成本换算的评测口径。
- 未发现与现有 wiki 的直接事实冲突。

## 待确认

- PDF 未给出 Figure 3 的精确成本表；只保留 H800、`$2/GPU-hour` 与 masked-MHA short-prefill 条件，不估读曲线。
- 论文只称 indexer heads 较少且可用 FP8，没有给出 head 数、indexer dimension、partial RoPE dimension或端到端 FLOPs 占比。
- 真实生产流量下的任务分布、P99、显存、吞吐、索引准确率与 failure cases 仍待官方后续验证。
