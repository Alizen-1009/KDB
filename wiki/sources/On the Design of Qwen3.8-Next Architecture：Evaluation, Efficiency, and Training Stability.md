---
type: source
source_kind: 论文
topic: 模型架构
updated: 2026-08-27
---

# On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability

## 来源信息

- 标题：On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability
- 作者：Qwen Team
- 日期：2026-08-26
- 类型：论文
- 原始文件：`raw/papers/qwen3.8-Next.pdf`

## 2-3 条核心摘要

- 论文给出 [[../entities/Qwen3.8-Flash-Next|Qwen3.8-Flash-Next]] 的架构与消融：主干为 `125B` 总参数、每 token 激活 `6B` 的稀疏 MoE，此外还有放在加速器外的 `51B` n-gram embedding 参数；这 `51B` 是主干 `125B` 之外的额外容量，不能把二者含糊合并成“125B 中包含 51B”。
- 主干以每四层 `3` 层 GDN 加 `1` 层全局注意力进行混合；在 `256K` continued pretraining（CPT）阶段，backbone 与 MTP 中的全注意力都替换为 [[Qwen Sparse Attention]]。残差路径采用四分支 [[Gated Residual]]，Layer 2 增加单层 [[N-gram Embedding]]，训练主要使用 [[Muon Optimizer]]。
- 论文把 loss、下游评测、训练/推理成本和稳定性联合评估：多处出现 loss 与下游排序不一致、预训练与 post-training 结论不一致，说明架构选择不能只依据单一预训练 loss。

## 值得关注的论断

- 表 11 中，Qwen3.8-Flash-Next-Base 在 `14` 项评测上全部超过 Qwen3.8-27B-Base；相对 `397B` 总参数、`17B` 激活的 **Qwen3.7-Plus-Base**，胜 `8/14`，最大落后为 MultiPL-E 的 `2.59` 分。论文称其约使用 `1/3` 激活参数、`1/3` 训练 tokens 和 `1/9` 训练 FLOPs，但没有披露最终绝对 token 数或 FLOPs。
- GDN hybrid、QSA、GR 与 n-gram embedding 的若干选择都体现了晚阶段边界：NoPE 在预训练近似但 post-training 更易 endless generation；GR top-2 sparse read 在预训练近乎无损但 post-training 明显退化；扩大 n-gram 表持续降低 loss，但多数下游准确率饱和或波动。
- 中尺度加压实验显示 Muon、GR/GatedNorm 与新结构具有更大的稳定性余量，但这是为复现大规模不稳定而设计的代理 stress test；组合收益不能完全归因到某一个组件。

## 关键结果与事实边界

### 模型与基线

- Qwen3.8-Flash-Next 主干：`125B` 总参数、每 token 激活 `6B`；另有 `51B` n-gram embedding 参数存放在加速器外。
- 表 11 基线是 **Qwen3.7-Plus-Base**（`397B` 总参数、`17B` 激活），不是 `Qwen3-Next 397B-A17B`。
- 表 11 还包括 Qwen3.8-27B-Base；Qwen3.8-Flash-Next-Base 在全部 `14` 项上胜过它。

### GDN hybrid

- 层比例为每四层 `3 GDN + 1 full attention`。在 `28` 层 `25B-A3B`、先训 `400B` tokens@`4K` 再训 `80B` tokens@`32K` 的消融中，full attention / SWA hybrid / GDN hybrid 的九项平均分分别为 `49.87 / 51.15 / 53.81`；GDN 对 full attention 胜 `8/9`，对 SWA 胜 `7/9`。
- GDN 输出使用 bounded sigmoid gate 与 zero-centered RMSNorm。论文在 NVIDIA GPUs 的多个设置中报告 FlashQLA 相对 FLA Triton kernel 的 forward `2–3×`、backward 约 `2×`；该数字不能泛化到其它硬件、shape 或端到端训练。
- full-attention 层保留 [[RoPE]]：NoPE 预训练表现近似，但 post-training 后 endless generation 比例明显更高。

### QSA

- QSA 只在 `256K` CPT 引入，替换 backbone 与 MTP 的所有 full-attention 层。其 MQA indexer 使用 `4` 个 query heads、`1` 个共享 key head，head dimension `128`，其中 `64` 维使用 partial RoPE。
- key 先按 `r=4` 做平均池化，再给压缩块赋块起点位置；token budget `K=2048`，即最多选择 `512` 个完整块，并始终加入最后未完成的尾块。
- Stage 1 只训练 indexer：`1000` 步、学习率 `1e-3`、每步 `8×256K`，约 `2B` tokens。Stage 2 联合训练 backbone 与 indexer：`8000` 步、学习率 `2.5e-5`、每步 `96×256K`，约 `200B` tokens。
- 短任务上 QSA 在 `8` 项中 `7` 项不降，均分 `75.9→76.8`；RULER `512K–1M` 为 `90.08→93.00`，MRCR 在 `512K` 为 `30.66→40.53`、`1M` 为 `20.71→26.44`。MTP 四步投机解码平均接受长度 `4.06→4.07`。
- `1M` kernel 实验的 prefill 为 chunked prefill 最后 `16K`、`BS=1`；decode 为 `BS=4`、`next_n=4`，含 `3` 步 MTP。相对 FlashInfer paged GQA attention，包含 indexer 与 sparse core 的 QSA attention module 在该实验中 prefill/decode 分别为 `7.6×/4.9×`；这只是指定 kernel 条件下的 module-level 结果。

### Gated Residual

- 残差流扩为四分支；每支独立 RMSNorm，read 为逐分支、逐 channel 的 sigmoid gate，低秩瓶颈 `r=d/8`；write 是每分支动态标量。每个 block 的 attention 与 MLP 各使用独立 GR，并移除 `Hres`，以少一次完整残差状态读取。
- `25B-A3B`、`560B` tokens 的表 5 中，pre-norm / static mHC / dynamic mHC / GR 的平均分为 `50.91 / 52.49 / 54.47 / 54.66`，loss 为 `1.617 / 1.596 / 1.594 / 1.590`。dynamic 相对 static mHC 的 loss 只降 `0.002`，但平均分增 `1.98`，显示 loss 与下游表现不总一致。
- top-2 sparse read 在预训练近乎无损但 post-training 明显退化。论文观察到 FP8 残差存储相对 BF16 将残差流量减半且几乎无质量损失；未给出可泛化到任意硬件与实现的端到端加速。
- 与 AttnRes 的比较限定在特定 `28` 层 loss：未加 GatedNorm 的 Full AttnRes 为 `1.762`，GR（结构本身包含 gated read/GatedNorm）为 `1.762`；给 Full AttnRes 加 GatedNorm 后为 `1.758`。这些并非完全同构的配置，也不支持宣称二者存在普遍优劣。

### N-gram embedding

- 最终只在 Layer 2 放一层，使 host prefetch 可与 Layer 1 计算重叠；最终表容量为额外 `51B` 参数，采用确定性寻址并存于加速器外。
- 固定 MoE、将表从 `20×` 扩至 `200×` tokenizer vocabulary 时，loss `1.553→1.526`，但多数准确率饱和或波动，中文任务较持续改善。
- 论文称额外 FLOPs 与 latency 可忽略，但没有披露带宽、延迟或命中率；也不能从论文猜测 n-gram 阶数、slot/hash 细节或压缩方式。

### Muon 与分布式实现

- Muon 使用 Nesterov `μ=0.95` 和 `8` 次 Newton–Schulz 迭代，只用于真正充当二维 linear map 的权重。embedding、LM head、router、GR 的低秩/门投影仍用 AdamW；n-gram table 使用无 weight decay 的 Adam。
- 融合 qkv、GDN input、SwiGLU fc1 必须按语义子矩阵拆分后分别正交化，不能把拼接矩阵整体送入 Muon。
- 本文的 Canzona 是实现线索：按 whole tensor 将 NS FLOPs 平衡到 DP ranks，经 TP fused All-to-All 重构完整矩阵，并用 CUDA Graph 消除拆分后大量小 kernel 的 launch overhead。论文没有说明其已经开源。

### Scaling 与稳定性

- `20` 层 `10.8B-A0.89B`、`4T` tokens 实验中，batch `25.2M` 的 loss 为 `1.5702`，旧配方 `12.6M` 为 `1.5774`，`37.7M` 为 `1.5707`。batch 从 `6.3M` warmup 到 `25.2M` 不更好，且多 `18.8%` optimizer steps。
- `48` 层 `156B-A7B`、`419B` tokens 实验预测 `B=8.4M / LR=1.76e-3`，旧配方为 `4.2M / 6.8e-4`；最终 loss 优 `7.8e-3`，平均准确率 `60.55 vs 56.41`。预测点附近的设置只各评估一次，细小排序应视为噪声。
- stress test 使用 `28` 层 `25B-A3B`、恒定 `2×/4×` optimal LR 与 clip threshold `0.5`。`2×` 时 AdamW 为 `4.3` spikes/万步，两组 Muon 均为 `0.2`；`4×` 时 AdamW 为 `183` spikes/万步，且 `19932` 步中 `213` 次越阈，两组 Muon 均不越阈，Muon+GR 为 `0` spike。
- 单变量 GatedNorm 实验在 `3×` LR 下把 spike rate 从 `32.0` 降到 `3.2`/万步、越阈次数从 `256` 降到 `20`。这些是中尺度加压代理结果，不能把完整组合收益完全因果拆分。

## 关键概念

- [[混合注意力]]
- [[线性注意力递归状态]]
- [[Qwen Sparse Attention]]
- [[Gated Residual]]
- [[N-gram Embedding]]
- [[Conditional Memory]]
- [[Hyper-Connections]]
- [[mHC]]
- [[Attention Residuals]]
- [[Muon Optimizer]]
- [[Critical Batch Size]]
- [[Scaling Laws]]
- [[RoPE]]

## 相关实体

- [[../entities/Qwen3.8-Flash-Next]]
- [[../entities/阿里巴巴]]

## 与现有 wiki 的关系

- 更新注意力主线：把 GDN/full-attention hybrid、QSA 的 CPT 替换与 RoPE 边界接入既有页面。
- 更新残差主线：将 GR 与 Hyper-Connections、mHC、Attention Residuals 放在相同的残差路径设计空间中，并保留特定消融边界。
- 更新 conditional memory 与 scaling 主线：把 Qwen n-gram 实现和 Muon 下的 batch/LR 实验作为具体案例，不覆盖 Engram 或通用 scaling 结论。
- 未发现直接事实冲突；需要显式保留模型/论文/文件命名差异，以及不能把表 11 的 Qwen3.7-Plus-Base 误标为 Qwen3-Next。

## 待确认

- 论文未披露最终训练的绝对 token 数、绝对 FLOPs，以及 n-gram table 的带宽、延迟、命中率和具体寻址细节。
- QSA 与 GR 的 kernel 结果需绑定论文给定的硬件、batch、context、MTP 和 baseline 条件后引用。
