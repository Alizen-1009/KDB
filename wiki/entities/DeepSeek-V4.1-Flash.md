---
type: entity
entity_type: 模型
topic: 模型架构
updated: 2026-11-09
sources: 1
---

# DeepSeek-V4.1-Flash

## 一句话说明

围绕长上下文 KV 压缩和 Prefill 成本设计的多模态 MoE 模型，结合 [[../concepts/Causal Encoder-Decoder|CED]]、[[../concepts/CSA2]]、Engram 与 DSpark；本页依据架构解读及已有知识库核对记录整理。

## 类型

- 模型 / 多模态 MoE

## 核心信息

| 项目 | 本次整理口径 |
| --- | --- |
| 参数 | 约 552B 主干，另有约 196B Engram；不将两者重复计数 |
| 主干 | 40 层：20 Encoder + 20 Decoder，hidden 5120；另有 3 个 DSpark block |
| 激活参数 | 文章转述 Prefill 8B、Decode 16B；不是时间或 FLOPs 的直接等价物 |
| 注意力 | 64 个主 Q heads，共享 512 维 KV latent，内含 64 维 RoPE；SWA 窗口 128 |
| 全局缓存 | source 为零基层号 2/8/14/20；前三组二元压缩，后一组逐 token |
| Indexer | 32 个 Q heads，Indexer K 128 维，最终 Top-512；Decoder HSI 候选最多 16,384 个位置 |
| MoE | 384 routed experts 中选 6，专家中间维 2304 |
| 视觉 | 独立 32 层、1024 hidden ViT，patch 14；3×3 特征重排后投影入语言主干；视觉塔不是 CED Encoder |
| 长上下文 | 文章转述支持最长约 1M；具体能力需绑定任务与评测协议 |

## 关键机制

- [[../concepts/Causal Encoder-Decoder|CED]] 改变 Decoder 全局 KV 的依赖来源，长 prompt 大部分位置不必经过全部 Decoder；局部状态仍需有界重放。
- [[../concepts/CSA2]] 跨层共享全局 KV 并选择性复用索引，主 Q 和局部 SWA 仍逐层生成。
- [[Engram]] 在零基层号 1、14 注入，采用 2/3/4-gram，省略原版短因果卷积；是训练参数表，不是会话 KV。
- [[../concepts/mHC]] 使用四路残差流与 Single-Pass 系数偏移；[[../concepts/DSpark]] 使用 3 层草稿网络、一次 5 个草稿位置，仍需 target 验证。

## 相关概念

- [[../concepts/Causal Encoder-Decoder]]
- [[../concepts/CSA2]]
- [[../concepts/KV Cache]]
- [[../concepts/Shared KV Cache]]
- [[../concepts/mHC]]
- [[../concepts/DSpark]]

## 相关实体

- [[DeepSeek-AI]]
- [[Engram]]

## 相关来源

- [[../sources/详细分析 DeepSeek-V4.1 Flash 技术报告(1)  模型结构]]
- [本地技术报告](../../raw/papers/DeepSeek_V41_Tech_Report.pdf)
- [已有配置与技术报告深读](../../output/reports/DeepSeek-V4.1-Flash配置与技术报告深读.html)（派生报告，用于导航，不是独立事实来源）。

## 冲突与备注

- 原文笼统称主 KV 为 MXFP4，与后文 E2M1 + 每 16 通道 E4M3 scale 的描述不一致；沿用既有 KV 账本区分近 NVFP4 的主 KV 与 MXFP4 Indexer K。
- 890 B/token 仅为全局主 KV 与 Indexer K 的打包有效载荷，非总显存；参考实现 quant-dequant 也不等于实际 FP4 打包。
- 文章开头约 420 tok/s 是作者使用观察，缺少硬件、并发、输入输出长度等条件，不收录为可复现 benchmark。
- 缺失公式与作者表达力推导尚待核验；HSI 损失和空间映射假说不属于已确认实现。
