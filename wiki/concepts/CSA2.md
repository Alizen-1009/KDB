---
type: concept
topic: 注意力机制
updated: 2026-11-09
sources: 1
---

# CSA2

## 定义

[[../entities/DeepSeek-V4.1-Flash]] 的注意力设计，结合局部 SWA、全局稀疏读取与跨层 KV/索引复用；核心是将“共享内容”和“共享选中的地址”解耦。

## 核心机制

| 模式 | 全局主 KV / Indexer K | Top-K | 本层仍需计算 |
| --- | --- | --- | --- |
| Full | 新生成 | 新选择 | 主 Q、局部 SWA KV、主注意力、输出与后续子层 |
| Reindex | 复用对应来源层缓存 | 用本层 Indexer Q 重新选择 | 主 Q、局部 SWA KV、主注意力、输出与后续子层 |
| Reuse | 复用对应来源层缓存 | 复用最近索引层的选择 | 主 Q、局部 SWA KV、主注意力、输出与后续子层 |

共享 Top-K 不等于共享 softmax 权重或输出。Reuse 跳过的是全局 KV 重建与 Indexer，不是跳过整层；“只改 Q”仅限固定全局内容和地址这一观察范围。

### 与 V4 CSA 的区别

- Encoder 使用非重叠的二元组压缩，Decoder 压缩率为 1，保留逐位置条目；不是所有层都使用高倍率序列压缩。
- 移除 V4 CSA compressor 的重叠与绝对位置嵌入。
- Indexer K 从主 latent 的 pre-RoPE、pre-cache-quantization 版本投影获得；不再从 hidden states 单独运行一套 Indexer compressor。详见 [[CSA-HCA]]。

### 层级调度（零基层号）

| 范围 | 调度 | 全局缓存来源 |
| --- | --- | --- |
| Encoder 0–1 | 纯 SWA | 无全局缓存 |
| Encoder 2–19 | 三组，每组 1 Full + 5 Reuse | 2、8、14 |
| Decoder 20–39 | 首组 1 Full + 3 Reuse；后四组各 1 Reindex + 3 Reuse | 全部共享第 20 层 |

索引来源为 `2/8/14/20/24/28/32/36`。跨层状态共享不自动意味着权重绑定或 GPU L2 实际命中。

### 分层稀疏 Indexer（HSI）

- 第 20 层先扫描全部因果可见条目，产生自身 Top-512；同一份索引分数按每块 8 个位置取最大值，最多选 2048 块，生成每 query 最多 16,384 个位置的候选池。
- 24、28、32、36 层在该候选池内各自重选 Top-512；其后 Reuse 层复用最新选择。
- 固定候选池限制后续 Reindex 的打分范围，但首层全局扫描仍随历史长度增长；不能推出整个注意力或整个模型严格 O(1)。
- 文章称 HSI 在后训练引入，训推使用同样的候选限制；具体 Indexer 损失配方未披露，文中的混合损失和蒸馏形式属于作者猜测。

## 缓存与权衡

- 四个主 KV source 同时共享对应 Indexer K。全局缓存账本见 [[KV Cache]]：偶数长度下有效载荷为 `890 B/token/sequence`，不乘 40 层，也不乘主 Q 或 Indexer Q 的头数。
- Reindex 可修正已共享内容中的选择，但不能找回 HSI 候选池外条目；Reuse 不能纠正当前共享 Top-K 的召回遗漏。
- 更少的全局内容来源节省存储，代价是限制逐层独立改写全局记忆的自由度；局部 SWA 与后续变换仍在更新。
- 原文用“内容／地址／查询”和缓存层级解释机制，这是概念类比，不是硬件 Cache/TLB 实现事实。

## 表达力论证的边界

文章讨论固定字典读出、凸包和指数倾斜，但剪藏缺失关键公式，未完整核验。不能将其“保留整个注意力单纯形”或“逼近某顶点”提升为无条件结论：查询的有效维度、可达范围与字典几何均需限定；全局字典固定也不代表本层局部 SWA 字典固定。RoPE、输出逆旋转、投影和多头组合需纳入论证范围。

## 实现边界

既有 [V4.1 深读](../../output/reports/DeepSeek-V4.1-Flash配置与技术报告深读.html) 在固定 revision `dba1be0a40aa45a94ad051997016db3960a90277` 记录：参考 Indexer 先对完整历史打分再 mask 候选外位置，保留选择语义，但未兑现仅对候选打分的计算收益。本次未重新核验或 benchmark。

## 相关实体

- [[../entities/DeepSeek-V4.1-Flash]]

## 相关概念

- [[Causal Encoder-Decoder]]
- [[Shared KV Cache]]
- [[CSA-HCA]]
- [[KV Cache]]

## 相关来源

- [[../sources/详细分析 DeepSeek-V4.1 Flash 技术报告(1)  模型结构]]
- 一手资料定位：[技术报告](../../raw/papers/DeepSeek_V41_Tech_Report.pdf) §2.3。
