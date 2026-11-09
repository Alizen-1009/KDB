---
type: concept
topic: 模型架构
updated: 2026-11-09
sources: 1
---

# Causal Encoder-Decoder

## 定义

`Causal Encoder-Decoder（CED）` 在因果语言主干中分离上下文记忆的生成与消费：前半部分 Encoder 生成上下文表示，后半部分 Decoder 的全局 KV 从 Encoder 边界表示投影得到。这里的 Encoder 仍是因果的，不是双向编码器。

## 核心机制

以 [[../entities/DeepSeek-V4.1-Flash]] 为例，以下层号均从 0 开始：

1. 0–19 层 Encoder 处理完整 prompt，建立自身全局缓存。
2. 从 Encoder 末端表示生成第 20 层的全局 KV；结合 [[CSA2]]，后续 Decoder 层共享这一份全局记忆。
3. Decoder 的局部 SWA KV 仍由各层自身输入生成，不能从 Encoder 输出一次投影出所有局部状态。
4. Prefill 只让 prompt 尾部的一小段 Encoder 输出经过 20–39 层，补建 Decoder 局部状态并产生末端预测。
5. Decode 时每个新 token 仍经过完整 40 层。

CED 改变全局 KV 的依赖来源；CSA2 决定哪些层共享缓存和索引。二者不是同一个机制，亦不意味着层参数共享。

## SWA Bounded Replay

- 深层窗口 KV 的输入 hidden 已混合下层的历史信息，依赖会沿深度向前扩张；恢复最后一个窗口不等于只需计算最后一个窗口。
- 有界重放只处理最近 `W=128` 个位置，截断更早的局部依赖，是近似恢复。全局 KV 仍可覆盖长历史，没有把模型上下文整体缩短为 128。
- 需区分 Decoder 每次 Prefill 的尾部重放，以及缓存命中但 Encoder SWA 状态缺失时的恢复；不能据此宣称所有 cache-hit 边界都数学等价或逐 bit 一致。

## 计算账本与权衡

LLM 归纳，按相同每层成本的 token-layer 示意，长度为 `N` 的完整 Prefill 为 `40N`，CED 加尾部重放约为 `20N + 20W`，比例为 `1/2 + W/(2N)`。只有 `N` 远大于 `W` 时主体工作量才接近减半；这不是实际 FLOPs 或延迟的精确公式，未计投影、不同层成本、通信与调度。

- 全局记忆生成不再依赖所有 Decoder 层，但局部深层状态恢复仍有成本。
- 近似重放需要模型适配及质量验证；短后缀交互可能更受固定恢复开销影响。
- 文章受 YOCO 启发的架构叙事，不等于所有 YOCO、KV-Mirror 与 CED 实现完全一致。

## 实现边界

既有 [V4.1 配置与技术报告深读](../../output/reports/DeepSeek-V4.1-Flash配置与技术报告深读.html) 记录：固定 revision `dba1be0a40aa45a94ad051997016db3960a90277` 的 minimal forward 遍历完整主干，不能据此认定已实现生产级 Prefill 跳过调度。本次沿用该已有核对记录，未重新复测。

## 相关实体

- [[../entities/DeepSeek-V4.1-Flash]]

## 相关概念

- [[CSA2]]
- [[Shared KV Cache]]
- [[KV Cache]]

## 相关来源

- [[../sources/详细分析 DeepSeek-V4.1 Flash 技术报告(1)  模型结构]]
- 一手资料定位：[技术报告](../../raw/papers/DeepSeek_V41_Tech_Report.pdf) §2.2、§3.2.2。

## 研究备注

- 本次来源剪藏存在公式缺失；只整理可读正文及已有知识库核对结果，不补造缺失推导。精确感受野公式须明确层输入/输出和窗口端点定义。
