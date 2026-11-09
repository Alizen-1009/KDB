---
type: source
source_kind: 文章
topic: 模型架构
updated: 2026-11-09
---

# 详细分析 DeepSeek-V4.1 Flash 技术报告(1)：模型结构

## 来源信息

- 标题：详细分析 DeepSeek-V4.1 Flash 技术报告(1)：模型结构
- 作者：渣B / zartbot
- 日期：2026-09-17 08:26（正文署时）；剪藏日期 2026-11-09
- 类型：文章 / 技术报告二手解读
- 原始文件：[[../../raw/articles/详细分析 DeepSeek-V4.1 Flash 技术报告(1)  模型结构]]
- 原文：[微信文章](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247499768&idx=1&sn=d00b5f17a59ec29634538da5ab9acabd)
- 作者排版版：[中文网页版](https://zartbot.github.io/blog/model_arch/dsv41flash_arch/index.html)
- 一手资料定位：[DeepSeek-V4.1-Flash 技术报告](../../raw/papers/DeepSeek_V41_Tech_Report.pdf) §2.1–2.4、§3.2.2。
- 阅读范围：完整读取 824 行剪藏；存在公式和表格数值缺失，网页版文本提取也未补齐；远程配图未逐图核验。本次经用户确认，按可读正文与既有知识库核对记录入库，缺失数学推导不视为已验证。

## 2-3 条核心摘要

- **CED 降低 Prefill 主体工作量**：Encoder 生成上下文表示，Decoder 全局 KV 由其边界表示投影；大部分历史位置不用经过完整 Decoder，但局部 SWA 仍需尾部重放。Bounded Replay 是有界近似，不是无条件精确恢复。
- **CSA2 解耦缓存共享和索引共享**：Full 更新内容与索引，Reindex 共享内容但重选位置，Reuse 共享二者；每层主 Q、局部 SWA 和主注意力仍需计算。HSI 为后续 Decoder Reindex 限制候选范围。
- **多维压缩协同**：条目维、序列维、层维和精度共同影响缓存量。四组全局缓存结合 FP4 得到约 890 B/token 的有效载荷；Engram、Single-Pass mHC 与 DSpark 分别服务于条件记忆、残差执行与投机生成，不能混算成同一项 KV 收益。

## 值得关注的论断

| 论断 | 记录方式与边界 |
| --- | --- |
| Prefill 只需前 20 层、计算减半 | 是长输入主体路径的概括；Decoder 尾部重放和全局投影仍有成本。已有粗账本为 20N+20W 对 40N，不是精确延迟公式。 |
| 全局 KV 890 B/token | 既有 KV Cache 页已核对：主 KV 288 B/entry，Indexer K 68 B/entry，来源份数按 3/2+1 折算；不包含局部状态和总部署开销。 |
| HSI 让索引成本有界 | 仅后续 Reindex 的每查询候选打分预算有界；第一层仍全局扫描。参考实现全量打分后 mask 与生产候选内打分不可混同。 |
| Reuse 只改变查询 | 仅固定全局字典与地址这一观察范围成立；局部 SWA、输出变换与后续 MoE 仍更新，且共享地址不等于共享注意力权重。 |
| 内容／地址／查询对应缓存层级与页表 | 作者解释框架，不是硬件 Cache/TLB 命中、物理驻留或权重共享的证明。 |
| 固定字典中的凸包与指数倾斜解释表达力 | 缺失公式待补核；查询可达范围、秩、字典几何和局部支路必须限定。“保留整个单纯形”等强表述不作为已确认结论。 |
| HSI 可能用主注意力蒸馏 Indexer，Q 感知映射可补偿共享 KV | 作者猜测；不是报告已披露的训练配方或已确认模块。 |

### 其他模块的可读正文要点

- 视觉：独立 ViT 先处理 patch 网格，经 3×3 pixel-unshuffle 和 projector 映射入语言主干；视觉内容 token 与换行、起止标记需分别计数。剪藏尺寸表缺失，具体缩放数值不补造。
- 多模态 MoE：文章描述文本与图像分别维护负载校正 bias；bias 用于选专家，专家输出加权保留原始路由分数，不能混为同一分数。
- Engram：2/3/4-gram、零基层号 1/14 两个注入点，省略原版短因果卷积；约 196B 参数独立于约 552B 主干。
- Single-Pass mHC：使用前一个子层产生的输入混合系数，减少当前数据依赖；不是删除 residual mixing 或让整个网络计算减半。
- DSpark：3 层 SWA-128 草稿网络一次提出 5 个草稿位置；主干预训练后单独训练草稿，后训练继续适配但草稿目标不回传主干；接受长度仍由验证决定。

## 关键概念

- [[../concepts/Causal Encoder-Decoder]]
- [[../concepts/CSA2]]
- [[../concepts/CSA-HCA]]
- [[../concepts/Shared KV Cache]]
- [[../concepts/KV Cache]]
- [[../concepts/mHC]]
- [[../concepts/DSpark]]

## 相关实体

- [[../entities/DeepSeek-V4.1-Flash]]
- [[../entities/DeepSeek-AI]]
- [[../entities/Engram]]

## 与现有 wiki 的关系

- 新建 CED、CSA2 和模型实体，为此前分散在 KV Cache 页及 [V4.1 深读](../../output/reports/DeepSeek-V4.1-Flash配置与技术报告深读.html) 中的内容建立导航。该 HTML 是派生报告，不增加独立来源数，不以本篇文章覆盖此前的一手核对结果。
- 局部更新 Shared KV Cache、KV Cache、CSA-HCA、mHC、DSpark、Engram 与 DeepSeek-AI，补充版本差异和来源回链。
- **格式冲突**：原文 §2.4.4 先称主 KV 采用 MXFP4，随后却描述 E2M1、每 16 通道一个 E4M3 scale、无二级全局 scale。与既有 KV Cache 页记录相比，应将主 KV 与 Indexer K 区分，保留原文不一致，不改 raw。
- **版本差异**：原版 Engram 含短卷积，本文 V4.1 版本省略；属于模型版本取舍，不删除旧机制。
- **解释边界**：CED Prefill 不是所有 Decoder 计算消失；“复用层只改 Q”“仅 Global KV 常驻 HBM”不可作为完整执行/显存描述，活跃请求仍需局部 SWA 状态。

## 待确认

- 补齐原始公式、表格及配图后，逐项核验表达力论证与视觉尺寸推导。
- FP4 格式命名以技术报告和绑定版本 kernel 为准；本次沿用已有布局核对，未重新读取完整论文或复测代码。
- HSI 的 Indexer 训练损失和候选限制细节；不把作者猜测当作实现。
- 约 420 tok/s、运行时约 1/4 与持久缓存约 1/8 的流量、精度、基线和部署条件；前者仅为作者使用观察，后两者也不能外推为所有工作负载的固定比例。
- 可学习 Q 感知映射、CoT 补偿与“递归 Transformer”的关系仍属作者研究设想或类比，不等同于已发表消融结论。
