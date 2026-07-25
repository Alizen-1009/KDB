---
type: source
source_kind: 文章
topic: KV Cache
updated: 2026-07-25
---

# SGLang的KDA管理与Prefix Cache难题

## 来源信息

- 标题：SGLang的KDA管理与Prefix Cache难题
- 作者：zR（探索方向中）
- 日期：2026-07-25
- 类型：截图文章 / 代码解读
- 原始文件：[[../../raw/articles/SGLang的KDA管理与Prefix Cache难题.md]]
- 原始链接：https://www.xiaohongshu.com/explore/6a5f043f0000000006012450

## 2-3 条核心摘要

- 混合 MLA/KDA 模型同时维护两类历史：全注意力层按 token 保存 latent KV；KDA 线性注意力层按请求、按层保存固定大小的递归状态。SGLang 用 Token KV Pool 与 MambaPool 分别承载物理资源，并在统一 Radix Tree 上协调生命周期。
- KDA/GDN 层的持久状态通常不只有长期矩阵状态，还包含 causal depthwise convolution 所需的 Conv State。Conv State 保存最近 `kernel_size-1` 个卷积输入特征，用于继续下一 token 的局部卷积；矩阵状态则把从序列开头到当前边界的长期历史压缩为固定大小的关联记忆。
- 普通 KV Cache 的 token/page 行可以在已保存边界独立复用；递归状态却只代表“推进到某一步后的聚合结果”，通常无法从较晚状态逆推出较早状态。因此 SGLang 需要在部分 Radix 节点保存同时包含各线性层 Conv State 与矩阵状态的 checkpoint，未命中 checkpoint 的区间要重新 prefill。

## 值得关注的论断

- Prefix Cache 的根本矛盾是状态粒度不同：KV Cache 是 token-addressable 的历史记录，KDA/GDN State 是 request-addressable 的递归聚合结果。统一缓存命中必须截断到两类状态都可恢复的最近边界。
- checkpoint 越密，前缀恢复越精细、重算越少，但状态显存越高；checkpoint 越稀则相反。线性注意力把随上下文增长的 KV 显存换成固定状态，也把 Prefix Cache 变成 checkpoint 显存与局部重算之间的交换。
- GDN 的长期矩阵状态更稳妥地称为 recurrent matrix state、fast-weight matrix 或关联记忆矩阵。本文截图将 KDA 的 `S` 称为递归状态；仅凭这份二手资料不足以确认它在官方数学定义中是否严格属于“逆协方差矩阵”。
- MTP 验证不能让 rejected draft token 污染主递归状态：来源描述的做法是先写暂存区，验证后按实际接受长度提交。具体 API 与状态布局依赖 SGLang 版本。

## 关键概念

- [[../concepts/线性注意力递归状态]]
- [[../concepts/递归状态 Prefix Caching]]
- [[../concepts/Chunked Gated Delta Rule]]
- [[../concepts/Prefix Caching]]
- [[../concepts/KV Cache]]
- [[../concepts/混合注意力]]
- [[../concepts/Speculative Decoding]]

## 相关实体

- [[../entities/SGLang]]

## 与现有 wiki 的关系

- 新增 [[../concepts/线性注意力递归状态]]，区分 Conv State 与长期矩阵状态，并解释 GDN 的状态用途与更新变量。
- 新增 [[../concepts/递归状态 Prefix Caching]]，解释 KDA/GDN checkpoint、共同恢复边界和重算权衡。
- 更新 [[../concepts/Chunked Gated Delta Rule]]、[[../concepts/Prefix Caching]]、[[../concepts/KV Cache]]、[[../concepts/混合注意力]]、[[../concepts/Speculative Decoding]] 与 [[../entities/SGLang]]。
- 未发现与现有 wiki 的直接冲突；现有 GDN 页只记录了 matrix state，本次补足 Conv State 与 Prefix Cache 语义。

## 待确认

- 本文是对 SGLang 主分支代码的二手截图解读，`HybridLinearKVPool`、`MambaPool`、`UnifiedRadixCache`、extra-buffer tracking 与 copy-on-write 的精确名称和行为需绑定源码 commit 核实。
- KDA 官方状态 `S` 的严格数学定义、形状和更新公式需要一手 KDA/Kimi K3 技术资料；不能直接套用 GDN 的 delta-rule 公式。
- 评论中“GDN 与 KDA 共用同一套 MambaPool + checkpoint 机制”与模型支持范围尚未用官方源码独立核实，正文仅保留为来源线索。
