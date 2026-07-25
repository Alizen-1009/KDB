---
type: source
source_kind: 文章
topic: 位置编码
updated: 2026-05-17
---

# 彻底搞懂RoPE计算原理：从1D到3D

## 来源信息

- 标题：彻底搞懂RoPE计算原理：从1D到3D
- 作者：kaiyuan
- 日期：2026-04-08
- 类型：文章 / 机制讲解 / 多模态位置编码实现导读
- 原始文件：[[../../raw/articles/彻底搞懂RoPE计算原理：从1D到3D|彻底搞懂RoPE计算原理：从1D到3D]]
- 原始链接：https://mp.weixin.qq.com/s/8_0V6Yxw-_03lCY3ujPVtA

## 2-3 条核心摘要

- 文章从二维点积与旋转矩阵出发解释 [[RoPE]]：`Q/K` 在各自绝对位置上旋转后再做内积，旋转角差会转化为相对位置项，因此 attention score 可以自然依赖 `n - m`。
- 多维 RoPE 本质上是把 `head_dim` 拆成多个二维旋转平面；数学推导常用相邻维度配对，常见工程实现则把前半维和后半维配对，并通过 `rotate_half`、`cos/sin` 广播或复数乘法避免显式构造旋转矩阵。
- 文章进一步把 1D RoPE 推广到视觉 2D/3D 场景：2D RoPE 可把旋转平面分配给 `row/col`，[[M-RoPE]] 则把通道维分配给 `t/h/w` 三个轴，并通过 interleaved layout 改善 Qwen2 VL chunked 分段带来的频率分布不均。

## 值得关注的论断

- RoPE 的关键不是单独维护位置向量，而是把位置写进 `Q/K` 的几何关系；`(R_m q_m)^T (R_n k_n) = q_m^T R_{n-m} k_n` 是理解相对位置建模的核心等式。
- `theta_i = 10000^{-2i/d}` 提供多尺度频率组合：高频更敏感于局部变化，低频覆盖更长尺度；长上下文能力仍取决于频率设计、scaling 策略和模型训练分布，不能只由 RoPE 本身保证。
- 多模态 RoPE 不宜简单把 `(t,h,w)` 展平成一个 1D 索引，因为这可能引入模态或样本间位置碰撞；按轴分配旋转平面能更直接保留时空坐标结构。

## 关键概念

- [[RoPE]]
- [[M-RoPE]]

## 相关实体

- [[../entities/kaiyuan]]
- [[../entities/Qwen VL]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`RoPE`
- 会创建哪些概念页：`M-RoPE`
- 会创建哪些实体页：`kaiyuan`、`Qwen VL`
- 是否存在冲突：与现有 `RoPE` / `Dual RoPE` 页面无直接冲突；本来源主要补强多维旋转平面、`rotate_half` 实现配对，以及视觉 2D/3D RoPE。

## 待确认

- 原文部分代码块存在排版损坏，例如 `rotate_half` 片段中混入孤立字符，后续若要做代码级复现，应以 Hugging Face `transformers` 对应版本源码为准。
- 文中关于远程衰减和长上下文外推的说明适合作为机制直觉；若要写长上下文专题，仍需补充 RoPE scaling、NTK-aware、YaRN、LongRoPE 等更专门资料。
