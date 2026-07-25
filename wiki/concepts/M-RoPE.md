---
type: concept
topic: 位置编码
sources: 1
updated: 2026-05-17
---

# M-RoPE

## 定义

`M-RoPE`（Multimodal RoPE / 多维 RoPE）是在 [[RoPE]] 基础上面向视觉或多模态 token 的位置编码方式：把位置从 1D 序列索引扩展为 `t/h/w` 等多轴坐标，并把不同旋转平面分配给不同轴。

## 它解决什么问题

- 文本 token 只有单调递增的 1D 位置，图像和视频 token 还需要表达行、列和时间维度
- 直接把 `(t,h,w)` 展平成单一位置索引，容易丢失轴向结构，也可能和文本位置或其他视觉样本发生位置碰撞
- 多模态模型需要一套能兼容纯文本、图像和视频输入的位置编码机制

## 核心机制

- 对视觉 token，位置通常表示为 `(t, h, w)`：时间帧、高度索引、宽度索引
- 将 `head_dim` 中的二维旋转平面分配给不同坐标轴，各轴使用自己的位置索引与同一类 RoPE 频率计算相位
- Qwen2 VL 风格的 chunked M-RoPE 会按 `mrope_section` 把通道切为时间、高度、宽度三段，例如 `[16, 24, 24]`
- 纯文本输入可令 `t = h = w = m`，使三段通道都退化为标准 1D RoPE
- Qwen2.5/Qwen3 VL 风格的 Interleaved-MRoPE 将 `t/h/w` 交错分配到旋转平面中，避免某个轴长期占据高频或低频通道

## 关键权衡

- 按轴编码比展平 1D 更保留视觉时空结构，但实现中需要维护更复杂的 position id、通道分配和 broadcast 逻辑
- chunked 分段实现直观，但可能造成频谱不均；interleaved 分配更均衡，但 layout 和 kernel 适配更复杂
- 具体 `mrope_section`、merge size、视觉 token 排列方式会影响模型可见的位置关系，需要结合模型实现版本核实

## 相关实体

- [[../entities/Qwen VL]]

## 相关来源

- [[../sources/彻底搞懂RoPE计算原理：从1D到3D]]

## 相关概念

- [[RoPE]]
- [[Dual RoPE]]

## 研究备注

- 后续可补 Qwen2 VL、Qwen2.5 VL、Qwen3 VL 在 Hugging Face `transformers` 中的具体 `position_ids`、`mrope_section` 和 interleaved layout 差异。
- 可与 Gemma 系列视觉二维 RoPE、视频 RoPE 和长上下文 RoPE scaling 放在同一专题中比较。
