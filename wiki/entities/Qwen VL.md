# Qwen VL

## 一句话说明

`Qwen VL` 是 Qwen 系列中的视觉语言模型方向；在本知识库当前语境中，它主要作为 2D/3D RoPE 与 [[M-RoPE]] 实现示例出现。

## 类型

- 项目 / 模型家族

## 核心信息

- `Qwen3VLVisionRotaryEmbedding` 被原文用作 2D 视觉 RoPE 的代码示例：图像 token 的 `row/col` 坐标会映射到不同旋转平面。
- Qwen2 VL 的 M-RoPE 示例按 `mrope_section` 把通道分给时间、高度、宽度三段，但原文指出这种 chunked 分段可能造成频率分布不均。
- Qwen2.5/Qwen3 VL 被原文列为 Interleaved-MRoPE 的改进方向：通过交错分配 `t/h/w` 轴，使频率布局更均衡。

## 相关概念

- [[../concepts/RoPE]]
- [[../concepts/M-RoPE]]

## 相关来源

- [[../sources/彻底搞懂RoPE计算原理：从1D到3D]]

## 冲突与备注

- 当前页面仅记录本次来源中涉及的位置编码线索；具体模型版本、源码接口和参数命名需按对应 `transformers` commit 或官方 repo 核实。
