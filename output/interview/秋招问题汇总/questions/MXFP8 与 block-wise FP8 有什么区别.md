# MXFP8 与 block-wise FP8 有什么区别？

- 整理状态：已整理
- 题目出处：用户提供的 Kimi AI Infra 一面复盘

## 30 秒回答

两者都给 FP8 数据配局部 scale，但不能只按“是否分块”区分。以 NVIDIA Transformer Engine 的 recipe 为例，传统 FP8 Blockwise Scaling 每 128 个元素的一维块，或每 `128×128` 的二维块共享 scale，可在 Hopper 上使用；MXFP8 是 Blackwell 原生 microscaling 格式，每连续 32 个 FP8 元素配一个 E8M0 scale。MXFP8 粒度更细，通常能减小局部量化误差，但 scale 数量、布局转换和硬件支持也不同。比较性能时要把量化、scale 搬运/重排和 GEMM 一起测，不能仅凭更细粒度断定更快。

## 深入解释

“scale 与 MMA overlap”是具体 kernel 的流水优化，不是 MXFP8 与 block-wise FP8 的定义差异；还要确认目标 kernel 是否使用原生 block-scaled MMA。

## 关联知识

- [[../../../../wiki/concepts/混合精度训练与推理|混合精度训练与推理]]
- [[../../../../wiki/entities/NVIDIA Blackwell|NVIDIA Blackwell]]

## 参考来源与待核实

- [NVIDIA Transformer Engine MXFP8](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html)；[FP8 Blockwise Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_blockwise_scaling/fp8_blockwise_scaling.html)。
- “block-wise FP8”可泛指多种粒度；这里特指 Transformer Engine 的 Float8BlockScaling recipe。

## 所属题单

- [[../sets/Kimi AI Infra一面|Kimi AI Infra 一面]]
- [[../README|秋招问题汇总]]
