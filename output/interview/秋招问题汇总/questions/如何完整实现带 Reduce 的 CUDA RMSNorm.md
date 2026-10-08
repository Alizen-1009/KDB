# 如何完整实现带 Reduce 的 CUDA RMSNorm？

- 整理状态：已整理
- 题目出处：用户提供的“千问 C 端 AI Infra 一面”Coding 题

## 问题背景

要求写出完整 CUDA 实现，包括沿 hidden dimension 计算平方和的 Reduce、`rsqrt` 归一化、权重缩放、边界处理与 kernel launch；后续整理时应分别验证正确性、数值精度、不同 hidden size 和性能。

## 30 秒回答

每一行先沿 hidden dimension 求 `sum(x²)`，计算 `inv=rsqrt(sum/H+eps)`，再逐元素写 `y=x*inv*weight`。基线实现可让一个 block 处理一行：线程跨步读取并累加 FP32 局部和，先 warp 内归约，再共享内存合并各 warp，广播 inv 后写回。

## 深入解释

要处理 H 非 2 的幂、行 stride、weight 广播和输入/输出 dtype；低精度输入建议 FP32 累加。大 H 或小行数时再考虑多 block/行，避免额外全局归约成本。

## 参考来源与待核实

- [[../../../../wiki/concepts/RMSNorm|RMSNorm]]

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
