# batch1 decode 融合 RoPE、RMSNorm 与 GEMV 后访存不合并如何优化？

- 整理状态：已整理
- 题目出处：用户提供的 CUDA 截图系列摘要，第 1 题。

## 30 秒回答

先以 NCU 确认具体指令的请求、sector 和实际带宽瓶颈，再按张量布局重排线程映射，使同一 warp 的 lane 访问连续对齐地址；合适时向量化加载，寄存器内局部累加并做 warp reduce，适量增加独立累加器隐藏延迟。融合降低中间读写和 launch，但可能增加寄存器、同步或破坏合并，最终要与未融合基线做正确性和时延比较。

## 深入解释

向量化减少指令数不自动保证跨 lane 合并：画出每 lane 每次访问的地址，检查对齐、尾部和事务覆盖；根据实际矩阵布局选择 warp 负责行或其他映射。ILP 与展开可能增加寄存器和 spill，逐项 profile。题设 45% 到 90% 只是待验证目标，不是保证，也不能脱离指标定义、硬件与输入视为实测结果。

## 关联知识

- [[../../../../wiki/concepts/内存合并访问|内存合并访问]]
- [[如何实现高性能行主序 GEMV|行主序 GEMV]]

## 参考来源与待核实

- [CUDA Best Practices：Coalesced Access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory)；具体融合顺序、布局和 NCU 数据未提供，需按真实模型验证。

## 所属题单

- [[../sets/CUDA 截图系列15题|CUDA 截图系列15题]]
- [[../README|秋招问题汇总]]
