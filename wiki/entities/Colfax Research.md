---
type: entity
entity_type: 组织
topic: GPU 编程
sources: 1
updated: 2026-08-21
---

# Colfax Research

## 一句话说明

`Colfax Research` 是发布高性能计算、CUDA、CUTLASS 与 GPU kernel 实现分析的技术研究团队 / 内容来源。

## 类型

- 组织 / 技术研究团队

## 核心信息

- 当前收录来源围绕 [[../entities/NVIDIA Blackwell|NVIDIA Blackwell]] 的 [[Cluster Launch Control]]，比较 single-tile、静态 persistent 与动态 persistent tile scheduling。
- 文章结合 NVIDIA PTX、CUDA Programming Guide、CUTLASS 文档与 [[CuTe DSL]] 示例，解释 CLC 的指令语义、pipeline 实现和同步约束。
- 团队还在 B200 上比较 CLC 与静态 persistent、single-tile GEMM scheduler；实验结果应绑定具体 shape、dtype、MMA tile、cluster shape 和代码版本。

## 相关概念

- [[Cluster Launch Control]]
- [[CuTe DSL]]
- [[Tiling]]
- [[GPU执行模型]]

## 相关来源

- [[../sources/Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs]]

## 冲突与备注

- 当前仅依据一篇来源建立实体页；作者列表、组织边界及更多项目需后续来源补充。
- 文章包含对均衡 GEMM 中 L2 hit rate 差异的未决分析，不能把推测写成已确认硬件机制。
