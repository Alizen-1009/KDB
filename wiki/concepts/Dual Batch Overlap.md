---
type: concept
topic: 并行与分布式
sources: 2
updated: 2026-07-26
---

# Dual Batch Overlap

## 定义

`Dual Batch Overlap`（DBO）是把一次 MoE forward 的 token batch 切成两个 microbatch，并交错推进二者的 dispatch、expert compute 与 combine，以一个 microbatch 的工作隐藏另一个的通信等待。

## 核心机制

```text
MB0: Dispatch -> Expert Compute -> Combine
MB1:      Dispatch -> Expert Compute -> Combine
```

vLLM 来源描述的执行流程包括：跨 ranks 通过 collective 判断 microbatching 是否有利；主线程创建 microbatch worker 并完成 CUDA Graph capture；MoE All-to-All 抽象在等待 GPU 工作时让出控制权，在两个 worker 间切换。

DBO 不打破同一 microbatch 的数据依赖，只做跨 microbatch overlap。它也不要求把 dispatch/GEMM/combine 融成一个 kernel。

## 与 MegaMoE 的区别

| 维度 | DBO | [[MegaMoE]] |
| --- | --- | --- |
| 层级 | Serving runtime / worker 调度 | 融合 MoE kernel / 算子流水 |
| 粒度 | 两个 microbatch | 同一 batch 内的 expert waves |
| 重叠 | MB0 与 MB1 | 不同 wave 的五阶段 |
| Kernel 边界 | 可继续调用独立 backend/GEMM | 尝试融合 dispatch、L1、activation、L2、combine |
| 主要风险 | 两批资源争用、状态复杂度 | 小 GEMM、碎消息、wave 调度成本 |

二者概念上可叠加，但若 MegaMoE 已同时占满 Tensor Core、HBM 与互联，DBO 可能增加争用、cache 干扰和功耗，而不是继续提速。

## 关键权衡

- 高 EP degree、通信等待显著时更可能受益。
- Batch 太小或 microbatch 切分后 GEMM 太碎时可能变慢。
- 需要维护两份 microbatch 输入、执行状态和 graph/synchronization 生命周期。
- 阈值应按 prefill/decode token 数和具体 backend 调整，不是固定通用参数。

## 相关实体

- [[../entities/vLLM]]
- [[../entities/DeepEP]]

## 相关来源

- [[../sources/vLLM Large Scale Serving DeepSeek @ 2.2k toksH200 with Wide-EP]]
- [[../sources/MegaMoE — 让 all-to-all 消失]]

## 相关概念

- [[Wide Expert Parallelism]]
- [[Expert Parallelism]]
- [[通信-计算重叠]]
- [[MegaMoE]]

## 研究备注

- MegaMoE 来源只用于机制对比；DBO 实现细节的一手来源是 vLLM 官方博客。
