---
type: entity
entity_type: 项目
topic: 并行与分布式
sources: 0
updated:
---

# DeepEP

## 一句话说明

`DeepEP` 是 DeepSeek 开源的 MoE Expert Parallel 通信库，为 token dispatch/combine 提供高吞吐与低延迟 All-to-All kernels，并被 vLLM 集成为 Wide-EP backend。

## 类型

- 项目 / MoE 通信库

## 核心信息

- 面向 MoE token dispatch/combine，而非完整 Router、expert GEMM 或 serving runtime。
- vLLM 官方博客将其作为 Wide-EP 可选 backend，并指出 PD 分离可让 Prefill/Decode 分别选择适合吞吐或延迟的路径。
- 与 [[../entities/NCCL Extensions|NCCL Extensions]] 的 `nccl_ep` 属于相邻生态位，但 API、硬件路径和支持矩阵不能混为一谈。

## 相关概念

- [[Expert Parallelism]]
- [[Wide Expert Parallelism]]
- [[Dual Batch Overlap]]
- [[通信-计算重叠]]

## 相关来源

- [[../sources/vLLM Large Scale Serving DeepSeek @ 2.2k toksH200 with Wide-EP]]

## 冲突与备注

- 当前仅由 vLLM 官方博客间接摄入，尚未 ingest DeepEP repo；精确 kernel、拓扑和 low-latency/high-throughput 能力需读官方仓库。
