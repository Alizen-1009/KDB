---
type: entity
entity_type: 框架
topic: GPU 编程
sources: 1
updated: 2026-06-12
---

# FlashInfer

## 一句话说明

面向 LLM serving / inference 的高性能 GPU kernel 库与 kernel generator，提供 attention、GEMM、MoE、sampling、通信等推理热点算子的统一接口。

## 类型

- 项目 / 推理 kernel 库

## 核心信息

- FlashInfer 的定位不是完整 serving 引擎，而是给 vLLM、SGLang、自研引擎等系统集成的 kernel layer。
- 典型能力包括 paged / ragged KV cache attention、prefill / decode / append attention、MLA、sparse / cascade attention、GEMM、fused MoE、sampling、RoPE、norm、activation、quantization 和通信相关算子。
- NVIDIA 已将部分高性能 LLM inference kernels 通过 FlashInfer 释放，包含来自 TensorRT-LLM 的 kernel，便于 vLLM、SGLang 和自研引擎集成。
- FlashInfer PR #4262 加入来源所称的 [[CAKE KDA]]：面向 B200/SM100a、BF16、`head_dim=128` 的 fused recurrent KDA prefill backend，使用 chunk 32、五级 SMEM producer/consumer pipeline 和 FP32 TMEM-resident state。
- 来源称该 PR 的六个 B200 workload 相对 MoonshotAI/FlashKDA 获得 `2.0512×` geometric-mean speedup；这是版本与 workload 绑定的来源声称。

## 相关概念

- [[CUDA Kernel]]
- [[PagedAttention]]
- [[Flash Decoding]]
- [[KV Cache]]
- [[MoE]]
- [[KDA]]
- [[Tensor Memory]]

## 相关实体

- [[TensorRT-LLM]]
- [[vLLM]]
- [[SGLang]]
- [[CAKE KDA]]
- [[FlashKDA]]
- [[NVIDIA Blackwell]]

## 相关来源

- [[../sources/REMINDER FF-KDA & CAKE KDA Highlights]]

## 备注

- FlashInfer API 和 backend 覆盖范围更新较快，引用具体 kernel 支持时需要带版本与硬件架构。
