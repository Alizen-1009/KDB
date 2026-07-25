---
type: entity
entity_type: 框架
topic: GPU 编程
sources: 0
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

## 相关概念

- [[CUDA Kernel]]
- [[PagedAttention]]
- [[Flash Decoding]]
- [[KV Cache]]
- [[MoE]]

## 相关实体

- [[TensorRT-LLM]]
- [[vLLM]]
- [[SGLang]]

## 备注

- FlashInfer API 和 backend 覆盖范围更新较快，引用具体 kernel 支持时需要带版本与硬件架构。
