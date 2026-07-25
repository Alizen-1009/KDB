---
type: entity
entity_type: 项目
topic: GPU 编程
sources: 1
updated: 2026-06-12
---

# Megakernels

## 一句话说明

`Megakernels` 是 HazyResearch 开源的低延迟 LLM forward megakernel 实现线索，用于将 Llama-1B forward pass 融合进单个 GPU kernel。

## 类型

- 项目 / 代码仓库

## 核心信息

- 来源文章称该仓库开源了 Llama-1B megakernel 的相关代码。
- 核心实现思路包括 on-GPU interpreter、per-SM instruction schedule、shared memory paging 和基于 counter 的 kernel 内显式同步。
- 目标场景是 Llama-3.2-1B、batch size 1、低延迟 decode；不应默认外推到高 batch serving 或多 GPU 场景。

## 相关概念

- [[Megakernel]]
- [[CUDA Kernel]]
- [[算子融合]]
- [[CUDA内存层次]]
- [[Programmatic Dependent Launch]]

## 相关来源

- [[../sources/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B]]

## 冲突与备注

- 仓库尚未作为 `raw/repos/` 来源 ingest；当前页面只记录博客提供的项目线索和高层机制。
