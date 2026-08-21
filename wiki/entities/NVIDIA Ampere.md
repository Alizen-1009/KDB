---
type: entity
entity_type: 硬件
topic: GPU 编程
sources: 1
updated: 2026-08-21
---

# NVIDIA Ampere

## 一句话说明

`NVIDIA Ampere` 是 NVIDIA 于 2020 年推出的一代 GPU 架构；本页当前聚焦 A100 路线通过 `cp.async` 把 global→shared 数据搬运与 MMA 在单 CTA 内重叠的编程模型变化。

## 类型

- GPU 架构 / 硬件平台

## 核心信息

- 来源以 A100 80GB 类配置代表 Ampere，列出 108 SM、40MB L2、约 2.0TB/s HBM 带宽等数据；这些是具体 SKU/形态口径，不代表所有 Ampere 产品。
- `cp.async` 允许线程异步发出 global→shared copy，不必先经普通寄存器中转并同步等待整个 tile，适合构建 SMEM 多 stage pipeline。
- 典型 CTA 可在当前 tile 执行 MMA 时预取下一 tile，从主要依赖多 resident CTAs 隐藏延迟，扩展为单 CTA 内显式 load/compute overlap。
- 更深 pipeline stage 会增加 SMEM 占用；tile、stage、register 和 occupancy 仍需联合调优。

## 相关概念

- [[../concepts/CUDA内存层次]]
- [[../concepts/GPU执行模型]]
- [[../concepts/Tiling]]
- [[../concepts/Persistent Kernel]]
- [[../concepts/Occupancy]]

## 相关实体

- [[NVIDIA Hopper]]
- [[NVIDIA Blackwell]]

## 相关来源

- [[../sources/译 NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell]]

## 冲突与备注

- “Ampere 规格”必须区分 A100/GA100 与其他数据中心、专业和消费级 Ampere GPU；当前来源只足以支持 A100 风格的 kernel 演进概览。
- 来源把 Pre-Ampere latency hiding 与 double buffering 简化为多 CTA 驻留；术语需按具体 kernel 解释。
