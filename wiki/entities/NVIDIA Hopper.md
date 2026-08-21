---
type: entity
entity_type: 硬件
topic: GPU 编程
sources: 1
updated: 2026-08-21
---

# NVIDIA Hopper

## 一句话说明

`NVIDIA Hopper` 是 NVIDIA 于 2022 年推出的数据中心 GPU 架构；本页当前聚焦 H100 路线的 TMA、异步 WGMMA、warp specialization 与 persistent work-tile pipeline。

## 类型

- GPU 架构 / 硬件平台

## 核心信息

- 来源以 H100 80GB 类配置代表 Hopper，列出 132 SM、50MB L2、约 3.35TB/s HBM 带宽等数据；这些不是所有 Hopper SKU 的统一规格。
- TMA 以 tensor tile 为单位在 global memory 与 shared memory 之间异步搬运数据，减少逐线程地址生成和 load 指令负担。
- WGMMA 允许 warpgroup 异步提交 Tensor Core MMA，促进 TMA producer、MMA consumer 与 epilogue/store warps 的职责分离。
- [[../concepts/Persistent Kernel]] 让已驻留 CTA 连续处理多个 work tiles，摊销后续 tile 的 pipeline/descriptor/调度成本，并支持前后 tiles 的 load/compute/store overlap。
- 来源把 Hopper 的主要压力归纳为 WGMMA consumers 的 register footprint，以及 Tensor Core pipeline 与 CUDA Core/ALU epilogue 对执行资源和数据路径的协同。
- 第四代 Tensor Core 新增 FP8 `E4M3/E5M2` 输入并支持 FP16/FP32 accumulator；Transformer Engine 通过软件与硬件协同按 tensor/layer 管理 FP8 与 16-bit 计算、scale 和 recasting。
- Hopper 首次把 thread block cluster 纳入 CUDA 层级：cluster 内 blocks 保证在同一 GPC 的多个 SM 上并发调度，并可通过 Distributed Shared Memory 对其他 block 的 SMEM 执行 load/store/atomic。
- Asynchronous transaction barrier 同时跟踪线程到达与异步 transaction 完成，为 TMA、cluster 数据交换和 warp-specialized producer-consumer pipeline 提供同步。
- Compute capability 9.0 的 unified L1/texture/SMEM 为 256 KB/SM，可配置 shared memory 最高 228 KB/SM、单 block 最多 227 KB；64 warps/SM 和 64K 32-bit registers/SM 上限与 A100 同量级，因此大 SMEM/WGMMA register footprint 仍会压低 occupancy。
- Hopper 还增加 2× FP32 operations/cycle/SM、DPX 动态规划指令、HBM3/更大 L2/inline compression，以及第四代 NVLink、PCIe Gen 5、第二代 MIG 和 confidential computing；其中 HBM、L2、链路数量与带宽必须绑定 H100 SXM/PCIe SKU。
- [[../concepts/Persistent Kernel]] 不是 Hopper 新增的硬件特性，而是长期驻留 worker 循环处理多个逻辑 tiles 的 kernel 组织方式；TMA、WGMMA、cluster 和 transaction barrier 让这种方式更适合构建深度异步流水。

## 相关概念

- [[../concepts/Persistent Kernel]]
- [[../concepts/GPU执行模型]]
- [[../concepts/CUDA内存层次]]
- [[../concepts/Tiling]]
- [[../concepts/Occupancy]]
- [[../concepts/FlashMLA]]

## 相关实体

- [[NVIDIA Ampere]]
- [[NVIDIA Blackwell]]

## 相关来源

- [[../sources/译 NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell]]

## 外部一手资料

- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

## 冲突与备注

- 来源称 Hopper-specific 代码“不向前兼容”Blackwell；更稳妥地说，WGMMA-specific implementation、native cubin 和架构调度假设可能需要 SM100 port，不能外推为所有 CUDA/PTX 程序不兼容。
- TMA/WGMMA 的具体 shape、barrier、descriptor、register 与 SMEM 限制必须绑定 PTX/CUDA/CUTLASS 版本。
- 完整 GH100、H100 SXM5 与 H100 PCIe 的 SM、HBM、L2 和互联配置不同；不能把完整芯片的 144 SM/60MB L2 与 SXM5 的 132 SM/50MB L2 混用。
- NVIDIA 架构文章中的相对 speedup 来自官方特定 workload 或初期产品口径，不代表任意 LLM kernel 的实测加速。
