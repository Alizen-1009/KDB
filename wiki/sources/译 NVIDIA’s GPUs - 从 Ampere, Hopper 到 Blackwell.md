---
type: source
source_kind: 文章
topic: GPU 编程
updated: 2026-08-21
---

# [译] NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell

## 来源信息

- 标题：[译] NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell
- 译者：cervol
- 日期：2025-12-26（页面编辑时间；frontmatter 未填写 published）
- 类型：翻译文章 / NVIDIA 数据中心 GPU 架构与 kernel 流水线概览
- 原始文件：[[../../raw/articles/译 NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell.md]]
- 原始链接：原文 clipping 只保留“原文链接:”文字，未保留实际 URL
- 译文链接：https://zhuanlan.zhihu.com/p/1987901646806729625

## 2-3 条核心摘要

- 文章把三代 GPU kernel 优化主线概括为流水线范围的扩展：[[../entities/NVIDIA Ampere]] 用 `cp.async` 在单 CTA 内重叠 HBM→SMEM load 与 MMA；[[../entities/NVIDIA Hopper]] 用 TMA、异步 WGMMA 和 [[../concepts/Persistent Kernel]] 把 overlap 扩展到多个 work tiles；[[../entities/NVIDIA Blackwell]] 用 `tcgen05` 与 [[../concepts/Tensor Memory|TMEM]] 承载 accumulator，进一步让 output write-back 进入 load/compute/store 三阶段流水。
- 文章强调性能不只来自 FLOPs 和带宽增长，也来自 latency-hiding 模型变化：从依赖多 resident CTA/warp 隐藏停顿，到 CTA 内异步预取，再到 warp-specialized persistent pipeline 与独立 TMEM accumulator 生命周期。
- 文中列出的 `80GB/108 SM/40MB L2/2.0TB/s`、`80GB/132 SM/50MB L2/3.35TB/s`、`148 SM/192MB L2/7.672TB/s` 实际分别接近 A100 80GB、H100 与 B200 的特定 SKU 口径，不是 Ampere/Hopper/Blackwell 整个架构家族的统一规格。

## 值得关注的论断

- Ampere 的核心变化是线程发出 `cp.async` 后可以继续执行，使输入搬运与 MMA 在单 CTA 内形成软件流水；代价是 stage 数、SMEM 占用、tile 大小与 occupancy 仍需联合权衡。
- Hopper 的 TMA 把 tensor tile 的地址生成与 bulk transfer 从逐线程 load 中抽离，WGMMA 允许异步提交 warpgroup MMA；二者促进 load/MMA/store warp specialization 和长期驻留 CTA 跨 tiles 复用 pipeline。
- Blackwell 的 `tcgen05` 把大型 MMA accumulator 放入 TMEM，降低 Hopper WGMMA consumer 长期占用普通寄存器的压力，使 Tensor Core 计算与 CUDA Core/epilogue 更容易并行，但 TMEM 指令和访问模式更受限制。
- [[../concepts/Persistent Kernel]] 仍需要一次 kernel launch；其主要收益是让已驻留 CTA/cluster 连续处理多个逻辑 tiles，摊销后续 tile 的 CTA/pipeline 初始化和调度成本，并允许跨 tile overlap；只有原方案确实跨多个 kernels 时才同时减少 kernel 边界，而不是“完全没有 launch”。

## 架构编程模型对比

| 维度 | Ampere / A100 路线 | Hopper / H100 路线 | Blackwell / B200 路线 |
| --- | --- | --- | --- |
| 输入搬运 | `cp.async` | TMA | TMA |
| 主 MMA | 同步 MMA | 异步 WGMMA | `tcgen05.mma` |
| Accumulator | 普通 register 为主 | 普通 register 为主 | TMEM 为主 |
| 典型 overlap | CTA 内 load↔compute | persistent tiles 间 load/compute/store | load/compute/store 与 TMEM epilogue 解耦 |
| 主要压力 | SMEM stage、tile、occupancy | WGMMA consumer register、Tensor Core/ALU 协同 | TMEM/SMEM layout、显式生命周期、有限指令路径 |
| 典型调度 | 多 CTA 或单 CTA staged pipeline | warp specialization + persistent scheduler | warp specialization + TMEM + cluster/2-CTA 特化 |

## 来源中的规格口径

| 来源标签 | 来源列出的数据 | 更稳妥的归档口径 |
| --- | --- | --- |
| Ampere | 80GB HBM、2.0TB/s、108 SM、40MB L2、每 SM 4 Tensor Cores、192KB L1/SMEM、65536 registers | 接近 A100 80GB 特定形态，不能推广到所有 GA100/GA10x |
| Hopper | 80GB HBM、3.35TB/s、132 SM、50MB L2、每 SM 4 Tensor Cores、256KB L1/SMEM、65536 registers | 接近 H100 特定形态；L1/shared 的物理总量、可配置容量和 per-block limit 应分开核实 |
| Blackwell | 7.672TB/s、148 SM、192MB L2、每 SM 4 Tensor Cores、228KB L1/SMEM、65536 registers、256KB TMEM | 接近 B200 特定形态；不能代表 B100、GB200 或其他 Blackwell SKU |

## 来源中的归一化峰值表

文章以 A100 为 baseline 给出：

| 指标 | A100 | H100 | H200 | B100 | B200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Peak memory bandwidth | `1.0×` | `1.6×` | `2.4×` | `3.9×` | `3.9×` |
| NVLink bandwidth | `1.0×` | `1.5×` | `1.5×` | `3.0×` | `3.0×` |
| Peak dense BF16 TFLOPS | `1.0×` | `3.2×` | `3.2×` | `5.6×` | `7.2×` |
| Peak dense FP8 TFLOPS | N/A | `1.0×` | `1.0×` | `1.8×` | `2.3×` |

这些是来源汇总的峰值比值，不是实测应用性能。卡型、SXM/PCIe 形态、功耗、稀疏/稠密口径、boost clock 和数据手册版本需逐项核对。

## 四阶段流水线叙事

### Pre-Ampere

文章把单 CTA 描述为 load→wait→MMA→wait→store，并认为主要依靠多个 CTA 同时驻留来隐藏延迟。该叙事适合说明 occupancy-based latency hiding，但把它称为 double buffering 过于宽泛：double buffering 也常指单 CTA 内两套 SMEM buffers，不能与“多个 resident CTAs”直接等同。

### Ampere

`cp.async` 允许 global→shared copy 异步发出，CTA 可在当前 tile MMA 时预取后续 tile。图中仍画出了 work blocks 之间的空隙，并将其归因于下一 block/CTA 的调度与初始化成本。

### Hopper

TMA 和 WGMMA 支持更明确的 producer/consumer 分工。图中用“Only ONE block launched”表达一个长期驻留 CTA 连续处理多个 tiles；更准确地说，它是一次 kernel launch 中的 persistent CTA，而不是 GPU 只启动一个全局 block 或完全消除 launch。

### Blackwell

B200 图把 input load、MMA 与 output store 排成跨 tiles 的三阶段重叠：tile N+1 load、tile N compute、tile N-1 store。TMEM 减少 accumulator 对普通 RF 的长期占用，但实际 write-back 通常仍经过 TMEM→RF/SMEM→GMEM 的显式搬运和 epilogue，不能把 TMEM 当作可直接任意 store global 的通用缓存。

## 关键概念

- [[../concepts/Persistent Kernel]]
- [[../concepts/GPU执行模型]]
- [[../concepts/CUDA内存层次]]
- [[../concepts/Tensor Memory]]
- [[../concepts/Megakernel]]
- [[../concepts/Tail Effect]]

## 相关实体

- [[../entities/NVIDIA Ampere]]
- [[../entities/NVIDIA Hopper]]
- [[../entities/NVIDIA Blackwell]]

## 与现有 wiki 的关系

- 新建 [[../entities/NVIDIA Ampere]] 与 [[../entities/NVIDIA Hopper]]，并扩充 [[../entities/NVIDIA Blackwell]] 的代际比较视角。
- 新建 [[../concepts/Persistent Kernel]]，区分 persistent work-tile kernel、普通 kernel 和 [[../concepts/Megakernel]]。
- 更新 [[../concepts/GPU执行模型]] 与 [[../concepts/CUDA内存层次]]，整理 `cp.async → TMA/WGMMA → tcgen05/TMEM` 的数据流演进。
- 更新 [[../concepts/Tensor Memory]]、[[../concepts/Megakernel]] 与 [[../concepts/Tail Effect]]，补充 accumulator residency、跨 tile 执行和尾部权衡。
- 与现有 wiki 的 Blackwell TMEM/CLC 资料没有直接冲突，但本文对 launch、兼容性和 TMEM store 路径的表述更简化，需显式保留边界。

## 待确认

- 原始英文文章 URL 与版本未保留，无法检查翻译删改和原始发布日期。
- Pre-Ampere “多个 CTA 等于 double buffering”属于概念简化；需要与具体架构和 kernel 实现区分。
- “Hopper 特性代码不向前兼容 Blackwell”不能作为 CUDA/PTX 的一般结论；WGMMA-specific kernel、native cubin、PTX target 和 JIT compatibility 需要分别讨论。
- WGMMA “与 Tensor Core 计算重叠”的准确含义需结合异步提交、warpgroup pipeline、scoreboard 和具体硬件吞吐解释，不能理解为任意 MMA 无限并发。
- Blackwell TMEM write-back 图省略了 RF/SMEM/epilogue 中间路径。
- 所有容量、带宽、SM 数、L2、Tensor Core 数及峰值比值都需回到 A100/H100/H200/B100/B200 官方数据手册逐项验证。
