---
type: source
source_kind: 文章
topic: GPU 编程
updated: 2026-08-21
---

# PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化

## 来源信息

- 标题：PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化
- 作者：阿里云大数据 AI / 阿里云 PAI 团队
- 日期：2026-05-22（页面编辑时间；frontmatter 未填写 published）
- 类型：文章 / FlashAttention-4 kernel 实现与性能分析
- 原始文件：[[../../raw/articles/PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化.md]]
- 原始链接：https://zhuanlan.zhihu.com/p/2041223880157679871
- 代码线索：Dao-AILab/flash-attention PR #2412

## 2-3 条核心摘要

- 文章将 FlashAttention-4 在 `head_dim=256` 下的主要障碍归因于 [[../concepts/Tensor Memory|Tensor Memory（TMEM）]] 容量：Forward 的 `O` accumulator 与 Backward 的 `dQ/dK/dV` 会随 head dimension 增大，原 `head_dim=128` 的 tile、stage 和双缓冲方案无法直接复用。
- Forward 仍使用大 tile `128×128×256`，但根据“单次 MMA 工作量翻倍、Softmax 工作量基本不变”的新比例，把 Q stage 从 2 降到 1，只在 TMEM 中保留一个 O tile；流水重叠从不同 Q tile 的 ping-pong 改为同一 Q 下不同 K tile 的 ping-pong，并用 2-CTA 协作分摊 SMEM 压力。
- Backward 将原单 kernel 拆成 `dQ kernel` 与 `dKdV kernel`，用 7 次 MMA 替代原方案约 5 次 MMA，以更多重算换取更低的单 kernel TMEM/SMEM 峰值和更大的有效 tile。`dQ` 采用 `128×128`、Outer-Q/Inner-K，`dKdV` 采用 `128×64`、Outer-K/Inner-Q。

## 值得关注的论断

- 大 head dimension 不能机械沿用较小 head dimension 的 pipeline stage 数；stage 应由各阶段 microbenchmark、Tensor Core/CUDA Core 耗时比例、数据依赖和 TMEM/SMEM/RF 容量联合决定。
- 2-CTA / CTA Pair 通过 DSMEM 共享操作数并扩大协作 tile，可降低单 CTA 的 SMEM 压力；但它同时引入 cluster 内同步、物理 layout 重映射和更复杂的数据交换。
- Backward 双 kernel 方案体现了片上资源约束下的典型权衡：即使增加重算和一次额外 kernel 边界，只要换来的大 tile、Tensor Core 利用率和访存效率足够高，整体仍可能更快。
- 来源称实现已通过 PR #2412 合入 `Dao-AILab/flash-attention`，并服务 Qwen3.5 等大 head dimension 模型的千卡训练；该工程状态和适用版本需以仓库 commit 与发布说明为准。

## HeadDim 128 与 256 的流水差异

| 维度 | 社区 FA4 `head_dim=128` | PAI-FA `head_dim=256` |
| --- | --- | --- |
| Forward 掩盖关系 | 来源描述约 2 MMA 掩盖 1 Softmax | 来源描述约 1 MMA 掩盖 1 Softmax |
| Q/O stage | 可使用多 stage / 双缓冲 | Q stage 降到 1，仅保留一个 O tile |
| Ping-pong 对象 | 不同 Q tile | 同一 Q 下的不同 K tile |
| Backward 组织 | 单 kernel，来源按约 5 个 MMA 计 | `dQ + dKdV` 双 kernel，合计约 7 个 MMA |
| dQ tile / loop | 原社区流水 | `128×128`，Outer-Q/Inner-K |
| dKdV tile / loop | 原社区流水 | `128×64`，Outer-K/Inner-Q |
| 主要约束 | 流水重叠与片上资源 | TMEM/SMEM 峰值、流水重叠与 2-CTA layout |

## Blackwell 数据流

- Load：`Q/K/V` 从 GMEM 进入 SMEM。
- Score：Tensor Core 计算 `S = QK^T`，结果进入 TMEM。
- Softmax：数据按 `TMEM → RF → TMEM` 路径做逐元素与归约处理。
- Output：Tensor Core 计算 `O = PV`，累加器驻留 TMEM。
- Correction / Epilogue：历史输出重缩放后经 RF/SMEM，最终写回 GMEM。
- 文章把 `tcgen05.mma`、TMEM、TMA、warp specialization、async pipeline 和 CTA Pair 视为一套联合编排机制，而非彼此独立的单点优化。

## Benchmark：仅记录来源口径

- 开头称 L20A 上 Forward 最高约 `1700 TFLOPS`、Backward 最高约 `980 TFLOPS`，长序列整体相对 FA3 超过 `2×`。
- Benchmark 小节称 L20C 上 Forward 在 GQA、non-causal、`seqlen=16k` 时约 `1839 TFLOPS`；相对 FA3 在 `seqlen≥8k` 时约 `2.0–2.3×`，`seqlen=4k` 时约 `1.15×`，并称 FA3 在 `seqlen=128k` OOM 而 FA4 约 `1300 TFLOPS`。
- Backward 小节称 `seqlen=4k` 时约 `1.4×`、`seqlen=64k` 时约 `2.6×`，长序列峰值约 `950 TFLOPS`。
- 总结又使用 Forward `1600 TFLOPS`、Backward `950 TFLOPS`。这些数字和硬件名称彼此不完全一致，不能合并成单一 benchmark 结论。

## 关键概念

- [[../concepts/Tensor Memory]]
- [[../concepts/FlashAttention]]
- [[../concepts/CUDA内存层次]]
- [[../concepts/GPU执行模型]]
- [[../concepts/Tiling]]
- [[../concepts/重计算]]

## 相关实体

- [[../entities/阿里云 PAI 团队]]
- [[../entities/NVIDIA Blackwell]]
- [[../entities/阿里巴巴]]

## 与现有 wiki 的关系

- 新建 [[../concepts/Tensor Memory]]，把 TMEM 与 register、SMEM、Tensor Core accumulator 的关系单独整理。
- 更新 [[../concepts/FlashAttention]]，补充 FA4 `head_dim=256` 的 Forward/Backward 专用 pipeline。
- 更新 [[../concepts/CUDA内存层次]]、[[../concepts/GPU执行模型]] 和 [[../concepts/Tiling]]，补充 TMEM、`tcgen05.mma`、CTA Pair、warp specialization 与 shape-dependent stage 设计。
- 更新 [[../concepts/重计算]]，补充 Backward 双 kernel 用 7 次 MMA 换片上存储空间的算子级案例。
- 更新 [[../entities/NVIDIA Blackwell]] 与 [[../entities/阿里巴巴]]，新建 [[../entities/阿里云 PAI 团队]]。
- 未发现与现有 wiki 的直接冲突；该来源把此前 CLC 资料中的 Blackwell cluster 视角扩展到 2-CTA MMA、DSMEM 与 TMEM 数据流。

## 待确认

- 来源依次使用 L20A、L20C 两个硬件名称，并给出 `1700/980`、`1839/950`、`1600/950 TFLOPS` 等不同峰值；需核对 SKU、时钟、dtype、causal/GQA、序列长度、测量公式和具体 commit。
- 支持表称 FA3 深度绑定 Hopper、无法在 SM100 运行，但 Benchmark 又直接给出 L20C 上 `FA4 vs FA3`；需确认 FA3 baseline 是否运行于另一平台、是否经过移植，或硬件名称是否存在笔误。
- 原文 dKdV 流程把 `dV`、`dK` 两项的结果变量误写为 `dQ`；本页只按 dKdV kernel 的目标与标准梯度依赖归纳，不修改 raw 原文。
- `256KB TMEM`、分配粒度、访问方向和 `tcgen05.mma` 语义应进一步对照对应 PTX/CUTLASS 版本；本文属于实现团队解释，不替代正式 ISA 文档。
- “已合入官方仓库”和“有效支撑千卡规模训练”需绑定 PR merge commit、训练配置和生产版本，不能由文章单独外推。
