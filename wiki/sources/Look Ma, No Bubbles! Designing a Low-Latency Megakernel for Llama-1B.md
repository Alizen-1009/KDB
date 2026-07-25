---
type: source
source_kind: 论文
topic: GPU 编程
updated: 2026-06-12
---

# Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B

## 来源信息

- 标题：Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B
- 作者：HazyResearch / Stanford 相关作者；原始资料未在 frontmatter 中记录具体作者
- 日期：2025-05-27；原始资料创建于 2026-06-12
- 类型：文章 / 系统优化案例 / kernel 设计说明
- 原始文件：[[../../raw/articles/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B 1|Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B 1]]
- 原始链接：https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
- 外部线索：文章指向开源代码仓库 `HazyResearch/Megakernels`

## 2-3 条核心摘要

- 文章把 Llama-3.2-1B、batch size 1、单序列 decode 定义为强 memory-bound 的低延迟场景：每次 forward 主要受限于从 GPU global memory 读取模型权重的速度，而不是算力峰值。
- 作者认为 vLLM / SGLang 在该场景下难以吃满 H100 带宽的根因不是单个算子慢，而是整次 forward 被拆成约百个小 [[CUDA Kernel]] 后，kernel 严格顺序、[[Tail Effect]]、launch / teardown、权重与激活加载延迟共同形成 `memory pipeline bubbles`。
- 方案是把整个 Llama-1B forward pass 融进一个 [[Megakernel]]：使用 on-GPU interpreter 执行预先生成的 per-SM instruction schedule，并通过 shared memory paging、显式 counter 同步和 chunk 级依赖来让权重加载尽量连续。

## 值得关注的论断

- 文章声称 H100 megakernel 达到约 `78%` memory bandwidth，且相对 vLLM / SGLang baseline 有超过 `1.5x` 的性能优势；这些数字应视为来源声称，需要绑定具体 prompt 长度、生成长度、dtype、硬件、baseline 配置和 commit。
- 文中给出一个 roofline 式直觉：单 H100 上 16-bit Llama-1B forward 的纯带宽上限约为 `3.35 TB/s / 2.48 GB ~= 1350 forward/s`；若每层 7 个 kernel、16 层、每个 kernel 边界乐观估计 `5 us` 停顿，则只能到约 `770 forward/s`。
- [[Programmatic Dependent Launch]]、CUDA Graphs 和 stream 可以缓解部分 launch / 依赖问题，但文章认为它们仍然保留较粗粒度同步或边界停顿；megakernel 的价值在于把依赖管理搬到 kernel 内部，以更细粒度地 pipeline weight load 和 activation handoff。

## 关键概念

- [[Megakernel]]
- [[CUDA Kernel]]
- [[算子融合]]
- [[Tail Effect]]
- [[Roofline 模型]]
- [[CUDA内存层次]]
- [[GPU执行模型]]
- [[Programmatic Dependent Launch]]
- [[Profiling]]

## 相关实体

- [[../entities/HazyResearch]]
- [[../entities/Megakernels]]
- [[../entities/vLLM]]
- [[../entities/SGLang]]

## 与现有 wiki 的关系

- 更新概念页：[[Megakernel]]、[[Programmatic Dependent Launch]]、[[CUDA Kernel]]、[[算子融合]]、[[Tail Effect]]、[[Roofline 模型]]、[[CUDA内存层次]]、[[GPU执行模型]]、[[Profiling]]
- 更新实体页：[[../entities/HazyResearch]]、[[../entities/Megakernels]]、[[../entities/vLLM]]、[[../entities/SGLang]]
- 是否存在冲突：未发现直接冲突；该来源补充了“超大融合并非通用最佳实践，但在 batch size 1、固定 Llama-1B、低延迟 decode 的极窄场景中，减少 kernel 边界本身可能成为主收益”的特化案例。

## 待确认

- 文章中的 benchmark 需要回到 `HazyResearch/Megakernels` 代码、具体 GPU 型号、驱动/CUDA 版本、baseline 配置、prompt/generation 长度和测量脚本核实。
- Megakernel 是否能稳定迁移到更大模型、更高 batch、动态 batch、复杂 serving 调度、量化和多 GPU 场景，来源只给出方向性判断，尚不能写成通用结论。
- 文中提到 H100 上 CUDA stream dummy launch 约 `2.1 us`、CUDA Graph launch 约 `1.3 us`，以及 B200 runtime breakdown，均应保留为该来源的实验观察。
