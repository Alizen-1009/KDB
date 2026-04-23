# 斯坦福CS336 Lecture 5 - GPUs

## 来源信息

- 官方课程：Stanford CS336: Language Modeling from Scratch
- 官方讲义仓库：https://github.com/stanford-cs336/spring2025-lectures
- 官方讲义 PDF：https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf
- 镜像视频：[Bilibili - 5.斯坦福CS336：GPU原理与分布式训练基础](https://www.bilibili.com/video/BV1T21PBqErg?p=5)
- 讲师：Tatsu Hashimoto
- 课程时间：Spring 2025
- 原始类型：课程讲义 / 视频镜像

## 原始说明

- 本文件以 Stanford 官方讲义 `Lecture 5 - GPUs` 为主来源。
- B 站分 P 标题写为“GPU原理与分布式训练基础”，但官方讲义主体内容集中在 GPU / CUDA 基础、性能分析与 FlashAttention，不是系统性的分布式训练讲。
- 由于 B 站该分 P 无公开字幕，本次 ingest 以官方 PDF 内容为主要文本锚点，视频作为镜像入口。

## 讲义结构

### Part 1: GPUs in depth

- GPU 与 CPU 的差别：CPU 更偏向低延迟、少量复杂线程；GPU 更偏向高吞吐、大量轻量线程。
- GPU 的执行单元层级包括 `SM`、`SP`、`thread`、`block`、`warp`。
- GPU 的内存层级包括寄存器、shared memory / L1、L2、global memory，越靠近 SM 越快、越贵。
- Warp 是 32 个连续线程共同执行的基本调度单位。

### Part 2: Understanding GPU performance

- GPU 性能分析的关键问题是：如何避免 memory bound。
- 讲义强调的优化思路包括：
  - 低精度计算
  - 算子融合
  - 重计算
  - 内存合并访问（memory coalescing）
  - Tiling
- 一个核心背景是：`compute scaling` 快于 `memory scaling`，因此现代 GPU 越来越容易被数据搬运而非算力本身限制。
- 讲义用 roofline 视角解释为什么同一个算子在不同 arithmetic intensity 下会表现出截然不同的性能。

### Part 3: FlashAttention

- FlashAttention 并不是“凭空更快的 attention”，而是把前面关于 tiling、fusion、online softmax 和重计算的硬件直觉组合起来。
- 注意力计算可被拆为 `QK^T`、softmax、与 `V` 相乘三部分，中间的 softmax 需要特殊处理才能 tile-wise 计算。
- 讲义强调 online softmax / incremental softmax 的作用：让 softmax 可以按 tile 增量计算，而不必完整物化整张 attention score matrix。
- backward 过程的高效实现依赖重新计算而不是完整保存中间激活。

## 从讲义中抽出的高信号结论

- GPU 优化的核心不是单纯“多做并行”，而是减少高成本 global memory 访问、提高 arithmetic intensity，并尊重硬件层级。
- Tiling 是这讲的主轴，因为它同时连接了共享内存复用、coalescing 和矩阵乘法性能。
- FlashAttention 最值得学的不是某个 kernel 技巧本身，而是“从内存访问模式反推算法结构”的方法。

## 与后续课程的边界

- 本讲更多是 GPU / CUDA / kernel 性能基础。
- 多机多卡、collective communication、张量并行、流水并行等系统并行训练主题，更接近 Stanford CS336 后续的 `Lecture 7 - Parallelism basics`。
