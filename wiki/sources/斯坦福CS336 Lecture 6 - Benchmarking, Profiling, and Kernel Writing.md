# 斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing

## 来源信息

- 标题：斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing
- 作者：[[../entities/Stanford CS336]]
- 日期：2025 Spring
- 类型：可执行课程讲稿 / 代码
- 原始文件：[[../raw/articles/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing|斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]

## 2-3 条核心摘要

- 这讲把 GPU 性能知识从“理解硬件”推进到“真的去测、去看、去写 kernel”：benchmarking 决定你看见什么，profiling 决定你理解什么。
- 课程用 `GeLU / softmax / matmul / MLP` 这几个例子，把 `PyTorch -> torch.compile -> Triton -> CUDA` 放到同一坐标系里比较，而不是把它们当成割裂的工具。
- 这讲的核心工程原则非常统一：优化不是先写更复杂的代码，而是先用 benchmark 和 profiler 定位瓶颈，再围绕减少 reads/writes、提高复用和融合来组织计算。

## 值得关注的论断

- GPU 上很多优化判断如果不先 benchmark/profile，几乎一定会被“感觉”误导。
- 手写 kernel 的价值不仅在于极致性能，也在于它能暴露编程模型和真实硬件之间的缝隙。
- `torch.compile` 和 Triton 代表的方向是：越来越多的融合与代码生成会自动化，但理解底层访问模式仍然是判断性能的前提。

## 关键概念

- [[Benchmarking]]
- [[Profiling]]
- [[CUDA Kernel]]
- [[Triton]]
- [[Torch Compile]]
- [[算子融合]]
- [[Tiling]]

## 相关实体

- [[../entities/Stanford CS336]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`Benchmarking`、`Profiling`、`CUDA Kernel`、`Triton`、`Torch Compile`、`算子融合`、`Tiling`
- 会更新哪些实体页：`Stanford CS336`
- 是否存在冲突：与现有 wiki 无直接冲突，但需要注意这讲是“kernel 与性能工程实践”，不是单独的 Triton 入门或编译器课程

## 待确认

- 后续可补 Stanford CS336 `Lecture 7 - Parallelism basics`，把“单卡性能诊断与 kernel”自然接到“多卡并行训练”
