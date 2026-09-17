# CUDA Thrust、CUB 与 CUTLASS 各自解决什么问题？

- 整理状态：已整理
- 题目出处：用户提供的一面面经（SGLang / CUDA 系统专题）

## 其他问法

- 介绍一下 CUDA Thrust。
- CUTLASS 是什么，和 Thrust、CUB 有什么区别？

## 30 秒回答

Thrust 是类似 C++ STL 的高层并行算法库，提供 `sort/reduce/scan/transform` 和 `device_vector`；CUB 更低层且 CUDA-specific，既有 device-wide 算法，也有 warp/block collectives，适合构建自定义 kernel；CUTLASS 面向高性能 GEMM、卷积和相关 fused kernel，核心是 tile、layout、MMA atom、pipeline 与 epilogue。简单说：通用并行算法优先 Thrust，需要控制 block/warp 原语用 CUB，Tensor Core 矩阵核与融合模板用 CUTLASS。

## 深入解释

```text
Thrust   高层算法：thrust::sort / reduce / scan / transform
  ↓ CUDA backend 常复用更底层实现
CUB      Device / Block / Warp 级 collectives

CUTLASS  另一条专注矩阵计算的栈：GEMM / Conv / Attention-like kernel
  └─ CuTe 用 layout algebra 描述 tensor、tile 与硬件 atom
```

Thrust 适合快速写数据预处理和通用并行算法，但复杂算子链可能产生多个 kernel launch 和中间结果；CUB 允许把 `BlockReduce`、`WarpScan` 等嵌入自己的 kernel；CUTLASS 不是 STL 算法库，它更接近可组合的高性能矩阵 kernel 模板库。

## 面试官可能追问

- Thrust 调用是否一定零开销？如何指定 stream？
- `thrust::reduce` 与 `cub::DeviceReduce`、`cub::BlockReduce` 的抽象层次有何不同？
- CUTLASS、CuTe DSL、Triton 和手写 CUDA 如何选？

## 关联知识

- [[../../../../wiki/concepts/CuTe DSL|CuTe DSL]]
- [[../../../../wiki/concepts/CUDA Kernel|CUDA Kernel]]
- [[../../../../wiki/concepts/Warp Shuffle Reduce|Warp Shuffle Reduce]]

## 参考来源与待核实

- [NVIDIA CCCL：Thrust、CUB、libcu++](https://github.com/NVIDIA/cccl/tree/a39a9ce9a31d57b93609e629049ede7ab7bb66c0)
- [NVIDIA CUB 文档：CUB 与 Thrust](https://nvidia.github.io/cccl/cub/)
- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
- API 和 backend 细节绑定 CUDA Toolkit / CCCL / CUTLASS 版本；不能承诺 Thrust 对任意 shape 都与手写特化 kernel 一样快。

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/SGLang与CUDA系统一面|SGLang 与 CUDA 系统一面]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
