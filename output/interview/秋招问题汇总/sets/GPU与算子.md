# GPU与算子

围绕 GPU 执行与访存、算子融合、Attention/Reduce 等手写题复习。先说清数据形状与正确性，再解释并行方式和资源权衡。

这是复习分类，不代表有统计证据的考频排名。状态以各问题页为准。

## 题目

1. [[../questions/为什么异步 H2D 通常需要 pinned memory|为什么异步 H2D 通常需要 pinned memory？]]
1. [[../questions/Memory-bound kernel 如何判断优化空间与端到端收益|Memory-bound kernel 如何判断优化空间与端到端收益？]]
1. [[../questions/为什么 Hopper Blackwell 上普通 LDG 可能难以打满带宽|为什么 Hopper、Blackwell 上普通 LDG 可能难以打满带宽？]]
1. [[../questions/如何实现高性能行主序 GEMV|如何实现高性能行主序 GEMV？]]
1. [[../questions/什么数据适合放进 CUDA shared memory|什么数据适合放进 CUDA shared memory？]]
1. [[../questions/Hopper 为什么设计 WGMMA，传统 MMA 有什么局限|Hopper 为什么设计 WGMMA，传统 MMA 有什么局限？]]
1. [[../questions/PyTorch 算子如何注册 CPU 与 CUDA 实现并由 Dispatcher 分发|PyTorch 算子如何注册 CPU 与 CUDA 实现并由 Dispatcher 分发？]]
1. [[../questions/如何完整实现带 Reduce 的 CUDA RMSNorm|如何完整实现带 Reduce 的 CUDA RMSNorm？]]
1. [[../questions/如何用 Triton 实现并优化 Softmax|如何用 Triton 实现并优化 Softmax？]]
1. [[../questions/torch.compile 的 Dynamo、AOTAutograd 与 Inductor 如何协作|torch.compile 的 Dynamo、AOTAutograd 与 Inductor 如何协作？]]
1. [[../questions/GEMM 常用的 GPU 优化手段有哪些|GEMM 常用的 GPU 优化手段有哪些？]]
1. [[../questions/FlashAttention-1、2、3 分别优化了什么|FlashAttention-1、2、3 分别优化了什么？]]
1. [[../questions/FlashDecoding 与 FlashDecoding++ 如何优化小 Batch 长上下文|FlashDecoding 与 FlashDecoding++ 如何优化小 Batch 长上下文？]]
1. [[../questions/CUDA 语境中的全双工、半双工与 cache line 分别是什么|CUDA 语境中的全双工、半双工与 cache line 分别是什么？]]
1. [[../questions/CUDA Thrust、CUB 与 CUTLASS 各自解决什么问题|CUDA Thrust、CUB 与 CUTLASS 各自解决什么问题？]]
1. [[../questions/FlashAttention-2 与 FlashDecoding 为什么更快，分别优化什么|FlashAttention-2 与 FlashDecoding 为什么更快，分别优化什么？]]
1. [[../questions/如何理解 CUDA 编程与显存管理，避免 OOM 并优化 kernel|如何理解 CUDA 编程与显存管理，避免 OOM 并优化 kernel？]]
1. [[../questions/XLA 与 TVM 等 AI 编译栈如何优化计算图，有何侧重|XLA 与 TVM 等 AI 编译栈如何优化计算图，有何侧重？]]
1. [[../questions/FlashAttention 的内存优化核心思想是什么|FlashAttention 的内存优化核心思想是什么？]]
1. [[../questions/Online Softmax 为什么可以分块计算，再合并成完整 softmax|Online Softmax 为什么可以分块计算，再合并成完整 softmax？]]
1. [[../questions/融合算子如何设计，收益和边界是什么|融合算子如何设计，收益和边界是什么？]]
1. [[../questions/CUDA Graph 的加速原理与 capture-replay 约束|CUDA Graph 的加速原理与 capture-replay 约束？]]
1. [[../questions/如何手写 CUDA Reduce，并用 grid-stride 和 warp shuffle 优化|如何手写 CUDA Reduce，并用 grid-stride 和 warp shuffle 优化？]]
1. [[../questions/CUDA Reduce 的 block size 为什么常选择 128 或 256|CUDA Reduce 的 block size 为什么常选择 128 或 256？]]
1. [[../questions/固定 shape 下如何制定算子调优策略|固定 shape 下如何制定算子调优策略？]]
1. [[../questions/Tensor Core 与 CUDA Core 活跃度如何区分|Tensor Core 与 CUDA Core 活跃度如何区分？]]
1. [[../questions/GPU 中 SM、block、warp 和 thread 如何协作执行 CUDA kernel|GPU 中 SM、block、warp 和 thread 如何协作执行 CUDA kernel？]]
1. [[../questions/SM Active、SM Issue、Occupancy 与管线活跃度有何区别|SM Active、SM Issue、Occupancy 与管线活跃度有何区别？]]
1. [[../questions/Marlin 如何让 W4A16 权重量化真正获得推理加速|Marlin 如何让 W4A16 权重量化真正获得推理加速？]]
1. [[../questions/A100 与 H20 的硬件差异如何影响任务放置|A100 与 H20 的硬件差异如何影响任务放置？]]

## 返回

- [[../README|秋招问题汇总]]
