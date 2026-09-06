# GPU与算子

围绕 GPU 执行与访存、算子融合、Attention/Reduce 等手写题复习。先说清数据形状与正确性，再解释并行方式和资源权衡。

这是复习分类，不代表有统计证据的考频排名。状态以各问题页为准。

## 题目

1. [[../questions/FlashAttention-2 与 FlashDecoding 为什么更快，分别优化什么|FlashAttention-2 与 FlashDecoding 为什么更快，分别优化什么？]]
2. [[../questions/如何理解 CUDA 编程与显存管理，避免 OOM 并优化 kernel|如何理解 CUDA 编程与显存管理，避免 OOM 并优化 kernel？]]
3. [[../questions/XLA 与 TVM 等 AI 编译栈如何优化计算图，有何侧重|XLA 与 TVM 等 AI 编译栈如何优化计算图，有何侧重？]]
4. [[../questions/FlashAttention 的内存优化核心思想是什么|FlashAttention 的内存优化核心思想是什么？]]
5. [[../questions/Online Softmax 为什么可以分块计算，再合并成完整 softmax|Online Softmax 为什么可以分块计算，再合并成完整 softmax？]]
6. [[../questions/融合算子如何设计，收益和边界是什么|融合算子如何设计，收益和边界是什么？]]
7. [[../questions/CUDA Graph 的加速原理与 capture-replay 约束|CUDA Graph 的加速原理与 capture-replay 约束？]]
8. [[../questions/如何手写 CUDA Reduce，并用 grid-stride 和 warp shuffle 优化|如何手写 CUDA Reduce，并用 grid-stride 和 warp shuffle 优化？]]
9. [[../questions/CUDA Reduce 的 block size 为什么常选择 128 或 256|CUDA Reduce 的 block size 为什么常选择 128 或 256？]]
10. [[../questions/固定 shape 下如何制定算子调优策略|固定 shape 下如何制定算子调优策略？]]
11. [[../questions/Tensor Core 与 CUDA Core 活跃度如何区分|Tensor Core 与 CUDA Core 活跃度如何区分？]]
12. [[../questions/GPU 中 SM、block、warp 和 thread 如何协作执行 CUDA kernel|GPU 中 SM、block、warp 和 thread 如何协作执行 CUDA kernel？]]
13. [[../questions/SM Active、SM Issue、Occupancy 与管线活跃度有何区别|SM Active、SM Issue、Occupancy 与管线活跃度有何区别？]]
14. [[../questions/Marlin 如何让 W4A16 权重量化真正获得推理加速|Marlin 如何让 W4A16 权重量化真正获得推理加速？]]
15. [[../questions/A100 与 H20 的硬件差异如何影响任务放置|A100 与 H20 的硬件差异如何影响任务放置？]]

## 返回

- [[../README|秋招问题汇总]]
