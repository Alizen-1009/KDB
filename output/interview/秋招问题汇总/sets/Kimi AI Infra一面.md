# Kimi AI Infra 一面

用户提供的一面回忆，覆盖 MoE 通信与 grouped GEMM、长序列和 decode attention 并行、GPU 异步流水与量化。以下按用户原顺序排列；口述短答和依据在各问题页。

## 题目

1. [[../questions/DeepEP normal 与 low-latency 模式有什么区别|DeepEP normal / low-latency mode 区别？]]
2. [[../questions/Ulysses Sequence Parallelism 与 Ring Attention 有什么区别|Ulysses SP 和 Ring Attention 区别？]]
3. [[../questions/Decode Context Parallel 如何切分 KV 并合并 attention|DCP 怎么做？]]
4. [[../questions/Online Softmax 为什么可以分块计算，再合并成完整 softmax|Online Softmax / LSE merge 怎么做？]]
5. [[../questions/Ampere Hopper Blackwell 的异步计算流水如何演进|Ampere → Hopper → Blackwell 有什么变化？]]
6. [[../questions/Tensor Memory 流水如何组织，num_stages 怎么选|Tensor Memory 怎么做 pipeline？num_stages 怎么选？]]
7. [[../questions/寄存器与共享内存如何影响 occupancy，Persistent Kernel 如何隐藏延迟|register / shared memory 怎么影响 occupancy？Persistent Kernel 怎么隐藏 latency？]]
8. [[../questions/Grouped GEMM 的 contiguous 与 masked 布局及 tile 调度有什么区别|Grouped GEMM：contiguous / masked 区别？tile 怎么调度？]]
9. [[../questions/MXFP8 与 block-wise FP8 有什么区别|MXFP8 和 block-wise FP8 区别？]]
10. [[../questions/TMA 与 cp.async 如何选择|TMA 和 cp.async 怎么选？]]

## 返回

- [[../README|秋招问题汇总]]
