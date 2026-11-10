# C++ 与推理系统及 CUDA 面试回顾

本题单按本次对话顺序整理；答案以对应问题页为准。实现细节需结合框架版本和硬件核实，性能收益须在目标工作负载下实测。

1. C++ 静态与动态多态、虚函数机制及虚表顺序：[[../questions/C++ 运行时多态与虚函数表如何工作|C++ 多态与虚表]]。
2. PD 分离的调度队列与 KV 传输：[[../questions/PD 分离中调度队列与 KV 传输如何衔接|PD 队列与传输]]；[[../questions/PD 解耦与物理分离如何实现，KV 和请求状态怎样交接|状态交接]]。
3. 如何判断算子是否值得优化及识别瓶颈：[[../questions/GPU 算子有哪些优化方法，如何按瓶颈选择|算子优化路径]]；[[../questions/Memory-bound kernel 如何判断优化空间与端到端收益|端到端收益]]；[[../questions/如何用 Roofline 模型区分计算、访存和其他瓶颈|Roofline]]。
4. 尚未量化的模型如何推进量化：[[../questions/如何系统推进未量化模型的部署量化|量化流程]]。
5. AWQ 反量化后为什么仍可能加速：[[../questions/Marlin 如何让 W4A16 权重量化真正获得推理加速|量化 kernel 与反量化]]；[[../questions/GPTQ 与 AWQ 的数学原理和工程技巧有什么区别|AWQ 原理]]。
6. 动态图、静态图、融合与编译成本：[[../questions/动态图和静态图如何支持融合及摊销编译成本|图编译与融合]]。
7. vLLM 调度与 KV 管理：[[../questions/连续批处理与传统批处理有什么区别，吞吐提升取决于什么|连续批处理]]；[[../questions/PagedAttention 的原理是什么，解决了什么痛点|PagedAttention]]；[[../../../../wiki/concepts/vLLM V1 统一调度器|vLLM V1 调度器]]。
8. Linear 的 K、N tile 为什么影响性能及如何选择：[[../questions/矩阵乘法的计算量、读取量与 tile 大小是什么关系|tile 与数据复用]]。
9. M、N、K 较小时如何提升 Matmul 效率：[[../questions/小 M N K 的矩阵乘如何提高效率|小矩阵乘]]。
10. Tensor Core 与 CUDA Core 的区别及 MMA 调用：[[../questions/WMMA 如何使用 Tensor Core 做混合精度矩阵乘|WMMA 与 Tensor Core]]；[[../questions/Tensor Core 与 CUDA Core 活跃度如何区分|硬件管线区别]]。
11. PCIe 带宽瓶颈与 H2D 延迟隐藏：[[../questions/为什么异步 H2D 通常需要 pinned memory|异步 H2D]]；[[../questions/CUDA 多流如何用 event 重叠拷贝与计算|拷贝与计算重叠]]。
12. Occupancy 是什么，越高是否越好：[[../questions/CUDA occupancy 如何计算，block 和 grid 怎么调|Occupancy]]。
13. CUDA 原子操作的机制、用途与性能影响：[[../questions/CUDA 原子操作何时需要及如何降低竞争|原子操作]]。
14. CUDA Stream 与计算、传输重叠：[[../questions/CUDA 多流如何用 event 重叠拷贝与计算|多流与 event]]。
15. Unified Memory 与 Page Fault：[[../questions/CUDA Unified Memory 如何迁移，瓶颈在哪里|统一内存]]。
16. Loop Unrolling 与 ILP：[[../questions/CUDA 循环展开如何影响 ILP 与寄存器压力|循环展开]]。

## 返回

- [[../README|秋招问题汇总]]
