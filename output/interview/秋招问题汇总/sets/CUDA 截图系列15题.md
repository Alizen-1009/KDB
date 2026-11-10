# CUDA 截图系列15题

题目来自用户提供的本次对话截图系列摘要；顺序按原题，答案维护在各问题页。涉及具体硬件、实现版本和性能的结论须在目标环境核验。

1. [[../questions/batch1 decode 融合算子访存不合并如何优化|batch1 decode 融合与非合并访存]]
2. [[../questions/CUDA bank conflict 如何产生，padding 为什么有效|shared 转置与 bank conflict]]
3. [[../questions/CUDA occupancy 如何计算，block 和 grid 怎么调|spill 与 occupancy 调优]]
4. [[../questions/CUDA L2 persistence 如何设置及有哪些限制|L2 persistence]]
5. [[../questions/GPU SpMM 如何应对稀疏行长不均|SpMM 格式与负载均衡]]
6. [[../questions/如何手写 CUDA Reduce，并用 grid-stride 和 warp shuffle 优化|超长非二次幂 reduction]]
7. [[../questions/CUDA warp divergence 如何影响吞吐，怎样用重排优化|divergence 与分桶]]
8. [[../questions/WMMA 如何使用 Tensor Core 做混合精度矩阵乘|WMMA 的 123 维边界]]
9. [[../questions/TMA 与 cp.async 如何选择|Hopper TMA 与 mbarrier]]
10. [[../questions/CUDA 直方图如何用 shared memory 降低原子热点|shared histogram]]
11. [[../questions/CUDA 多流如何用 event 重叠拷贝与计算|多 stream 流水与队列饱和]]
12. [[../questions/CUDA Graph 的加速原理与 capture-replay 约束|CUDA Graph capture 和 replay]]
13. [[../questions/NVSwitch 和 InfiniBand 上如何设计分层集合通信|多机 NCCL 和拓扑]]
14. [[../questions/NVLink 多 GPU 如何做 P2P 并安排通信|UVA 与 P2P]]
15. [[../questions/GPUDirect RDMA 如何注册 GPU buffer 并保证可见性|GPUDirect RDMA 注册与可见性]]

返回：[[../README|秋招问题汇总]]
