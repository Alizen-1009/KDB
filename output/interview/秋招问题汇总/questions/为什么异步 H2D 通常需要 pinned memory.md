# 为什么异步 H2D 通常需要 pinned memory？

- 整理状态：已整理
- 题目出处：用户提供的百度 Infra 面试复盘

## 30 秒回答

GPU 的拷贝引擎需要在传输期间稳定访问主机缓冲区。Pinned memory 的物理页被锁定，适合 DMA，`cudaMemcpyAsync` 才能可靠地实现主机端异步返回及与其他 stream 工作重叠。Pageable memory 可能被换出，驱动通常要先复制到 pinned staging buffer，再传给 GPU，增加一次内存复制，也可能让异步 API 退化为阻塞行为。Pinned memory 分配和占用有成本，适合复用的传输缓冲区，不应无限制锁页。

## 深入解释

- pinned 是异步重叠的必要前提之一；copy engine、stream 依赖和硬件并发能力也要满足，不能仅凭 pinned 就保证 H2D 与 kernel 同时执行。
- 可用 `cudaMallocHost` / `cudaHostAlloc` 分配，或用 `cudaHostRegister` 锁定已有缓冲区。传输完成前不能重用或修改源缓冲区。

## 关联知识

- [[../../../../wiki/concepts/CUDA内存层次|CUDA 内存层次]]
- [[CUDA 语境中的全双工、半双工与 cache line 分别是什么|异步拷贝与 copy engine]]

## 参考来源与待核实

- [CUDA Programming Guide：Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)；[CUDA Best Practices：Pinned Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#pinned-memory)。

## 所属题单

- [[../sets/百度Infra面试|百度 Infra 面试]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
