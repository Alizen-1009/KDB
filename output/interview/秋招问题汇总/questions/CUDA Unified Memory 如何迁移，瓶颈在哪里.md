# CUDA Unified Memory 如何迁移，瓶颈在哪里？

- 整理状态：已整理
- 题目出处：[小红书分享图转录](https://chatgpt.com/library?file_id=libfile_08e498b5bde48191b68600b6e5b9507d&account_id=personal)，第 3 题

## 原题与条件

3. 描述 CUDA 统一内存（Unified Memory）的工作原理，并分析其在大规模数据迁移场景下的性能瓶颈。

## 30 秒回答

cudaMallocManaged 返回 CPU 与 GPU 可用的同一个指针。在支持按需迁移的平台上，访问可能触发 fault，运行时更新映射、迁移页或远程访问。

## 深入解释

看缺页停顿、互连带宽、超额分配后的驱逐，以及 CPU/GPU 交替触碰造成的页往返。用 Nsight Systems 查迁移时间线，按访问阶段 prefetch，按 GPU 划分工作集；固定流量可对照显式异步搬运。

## 关联知识

- [[../../../../wiki/concepts/CUDA内存层次|CUDA内存层次]]

## 参考来源与待核实

- [参考 1](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)

## 所属题单

- [[../sets/英伟达高性能计算工程师26题|英伟达高性能计算工程师26题]]
- [[../README|秋招问题汇总]]
