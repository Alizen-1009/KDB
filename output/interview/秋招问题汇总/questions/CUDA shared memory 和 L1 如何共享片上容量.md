# CUDA shared memory 和 L1 如何共享片上容量？

- 整理状态：已整理
- 题目出处：[小红书分享图转录](https://chatgpt.com/library?file_id=libfile_08e498b5bde48191b68600b6e5b9507d&account_id=personal)，第 2 题

## 原题与条件

2. 在 GPU 架构中，shared memory 和 L1 cache 是如何共享物理资源的？如何配置以最大化性能？

## 30 秒回答

在 Volta、A100、H100 等架构上，SM 的统一片上存储可以在 shared memory 和 L1 等用途间划分。shared 由 kernel 显式索引，L1 由硬件管理。

## 深入解释

用 cudaFuncAttributePreferredSharedMemoryCarveout 表示偏好。大动态 shared 另用 cudaFuncAttributeMaxDynamicSharedMemorySize 并传实际 launch 字节数。GEMM 要同时比较 staging 复用、L1 命中、寄存器与 CTA 驻留。

## 关联知识

- [[../../../../wiki/concepts/CUDA内存层次|CUDA内存层次]]

## 参考来源与待核实

- [参考 1](https://docs.nvidia.com/cuda/hopper-tuning-guide/)

## 所属题单

- [[../sets/英伟达高性能计算工程师26题|英伟达高性能计算工程师26题]]
- [[../README|秋招问题汇总]]
