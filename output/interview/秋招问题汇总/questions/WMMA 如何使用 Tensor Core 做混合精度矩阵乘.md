# WMMA 如何使用 Tensor Core 做混合精度矩阵乘？

- 整理状态：已整理
- 题目出处：[小红书分享图转录](https://chatgpt.com/library?file_id=libfile_08e498b5bde48191b68600b6e5b9507d&account_id=personal)，第 5、12 题

## 原题与条件

5. 在 Volta/Ampere 架构中，Tensor Core 的矩阵乘累加操作（如 WMMA）的编程模型是什么？如何与常规 CUDA core 协同？
12. 在深度学习推理场景中，如何利用 Tensor Core 进行混合精度计算？请说明 WMMA API 的使用方法及性能调优要点。

## 30 秒回答

WMMA 由整个 warp 共同持有 fragment，沿 K 循环执行 load_matrix_sync、mma_sync，最后 store_matrix_sync。FP16 输入、FP32 累加是典型路径。

## 深入解释

所有参与 lane 必须一致执行 warp collective，fragment 的 lane 到元素映射由实现决定。遵守目标 API 的 shape、layout、ldm 和对齐要求。线程做地址、搬运和 epilogue，Tensor Core 做 MMA。预填充看 MMA 利用率，decode 同时看小 M tile 利用率和权重流量。W4A16 常见做法是 kernel 内解包反量化后再交给受支持的 MMA。

## 关联知识

- [[../../../../wiki/concepts/混合精度训练与推理|混合精度训练与推理]]

## 参考来源与待核实

- [参考 1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions)

## 所属题单

- [[../sets/英伟达高性能计算工程师26题|英伟达高性能计算工程师26题]]
- [[../README|秋招问题汇总]]
