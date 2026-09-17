# PyTorch CUDA caching allocator 如何管理显存与缓解碎片？

- 整理状态：已整理
- 题目出处：用户提供的一面面经（PyTorch 图编译与推理系统专题）

## 其他问法

- Torch 中的显存管理机制是什么？
- 为什么删除 Tensor 后 `nvidia-smi` 显存不下降？

## 30 秒回答

PyTorch 不会为每个 Tensor 都直接执行一次昂贵且可能同步的 `cudaMalloc/cudaFree`，而是通过 CUDA caching allocator 向驱动申请较大的 segment，再切成 block 分配给 Tensor。Tensor 释放后，block 通常回到 PyTorch 空闲池供后续复用，所以 `allocated` 会下降而 `reserved` 未必下降；`empty_cache()` 只能把完全空闲的缓存段归还驱动，不能释放仍被 Tensor 引用的显存，也不能增加当前进程可用于活跃 Tensor 的理论容量。

## 深入解释

- `allocated`：活跃 Tensor 当前占用的 block。
- `reserved`：allocator 已从 CUDA 驱动取得、包含活跃块和缓存空闲块的总量。
- block 可拆分和在满足条件时合并；即使总空闲量够，大块申请也可能因碎片失败。
- 异步 stream 下还要保证 block 不会在原操作完成前被另一 stream 错误复用；自定义 stream 使用 Tensor 时要正确记录 stream 依赖。
- OOM 应先区分真实容量不足、生命周期过长、峰值 workspace、图保留和分配器碎片，再决定减 batch、checkpoint/offload、缩短上下文或调整 allocator 配置。

## 关联知识

- [[nvidia-smi 显存占用与框架 allocated、reserved 有何区别|显存观测口径]]
- [[../../../../wiki/concepts/CUDA内存层次|CUDA 内存层次]]

## 参考来源与待核实

- [PyTorch CUDA memory management](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
- allocator backend、`expandable_segments`、分裂策略和环境变量会随 PyTorch/CUDA 版本变化；面试时先讲稳定机制，再绑定版本谈参数。

## 所属题单

- [[../sets/PyTorch图编译与推理系统一面|PyTorch 图编译与推理系统一面]]
- [[../sets/量化与性能|量化与性能]]
- [[../README|秋招问题汇总]]
