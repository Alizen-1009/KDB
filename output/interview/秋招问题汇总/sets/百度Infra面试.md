# 百度 Infra 面试

用户提供的百度 Infra 面试复盘，围绕异步 H2D、访存瓶颈、GEMV 和 shared memory 设计。按用户原顺序列题；逐题短答和依据在问题页。

## 题目

1. [[../questions/为什么异步 H2D 通常需要 pinned memory|为什么异步 H2D 一般需要 pinned memory？pageable memory 有什么问题？]]
2. [[../questions/Memory-bound kernel 如何判断优化空间与端到端收益|Memory-bound kernel 怎么判断还有优化空间、值不值得继续优化？]]
3. [[../questions/为什么 Hopper Blackwell 上普通 LDG 可能难以打满带宽|为什么 Hopper / Blackwell 上只靠普通 LDG 可能很难打满带宽？]]
4. [[../questions/如何实现高性能行主序 GEMV|`A[M,N] × B[N,1]` 怎么写高性能 CUDA GEMV？]]
5. [[../questions/什么数据适合放进 CUDA shared memory|什么数据适合放 shared memory？]]

## 返回

- [[../README|秋招问题汇总]]
