# Matmul 的 M、N、K 都较小时如何提高效率？

- 整理状态：已整理
- 题目出处：用户本次对话提供

## 30 秒回答

三维都小时，单次工作量可能不足以填满 GPU，launch 和边界浪费也更显著。优先把独立任务做 batched matmul，或在时延允许时合并请求；再考虑小矩阵专用 kernel、合理 tile、融合及 CUDA Graph，最后按真实 shape 和硬件测量。

## 深入解释

- 输出块数量不足时，可尝试较小 tile，但要权衡重复读取、调度开销和 Tensor Core 指令粒度。
- bias、激活等 epilogue 融合可减少启动和中间访存；CUDA Graph 主要减少 CPU 端重复发射开销，并不减少数学工作量。
- 适当 padding 可能改善硬件利用，但无效运算增多；batching 可能增加排队延迟。
- Split-K 适合 M、N 小而 K 足够大、输出并行不足时；K 也很小时，额外归约与同步可能得不偿失。

## 关联知识

- [[矩阵乘法的计算量、读取量与 tile 大小是什么关系|矩阵乘 tile 与复用]]
- [[GEMM 常用的 GPU 优化手段有哪些|GEMM 优化]]

## 参考来源与待核实

- 本次对话归纳；具体 shape、dtype、GPU 下的最优 tile 与收益需 benchmark，本次未运行 GPU 测试。

## 所属题单

- [[../sets/C++与推理系统及 CUDA 面试回顾|本轮面试回顾]]
- [[../README|秋招问题汇总]]
