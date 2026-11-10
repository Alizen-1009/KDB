# CUDA 循环展开如何影响 ILP 与寄存器压力？

- 整理状态：已整理
- 题目出处：用户本次对话提供

## 30 秒回答

循环展开将多个迭代暴露给编译器，可能减少循环控制开销，并让独立加载和计算有更多调度空间。`#pragma unroll` 可指示编译器展开；但展开本身不能消除循环携带依赖，还可能增加寄存器与代码大小，因此需检查生成代码并实测。

## 深入解释

- 单个累加器的连续 `sum += x[i]` 存在依赖链；采用多个独立累加器可提供更多指令级并行度，最后归约。但调整浮点归约顺序可能改变数值。
- 寄存器压力上升可能降低 occupancy 或触发 spilling；代码过大也可能造成指令缓存压力。
- 比较不同展开因子时，结合耗时、寄存器、spill、指令及 stall 指标，而不是只看 occupancy。

## 关联知识

- [[CUDA occupancy 如何计算，block 和 grid 怎么调|CUDA occupancy]]
- [[GPU 算子有哪些优化方法，如何按瓶颈选择|按瓶颈选择优化]]

## 参考来源与待核实

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)：编译与循环展开行为按编译器和目标架构核实；本次未运行 GPU 测试。

## 所属题单

- [[../sets/C++与推理系统及 CUDA 面试回顾|本轮面试回顾]]
- [[../README|秋招问题汇总]]
