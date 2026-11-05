# CUDA 算子的常量参数放在哪种内存中？

- 整理状态：已整理
- 题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，AI Infra 硬件与算子第 8 题；2026-11-05 收录，实际面试日期与具体轮次未提供。

## 原题与条件

算子中的常量参数是放在哪个 memory 的？

## 30 秒回答

取决于它是编译期值、kernel 启动参数、显式 __constant__ 变量，还是指向只读数据的指针。编译期值可能成为立即数或被折叠，启动参数走参数机制，__constant__ 使用常量地址空间；const float* 所指数据仍可能在 global memory，const 本身不决定存储位置。

## 深入解释

| 情况 | 解释 |
|---|---|
| 编译期常量 | 可能立即数、常量折叠，甚至不产生实际加载 |
| kernel 标量参数 | 通过 kernel 参数传入；具体加载/缓存由架构与编译器决定 |
| 显式 __constant__ 变量 | CUDA constant memory |
| const float* p | 约束通过 p 的修改权限，不能据此认定数据在 constant memory |
| 运行时只读权重 | 常常仍在 global memory，并经相应缓存路径读取 |

Constant memory 对 warp 内同地址读取适合广播；不同 lanes 访问许多不同地址时，可能串行化。判断具体程序应查声明、分配和编译产物，而不是看到“常量”两个字就断言位置。

## 关联知识

- [[../../../../wiki/concepts/CUDA内存层次|CUDA内存层次]]

## 参考来源与待核实

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)。
- 本页不承诺特定 CUDA 版本的参数物理缓存路径；精确代码生成需检查目标编译结果。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
