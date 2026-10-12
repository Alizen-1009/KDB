# 矩阵乘法的计算量、读取量与 tile 大小是什么关系？

- 整理状态：已整理
- 题目出处：用户提供的 2026-10-10 面试回忆；答案根据本次对话整理

## 问题背景

用户要求仅用 M、N、K 和 block size 表示计算量、数据量及 tile 的影响。约定 C=A B，A 为 M×K，B 为 K×N，C 为 M×N；用 b 表示 block size，即方形 tile 边长，不是 CUDA block 的线程数。CTA 计算 b×b 的输出，每轮沿 K 推进 b。先假设 M/N/K 均可被 b 整除，不读取旧 C。

## 30 秒回答

计算量约为 2MNK FLOPs，A/B/C 大小分别为 MK、KN、MN 个元素。输入各读一次的理论最少搬运是 MK+KN+MN；朴素无缓存复用实现的输入读取是 2MNK。用 b×b tile 后，每轮读 2b² 个输入并做 2b³ FLOPs，每个输入在块内复用 b 次。忽略跨 block 缓存，总输入读取降为 2MNK/b，加上输出为 2MNK/b+MN。输入读取减少 b 倍不等于加速 b 倍，还受算力、缓存、资源占用和并行度限制。

## 深入解释

### 计算量与矩阵大小

一个输出做 K 次乘法与 K−1 次加法，精确数学操作数为 MN(2K−1)；按 K 次 FMA、每次计 2 FLOPs 的惯例，记为：

$$F\approx 2MNK$$

| 对象 | 元素数 |
|---|---:|
| A | MK |
| B | KN |
| C | MN |

FLOPs 表示操作数；FLOP/s 表示速度，还需要除以执行时间。所有数据量公式以元素为单位：同一 dtype 下 FP32 转字节乘 4，FP16/BF16 乘 2。

### 朴素与 tile 读取推导

朴素无缓存复用：每个输出分别读 K 个 A、K 个 B，总输入读取为 2MNK。

Tile 模型：输出 CTA 数为 MN/b²，每 CTA 沿 K 推进 K/b 轮。一轮读 A/B 各 b² 个元素，做 2b³ FLOPs，因此：

$$R_{tile}=\frac{MN}{b^2}\frac{K}{b}(2b^2)=\frac{2MNK}{b}$$

等价解释：A 被 N/b 个输出列块重复读取，B 被 M/b 个输出行块重复读取：

$$R_A=MK\frac{N}{b},\qquad R_B=KN\frac{M}{b}$$

| 模型 | 输入读取 | 输出写入 | 计算量 |
|---|---:|---:|---:|
| 朴素、无缓存复用 | 2MNK | MN | 约 2MNK |
| Tile、无跨块缓存复用 | 2MNK/b | MN | 约 2MNK |
| 输入各读一次的理论最少搬运 | MK+KN | MN | 约 2MNK |

“输入各读一次”是 compulsory traffic 的理想下界，不保证有限缓存下可以达到。上述 tile 读取是 CTA 发起的输入加载账本，不等于实际 HBM 流量；L2 命中能服务跨 CTA 的重复读取。

### 算术强度与 tile 的代价

计入输出、按 FLOPs/元素：

$$AI=\frac{2MNK}{2MNK/b+MN}=\frac{2Kb}{2K+b}$$

K 远大于 b 时，AI 约为 b；FP32 下约 b/4 FLOPs/byte。若 K 不大，输出写回不能忽略。

- A/B 单缓冲 shared memory：2b² 个元素；双缓冲：4b²。
- 每 CTA 输出 accumulator 总量随 b² 增长；每线程寄存器取决于线程映射，不能由 b 单独确定。
- 输出 CTA 数为 MN/b²：增大 b 改善块内复用，同时减少 CTA 数，可能让部分 SM 空闲。
- 大 tile 或多级流水可能降低驻留 CTA 数，甚至造成寄存器 spill。
- BK 单独增大不自动增加理论输入复用；这里 b 同时增大输出 tile 的 M/N 边长与 K 步长。

### 不整除的边界

若输入加载做有效 mask、每个有效输入在相应 CTA 中每轮只加载一次，总输入读取为：

$$MK\left\lceil\frac{N}{b}\right\rceil+KN\left\lceil\frac{M}{b}\right\rceil$$

有效数学计算量仍约 2MNK。若完整执行补零后的 tile，指令工作量可达到：

$$2b^3\left\lceil\frac{M}{b}\right\rceil\left\lceil\frac{N}{b}\right\rceil\left\lceil\frac{K}{b}\right\rceil$$

是否实际执行全部无效 FMA 取决于边界处理；不能把补齐操作数作为有效 FLOPs。

## 常见误区

- block size 必须先说明是 tile 边长还是线程数；b×b 输出不要求 b² 个线程，一个线程可以计算多个输出。
- 矩阵存储量、累计加载量与 HBM 字节数不能混用。
- 输入加载减少 b 倍不保证性能提升 b 倍；切 tile 不改变有效数学工作量。

## 关联知识

- [[GEMM 常用的 GPU 优化手段有哪些|GEMM 优化手段]]
- [[../../../../wiki/concepts/Tiling|Tiling]]
- [[../../../../wiki/concepts/Roofline 模型|Roofline 模型]]

## 参考来源与待核实

- 本页公式为上述方形 tile、加载和缓存假设下的机制推导。
- [NVIDIA：CUTLASS Linear Algebra in CUDA](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)：分层分块与复用。
- [CUDA Best Practices：Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：实际带宽与内存访问优化。实际 HBM 流量、最佳 b 和速度需实测，本次未运行 GPU benchmark。

## 所属题单

- [[../sets/10月10号面试遇到的问题|10月10号面试遇到的问题]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
