# GPU 有哪些内存与缓存，地址空间和物理层次如何区分？

- 整理状态：已整理
- 题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，AI Infra 硬件与算子第 6 题；2026-11-05 收录，实际面试日期与具体轮次未提供。

## 原题与条件

讲一下 GPU 有哪些 memory？

## 30 秒回答

区分程序可见地址空间和物理层次。Register、local、shared、global、constant 描述不同作用域和访问方式；L1/L2 是硬件缓存。Local 是线程私有但通常由显存支持，shared 是显式管理的片上存储。还可补充 host/pinned、Unified Memory，以及特定架构的专用存储。

## 深入解释

| 类型 | 作用域/用途 | 关键边界 |
|---|---|---|
| Register | 线程局部标量、累加等 | 数量有限，过大可能 spill |
| Local memory | 线程私有数组、栈、spill 等 | 不是天然片上低延迟内存 |
| Shared memory | 通常 block 内协作 | 显式管理，受容量和 bank conflict 限制 |
| Global memory | 设备大容量地址空间 | 通常由显存支持，可经缓存访问 |
| Constant memory | 只读常量地址空间 | warp 同址读取适合广播，分散地址可能串行 |
| L1/L2 | 硬件管理缓存 | 不是与 global/shared 并列的变量声明方式 |
| Texture/read-only 路径 | 特定读取路径 | 具体能力与缓存关系依架构而定 |

Host/pinned memory 影响 CPU↔GPU 传输，Unified Memory 涉及统一管理与可访问性，不等于“没有搬运”。Blackwell TMEM 等专用片上存储有独立指令和用途，不能按普通 register/shared 随意使用。

知识库已有完整原理，见关联知识；这里不固定某型号容量、延迟或带宽数字。

## 关联知识

- [[../../../../wiki/concepts/CUDA内存层次|CUDA内存层次]]
- [[../../../../wiki/concepts/Bank Conflict|Bank Conflict]]
- [[../../../../wiki/concepts/Tensor Memory|Tensor Memory]]

## 参考来源与待核实

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)。
- 具体容量、共享作用域扩展和缓存行为需核对 GPU 架构。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
