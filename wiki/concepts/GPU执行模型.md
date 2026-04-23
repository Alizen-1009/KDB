# GPU执行模型

## 定义

描述 GPU 如何以 `thread / warp / block / SM` 这一层级结构组织并执行并行计算的模型。

## 它解决什么问题

- 帮助理解为什么同一段 CUDA / kernel 代码在 GPU 上会表现出和 CPU 完全不同的性能特征
- 为分析控制分歧、共享内存复用和并发调度提供统一抽象

## 核心机制

- 单个线程执行最细粒度的计算
- `warp` 是 32 个连续线程共同执行的基本调度单位
- `block` 是线程组，运行在单个 `SM` 上并共享 shared memory
- 多个 `SM` 彼此独立调度不同 block，从而形成大规模吞吐并行

## 关键权衡

- SIMT 模型让大量同构计算更容易扩展
- 一旦出现严重分支分歧或不规则访存，执行效率会迅速下降

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 5 - GPUs]]
- [[../sources/你一定要知道：CUDA优化六要]]

## 相关概念

- [[内存合并访问]]
- [[Bank Conflict]]
- [[Occupancy]]
- [[Warp Divergence]]
- [[Tail Effect]]
- [[Tiling]]
- [[FlashAttention]]

## 研究备注

- 后续可补具体到 NVIDIA GPU 的 SM 资源组成、occupancy 和 warp scheduler 细节
