# TMA 与 cp.async 如何选择？

- 整理状态：已整理
- 题目出处：用户提供的 Kimi AI Infra 一面复盘

## 30 秒回答

两者都能异步把 global memory 数据搬到 shared memory。`cp.async` 由线程发出 copy，适合较小、需要灵活逐线程寻址的搬运；TMA 由少量线程提交 tensor descriptor 描述的规则大块 tile，硬件处理地址生成，适合多维规则布局、较大 tile 以及 producer/consumer 流水。选择时看 tile 大小、布局与对齐、descriptor/setup 成本、同步方式及是否能与计算重叠；小而不规则的搬运不能因为是 Hopper/Blackwell 就一律改成 TMA。

## 关联知识

- [[../../../../wiki/concepts/Tiling|Tiling]]
- [[../../../../wiki/entities/NVIDIA Hopper|NVIDIA Hopper]]

## 参考来源与待核实

- [CUDA Programming Guide，异步搬运](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)；[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)。

## 所属题单

- [[../sets/Kimi AI Infra一面|Kimi AI Infra 一面]]
- [[../README|秋招问题汇总]]
