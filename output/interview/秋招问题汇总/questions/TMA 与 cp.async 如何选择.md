# TMA 与 cp.async 如何选择？

- 整理状态：已整理
- 题目出处：用户提供的 Kimi AI Infra 一面复盘

## 30 秒回答

两者都能异步把 global memory 数据搬到 shared memory。`cp.async` 由线程发出 copy，适合较小、需要灵活逐线程寻址的搬运；TMA 由少量线程提交 tensor descriptor 描述的规则大块 tile，硬件处理地址生成，适合多维规则布局、较大 tile 以及 producer/consumer 流水。选择时看 tile 大小、布局与对齐、descriptor/setup 成本、同步方式及是否能与计算重叠；小而不规则的搬运不能因为是 Hopper/Blackwell 就一律改成 TMA。

截图系列追问：Hopper 上可由 tensor map 描述 global tensor 的维度、步长与布局，发起异步 global-to-shared tile 搬运；以相应 transaction `mbarrier` 跟踪到达与传输完成，consumer 等待完成后读取。每个 stage 必须等所有消费者完成才可复用，并正确管理 barrier phase 与预期字节数；具体指令、API、对齐及支持条件随 CUDA/PTX 版本变化，不能照搬伪代码。

## 关联知识

- [[../../../../wiki/concepts/Tiling|Tiling]]
- [[../../../../wiki/entities/NVIDIA Hopper|NVIDIA Hopper]]

## 参考来源与待核实

- [CUDA Programming Guide，异步搬运](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)；[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)。

## 所属题单

- [[../sets/Kimi AI Infra一面|Kimi AI Infra 一面]]
- [[../sets/CUDA 截图系列15题|CUDA 截图系列15题]]（用户提供的截图系列摘要，第 9 题）
- [[../README|秋招问题汇总]]
