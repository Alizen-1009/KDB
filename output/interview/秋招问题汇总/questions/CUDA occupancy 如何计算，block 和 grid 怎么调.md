# CUDA occupancy 如何计算，block 和 grid 怎么调？

- 整理状态：已整理
- 题目出处：[小红书分享图转录](https://chatgpt.com/library?file_id=libfile_08e498b5bde48191b68600b6e5b9507d&account_id=personal)，第 7、9 题

## 原题与条件

7. 解释 CUDA Occupancy 的计算公式，并说明如何通过调整 block size、shared memory 大小和寄存器使用来优化 occupancy。
9. 在 CUDA 程序中，如何通过调整 block 和 grid 的大小来最大化 SM 占用率？请结合具体的 GPU 架构（如 A100 或 H100）说明。

## 30 秒回答

理论 occupancy = 驻留 CTA × ceil(threads/32) ÷ 最大驻留 warp 数。CTA 驻留同时受线程、warp、寄存器、shared 和 block 上限约束。Grid 决定有多少 CTA 可以填满整张卡及最后一轮的尾部。

## 深入解释

A100/H100 上限均为 64 warps/SM。256 threads/block、64 KiB shared/CTA 时，只按 164/228 KiB 容量估算，驻留至多 2/3 个 block，对应 25%/37.5%。实际核对 carveout、资源分配粒度、寄存器和 CUDA occupancy API。小 grid 可试小输出 tile 或 split-K，后者付出归约代价。比较 eligible warps、spill 和实测时延。

Occupancy 高不保证更快：足够的活跃 warp 可隐藏延迟，但强行减少寄存器可能造成 spill，缩小 tile 也可能破坏数据复用。应同时看带宽、计算吞吐、stall、spill 和实际耗时。

## 关联知识

- [[../../../../wiki/concepts/Occupancy|Occupancy]]

## 参考来源与待核实

- [参考 1](https://docs.nvidia.com/cuda/hopper-tuning-guide/)

## 所属题单

- [[../sets/英伟达高性能计算工程师26题|英伟达高性能计算工程师26题]]
- [[../sets/C++与推理系统及 CUDA 面试回顾|本轮面试回顾]]
- [[../README|秋招问题汇总]]
