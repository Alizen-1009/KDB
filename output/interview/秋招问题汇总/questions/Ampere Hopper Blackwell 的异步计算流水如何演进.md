# Ampere、Hopper、Blackwell 的异步计算流水如何演进？

- 整理状态：已整理
- 题目出处：用户提供的 Kimi AI Infra 一面复盘

## 30 秒回答

以数据中心 A100、H100、B200 的 Tensor Core kernel 为例：Ampere 的 `cp.async` 让 global 到 shared 的搬运能与当前 tile 计算重叠；Hopper 加入按 tensor tile 搬运的 TMA、异步 WGMMA、transaction barrier，更适合 producer/consumer 分工；Blackwell SM100 的主路径是 `tcgen05`，引入 Tensor Memory 承载 MMA accumulator，并与 TMA、barrier 组成更显式的流水。核心变化是数据搬运、MMA 和结果消费能更独立地排程，但不是新架构的所有 kernel 都必须使用这些机制。

## 关联知识

- [[../../../../wiki/entities/NVIDIA Ampere|Ampere]]
- [[../../../../wiki/entities/NVIDIA Hopper|Hopper]]
- [[../../../../wiki/entities/NVIDIA Blackwell|Blackwell]]
- [[../../../../wiki/concepts/Tensor Memory|Tensor Memory]]

## 参考来源与待核实

- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)；[Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)。
- 这里的 Blackwell 限定 SM100 数据中心路径，不泛指所有 Blackwell SKU。

## 所属题单

- [[../sets/Kimi AI Infra一面|Kimi AI Infra 一面]]
- [[../README|秋招问题汇总]]
