# CUDA 语境中的全双工、半双工与 cache line 分别是什么？

- 整理状态：整理中
- 题目出处：用户提供的一面面经（SGLang / CUDA 系统专题）

## 30 秒回答

半双工表示共享链路在同一时刻只能朝一个方向传输，全双工表示两个方向可以同时传输；它通常描述 PCIe、NVLink、网络链路或 DMA copy engine 能力。cache line 则是 cache 填充、命中和一致性处理的数据粒度。把“cache line 是半双工还是全双工”放在一起并不是标准 CUDA 术语，应该先向面试官确认他问的是 DRAM 数据总线、GPU 互联，还是 H2D/D2H copy overlap。

## 深入解释

- **链路层**：全双工关注 `A→B` 与 `B→A` 能否同时传输，以及双向带宽是分别计数还是共享。
- **CUDA copy 层**：若设备具有相应异步 copy engine，使用 pinned host memory、异步拷贝和不同非默认 stream，可能重叠 H2D、D2H 与 kernel；是否能同时双向拷贝要查设备属性和拓扑。
- **DRAM/cache 层**：同一 DRAM channel 的读写通常涉及方向切换和总线调度，不能把互联链路的“全双工”概念直接套到 cache line。
- **cache line**：一次 miss 可能搬入比程序实际使用更多的字节；访问是否合并、对齐以及相邻线程访问模式决定有效带宽和浪费。

## 常见误区

- `cudaMemcpyAsync` 不等于必然发生并发拷贝。
- PCIe/NVLink 的链路能力不等于应用能同时跑满两个方向。
- cache 能处理 load/store 不等于它应该被称为“全双工 cache”。更准确的说法是读写端口、pipeline、带宽和冲突。

## 参考来源与待核实

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- 待确认：面试官原话中的“cache 半双工、双工”具体指 GPU interconnect、DMA copy engine、HBM/GDDR channel，还是 cache read/write port。本页不替面试官强行指定唯一含义。

## 所属题单

- [[../sets/SGLang与CUDA系统一面|SGLang 与 CUDA 系统一面]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
