# CUDA 直方图如何用 shared memory 降低原子热点？

- 整理状态：已整理
- 题目出处：用户提供的 CUDA 截图系列摘要，第 10 题。

## 30 秒回答

每个 block 清零 shared 私有桶，同步后对输入做 shared atomicAdd，再同步，将各 block 的桶合并到全局；相比每个元素都更新全局桶，通常减少全局原子次数。热点仍可能在 shared 桶内串行，可进一步 warp 私有或分片，最后再合并；额外空间、清零和归并成本要一起测。

## 深入解释

桶多时私有直方图可能占满 shared、降低 occupancy；偏斜分布下才更可能受热点影响。可按数据聚合相同桶的更新，但要正确处理有效 lane、计数溢出和所有同步；分别测试均匀和偏斜数据并与全局原子基线比较。

## 关联知识

- [[../../../../wiki/concepts/Histogram|Histogram]]
- [[如何用 CUDA 直方图和前缀和查找第 K 大元素|直方图与第 K 大]]

## 参考来源与待核实

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html)：shared-memory histogram 模式；无具体硬件或性能实测。

## 所属题单

- [[../sets/CUDA 截图系列15题|CUDA 截图系列15题]]
- [[../README|秋招问题汇总]]
