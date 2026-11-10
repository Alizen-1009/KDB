# CUDA L2 persistence 如何设置，有哪些限制？

- 整理状态：已整理
- 题目出处：用户提供的 CUDA 截图系列摘要，第 4 题。

## 30 秒回答

对于跨 kernel 重复使用的工作集，可在支持的平台查询 L2 set-aside 和 access policy window 上限，设置适当的 persisting 区域，再在 stream 的窗口配置基址、大小、hit ratio 和访问属性。它是缓存优先级提示而非锁定或命中保证；窗口并发、竞争和容量都影响效果，必须实测。

## 深入解释

窗口大小及 set-aside 均受设备属性约束，工作集过大或多个 stream 竞争可能互相驱逐；hit ratio 可用于控制持久化比例，不是实测命中率。完成后恢复 normal 或重置持久化状态，避免影响其他工作。MIG、MPS 等模式可能限制设置方式或可用性，依设备与 CUDA 文档查询，比较整体延迟及其他流受影响情况。

## 参考来源与待核实

- [CUDA Programming Guide：L2 Access Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-l2-access-management)。平台、驱动及具体访问策略须实机验证。

## 所属题单

- [[../sets/CUDA 截图系列15题|CUDA 截图系列15题]]
- [[../README|秋招问题汇总]]
