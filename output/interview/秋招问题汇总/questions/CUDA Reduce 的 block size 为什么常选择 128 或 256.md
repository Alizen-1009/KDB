# CUDA Reduce 的 block size 为什么常选择 128 或 256？

- 整理状态：已整理
- 题目出处：既有面试原稿（见文末具体章节）

## 30 秒回答

128/256 是 4/8 个 warp，便于 warp 内归约和少量跨 warp 合并，通常也是平衡并发、每线程工作量和寄存器资源的好起点。它不是固定最优值；要按 reduce 长度、每线程寄存器、共享内存、block 数和目标 GPU 测量。

## 深入解释

小 reduce 可用单 warp；长 reduce 可能需要多个元素/线程或多 block 两阶段归约。occupancy 高也不保证最快。

## 关联知识

- [[../../../../wiki/concepts/Block Reduce|Block Reduce]]

## 参考来源与待核实

- [CUDA 最佳实践](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [[../../字节二面高压题拆解#9.3 优化点怎么讲|字节二面高压题拆解]]
- [[../../字节二面高压题拆解#10. 这轮面试最容易被追问的 10 个点|字节二面高压题拆解]]

- 待核实 / 原稿边界：原稿只说 128/256“往往更稳”并列为追问，没有给 occupancy、寄存器、shared memory、硬件或 benchmark 依据；短答留空，不能背成普适最优值。

## 所属题单

- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
