# Kernel 内同步、Kernel 间数据交接和跨卡通信有何区别？

- 整理状态：已整理
- 题目出处：用户在本次对话提供的技术问题与面试追问

## 30 秒回答

Kernel 内按参与线程的范围选 barrier。Kernel 间同 stream 用顺序，跨 stream 可用 event 建依赖。跨卡还需用 collective 或相应通信方式交换数据，并让消费者等通信完成。

## 深入解释

例如 block 内先写 shared memory 再由同 block 读取，可用 __syncthreads。Kernel A 写全局显存后，另一 stream 的 Kernel B 读之前要等 A 的 event。DCP all-gather 则把各 rank 的候选发给其他 rank，后续 TopK 等通信缓冲区就绪。

## 所属题单

- [[../sets/推理并行、昇腾调优与 MLIR 面试追问|推理并行、昇腾调优与 MLIR 面试追问]]
- [[../README|秋招问题汇总]]
