# 异构 GPU 显存不均时如何动态分配训练任务？

- 整理状态：已整理
- 题目出处：[小红书分享图转录](https://chatgpt.com/library?file_id=libfile_08e498b5bde48191b68600b6e5b9507d&account_id=personal)，第 19 题

## 原题与条件

19. 如何解决分布式训练中由于 GPU 显存不均衡导致的负载倾斜问题？请设计一个动态任务分配算法，并考虑 PCIe 拓扑和 NVLink 域的影响。

## 30 秒回答

先做静态 TP/PP placement，留足模型状态、activation 和 workspace。再按 token 数、峰值显存和预计完成时间，动态分配可独立处理的 microbatch。

## 深入解释

过滤放不下的副本，再用排队时间+预计计算+跨域传输打分。优先同 NVLink 域，跨 PCIe 迁移要覆盖成本。固定模型分片只在安全边界重配。DP 有效 token 不同就按全局有效 token 归一化，并保证 collective 次序一致。

## 关联知识

- [[../../../../wiki/concepts/流水线并行|流水线并行]]

## 所属题单

- [[../sets/英伟达高性能计算工程师26题|英伟达高性能计算工程师26题]]
- [[../README|秋招问题汇总]]
