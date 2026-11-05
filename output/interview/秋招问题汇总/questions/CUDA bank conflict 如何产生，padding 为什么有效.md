# CUDA bank conflict 如何产生，padding 为什么有效？

- 整理状态：已整理
- 题目出处：[小红书分享图转录](https://chatgpt.com/library?file_id=libfile_08e498b5bde48191b68600b6e5b9507d&account_id=personal)，第 6 题

## 原题与条件

6. 分析 CUDA 中 bank conflict 产生的条件，给出一个通过 padding 或调整数据布局解决 bank conflict 的实际案例。

## 30 秒回答

一次 shared 事务里，多个 lane 命中同一 bank 的不同地址需要拆分服务。同地址读取可以广播。

## 深入解释

32 个 bank、每个 word 4B 的模型下，lane l 读 float tile[32][32] 的 tile[l][c]，bank 是 c；改成 [32][33] 后 bank 是 (l+c)%32。矩阵转置先合并读、同步、再从 shared 按转置布局读并合并写；向量访问按实际拆分的事务判断。

## 关联知识

- [[../../../../wiki/concepts/Bank Conflict|Bank Conflict]]

## 参考来源与待核实

- [参考 1](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)

## 所属题单

- [[../sets/英伟达高性能计算工程师26题|英伟达高性能计算工程师26题]]
- [[../README|秋招问题汇总]]
