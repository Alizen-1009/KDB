# 千卡训练如何设计梯度同步并比较压缩与异步 SGD？

- 整理状态：已整理
- 题目出处：[小红书分享图转录](https://chatgpt.com/library?file_id=libfile_08e498b5bde48191b68600b6e5b9507d&account_id=personal)，第 17 题

## 原题与条件

17. 在千卡规模的分布式训练系统中，如何设计梯度同步策略以最小化通信开销？请对比同步 SGD、异步 SGD 和梯度压缩（如 1-bit SGD）的 trade-off。

## 30 秒回答

先以同步 DP 为基线。Backward 中 gradient bucket 一就绪就异步归约，再按拓扑映射并测试合适的 bucket 大小及梯度累积。

## 深入解释

同步 SGD 易于维护一致更新，受慢 rank 影响；异步 SGD 减少等待但引入 stale gradient；1-bit 压缩减少字节，同时引入编解码、scale、误差反馈和收敛代价。看 exposed communication、step time、固定验证指标所需的 wall time。梯度累积要明确全局 batch 与学习率语义。

## 关联知识

- [[../../../../wiki/concepts/DDP|DDP]]

## 参考来源与待核实

- [参考 1](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)

## 所属题单

- [[../sets/英伟达高性能计算工程师26题|英伟达高性能计算工程师26题]]
- [[../README|秋招问题汇总]]
