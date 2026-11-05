# DCP Attention 的 AG 加 RS 与 A2A 合并有何区别？

- 整理状态：已整理
- 题目出处：用户在本次对话提供的技术问题与面试追问

## 30 秒回答

各 rank 对本地 KV 算 partial output 和 LSE，再按全局 softmax 分母加权合并。AG+RS 可先收集 LSE，再对加权输出 reduce-scatter。A2A 把局部输出与 LSE 按目标 rank 交换，由接收方加权归并。

## 深入解释

这一步发生在局部 attention 完成后。稀疏索引的全局 TopK 则在 attention 前。部分 backend 还有 query all-gather 等步骤，四个具体 kernel 的参数与代码本次留待后续源码复盘。

## 参考来源与待核实

- [[../../../../wiki/concepts/Decode Context Parallel|Decode Context Parallel]]。
- [vLLM DCP A2A 实现](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/ops/dcp/)。

## 所属题单

- [[../sets/推理并行、昇腾调优与 MLIR 面试追问|推理并行、昇腾调优与 MLIR 面试追问]]
- [[../README|秋招问题汇总]]
