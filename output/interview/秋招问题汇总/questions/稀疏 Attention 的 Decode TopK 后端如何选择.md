# 稀疏 Attention 的 Decode TopK 后端如何选择？

- 整理状态：已整理
- 题目出处：用户在本次对话提供的技术问题与面试追问

## 30 秒回答

vLLM 的 DSA indexer 为每行 query score 选历史 token。文档中的 auto 依次考虑 cooperative、persistent、per-row，具体选择取决于设备和输入限制。

## 深入解释

超长 row 可以分段，每段选候选，再合并为整行 TopK。该技巧可避免由单个工作单元处理整个长 row。具体触发阈值和分段分支待按源码 commit 确认。

## 参考来源与待核实

- [vLLM CLI sparse-indexer-topk-backend](https://docs.vllm.ai/en/latest/cli/serve/)。

## 所属题单

- [[../sets/推理并行、昇腾调优与 MLIR 面试追问|推理并行、昇腾调优与 MLIR 面试追问]]
- [[../README|秋招问题汇总]]
