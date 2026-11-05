# DCP 稀疏索引如何选全局 TopK，两个 pack 有何区别？

- 整理状态：已整理
- 题目出处：用户在本次对话提供的技术问题与面试追问

## 30 秒回答

通用路径是每 rank 本地 TopK，打包 score 和全局 token ID，all-gather 后做全局 TopK。局部排名超过 K 的候选已有至少 K 个同分片候选排在它前面。在本地与全局使用一致的排序及同分规则时，本地各留 K 个足够得到全局结果。

## 深入解释

同分时须查具体 backend 的 tie-break，stable 的具体实现待按 commit 核实。pack_seq 面向不等长 speculative decode query，例如把长度 3 与 1 的请求排成两行、每行三格，再按有效长度处理。DCP candidate pack 在打分及本地筛选之后整理通信缓冲区。

## 参考来源与待核实

- [vLLM sparse_attn_indexer](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/sparse_attn_indexer/)。

## 所属题单

- [[../sets/推理并行、昇腾调优与 MLIR 面试追问|推理并行、昇腾调优与 MLIR 面试追问]]
- [[../README|秋招问题汇总]]
