# FlashDecoding 与 FlashDecoding++ 如何优化小 Batch、长上下文？

- 整理状态：已整理
- 题目出处：用户在本次对话中提供的面试题截图

## 30 秒回答

FlashDecoding 面向 `q_len≈1`、batch 小而 KV 很长的 decode：沿 context 把 KV 切成多个 split，让多个 CTA 并行计算局部 `max/denominator/output`，再用 Online Softmax 规则精确合并，从而补足仅靠 batch/head 时的并行度。FlashDecoding++ 不是简单的 split-KV v2，而是一套更宽的推理引擎优化：用统一缩放基准减少 partial-softmax 同步，并优化小 M 的 flat GEMM、双缓冲和按 shape/硬件选择数据流。

## 常见误区

- Split-KV 增加并行度，但仍需读取历史 KV，并引入 partial result 与归约开销。
- FlashDecoding++ 同时优化 attention 之外的 decode GEMM，不能只解释成更多 KV splits。

## 关联知识

- [[../../../../wiki/concepts/Flash Decoding|Flash Decoding]]
- [[../../../../wiki/concepts/Online Softmax|Online Softmax]]

## 参考来源与待核实

- [FlashDecoding++](https://arxiv.org/abs/2311.01282)
- 是否启用 split-KV 依赖 batch、head、context、page layout、硬件和归约开销，不是所有 decode 的固定最优路径。

## 所属题单

- [[../sets/近期对话面试问题回顾|近期对话面试问题回顾]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
