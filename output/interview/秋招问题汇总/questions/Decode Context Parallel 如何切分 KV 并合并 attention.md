# Decode Context Parallel 如何切分 KV 并合并 attention？

- 整理状态：已整理
- 题目出处：用户提供的 Kimi AI Infra 一面复盘

## 30 秒回答

DCP 针对 decode 每步 Q 很短、历史 KV 很长的情况，把同一请求的 KV 沿 context/token 维分给多个 rank。Q 需要让持有各 KV shard 的 rank 可用；各 rank 用本地 KV 算 partial attention，得到局部输出和 LSE，再按 LSE 权重精确合并输出。它尤其适合 KV heads 少、普通 TP 会复制 KV 的场景。收益是降低每 rank KV 容量和读取压力，代价是 Q 分发与 partial output/LSE 的通信；不是简单把局部输出求平均。

## 深入解释

在 vLLM 的 DCP 口径里，它复用已有 TP group，不额外乘一组 GPU；KV 可以按 token index 交错放置，而非必须连续分段。对 shard `r` 的归一化输出 `o_r` 与 `lse_r`，令 `L=logsumexp_r(lse_r)`，则 `o=Σ_r exp(lse_r-L)o_r`。

## 关联知识

- [[../../../../wiki/concepts/Decode Context Parallel|Decode Context Parallel]]
- [[../../../../wiki/concepts/Online Softmax|Online Softmax]]

## 参考来源与待核实

- [vLLM Decode Context Parallel 文档](https://docs.vllm.ai/en/latest/serving/context_parallel_deployment/)；具体通信原语与 KV 布局需绑定 vLLM 版本和 attention backend。

## 所属题单

- [[../sets/Kimi AI Infra一面|Kimi AI Infra 一面]]
- [[../README|秋招问题汇总]]
