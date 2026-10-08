# Ulysses Sequence Parallelism 与 Ring Attention 有什么区别？

- 整理状态：已整理
- 题目出处：用户提供的 Kimi AI Infra 一面复盘

## 30 秒回答

二者都沿 sequence 扩展长上下文 attention，但数据流不同。Ulysses 先用 All-to-All 把“部分 sequence、全部 heads”换成“完整 sequence、部分 heads”，本地完成 attention 后再 All-to-All 换回来；本地 softmax 不需跨 rank 合并。Ring Attention 固定本地 Q，让 KV 块沿 ring 逐轮传递，每轮做部分 attention，并用 online softmax 精确合并。Ulysses 易复用本地 attention kernel，但受 head 切分和 All-to-All 影响；Ring 可把 KV 通信与计算流水重叠，但有多轮通信和因果 mask 负载均衡问题。

## 关联知识

- [[../../../../wiki/concepts/DeepSpeed Ulysses|DeepSpeed Ulysses]]
- [[../../../../wiki/concepts/Ring Attention|Ring Attention]]
- [[../../../../wiki/concepts/Online Softmax|Online Softmax]]

## 参考来源与待核实

- [DeepSpeed Ulysses 论文](https://arxiv.org/abs/2309.14509)；[Ring Attention 论文](https://arxiv.org/abs/2310.01889)。
- GQA/MQA 下的 KV head 复制和通信量取决于具体实现，不能只用 Q head 数判断。

## 所属题单

- [[../sets/Kimi AI Infra一面|Kimi AI Infra 一面]]
- [[../README|秋招问题汇总]]
