---
type: source
source_kind: 文章
topic: 并行与分布式
updated: 2026-07-26
---

# vllm PCP 与 DCP 深度解析

## 来源信息

- 标题：vllm PCP 与 DCP 深度解析
- 作者：小力龙虾
- 日期：2026-04-27（页面编辑时间）
- 类型：文章 / 源码与 RFC 解读
- 原始文件：[[../../raw/articles/vllm PCP 与 DCP 深度解析.md]]
- 原始链接：https://zhuanlan.zhihu.com/p/2032220302676063866

## 2-3 条核心摘要

- PCP 与 DCP 都沿序列/上下文维做并行，但针对不同阶段：PCP 分摊长 Prompt 的 Prefill Attention 计算与激活，目标偏向 TTFT；DCP 分片 Decode 历史 KV，并通过分布式 Softmax 合并精确输出，目标偏向 KV 容量、带宽和吞吐。
- PCP 的两类典型变形是 Ring Attention 与 Ulysses：前者固定本地 Q，让 K/V blocks 沿 ring 流动并用 Online Softmax 累积；后者对 Q/K/V 一起 All-to-All，把 Sequence Shard 临时转换成 Head Shard，本地完成部分 heads 的完整 Attention，再 All-to-All 切回。
- Chunked Prefill 与 PCP 互补：Chunked Prefill 在调度/时间维切块，控制峰值和公平性；PCP 在空间维让多 GPU 并行计算当前 Prefill chunk。

## 值得关注的论断

- DCP 的稳定数学机制是：各 rank 用本地 KV shard 计算 `local_out/local_lse`，再以全局 log-sum-exp 重新缩放并合并输出；不应笼统理解为把完整 KV AllGather 到每张卡。
- 原文对 DCP group/world size 存在冲突：既称复用 TP group、不增加 GPU，又给出 `TP × DCP` 增加 world size 的例子。本知识库保留现有 vLLM 来源的“复用 TP group”口径，并把 topology/参数约束标为版本相关。
- Ulysses 交换的不只是 Q head，而是 Q/K/V 一起从 sequence-sharded layout 重排到 head-sharded layout。它受可切分 head/KV-head 数限制；Ring Attention 对 head 数依赖较小，但通信轮数随 CP degree 增长。
- 原文的 GPU 利用率、算术强度、单卡上下文极限、PCP 上线版本与 CLI 参数缺少稳定边界，不作为确定事实。

## 关键概念

- [[../concepts/Prefill Context Parallel]]
- [[../concepts/Decode Context Parallel]]
- [[../concepts/Ring Attention]]
- [[../concepts/DeepSpeed Ulysses]]
- [[../concepts/Chunked Prefill]]
- [[../concepts/Online Softmax]]
- [[../concepts/MLA]]

## 相关实体

- [[../entities/vLLM]]

## 与现有 wiki 的关系

- 新建 PCP、Ring Attention 与 Ulysses 页面。
- 更新 DCP 的分布式 Softmax 数学口径，并显式标注本文 topology/communication 冲突。
- 更新 Chunked Prefill、Online Softmax、MLA、通信-计算重叠与 vLLM 页面。
- 未静默覆盖既有 DCP 结论；冲突保留在来源页和概念页研究备注中。

## 待确认

- vLLM PCP RFC 的合并状态、正式参数名、backend 和版本。
- `ag_rs/a2a` 精确交换的张量、collective 次数及 MLA/GQA backend 差异。
- PCP 与 DCP 联合 group 构造和 world-size 约束。
- Ring/Ulysses 通信复杂度与 causal load-balancing 方案需回到原论文和实现核实。
