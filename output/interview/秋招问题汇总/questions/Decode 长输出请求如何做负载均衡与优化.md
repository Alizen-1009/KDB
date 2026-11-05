# Decode 长输出请求如何做负载均衡与优化？

- 整理状态：已整理
- 题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，AI Infra 与推理系统第 8 题；2026-11-05 收录，实际面试日期与具体轮次未提供。

## 原题与条件

Decode 如果输出长，怎么做负载均衡？怎么优化？

## 30 秒回答

长输出既延长 worker 占用时间，也让 KV 和每步 Attention 工作量持续增长，因此不能只按请求数均衡。结合上下文长度、KV bytes、step time 和预计剩余生成量路由，配合 Continuous Batching、长短分池和必要的迁移；迁移前比较 KV 搬运成本。

## 深入解释

两个 worker 各有 10 个请求，但一个全是短上下文、另一个全是长上下文，其负载可能显著不同。输出长度往往未知，预测只能作为有误差的估计，不能假设能精确获得剩余长度。

- Continuous Batching：完成请求及时退出，新请求及时补入，提高有效 batch。
- 长短请求分池或分桶：控制长期占用和尾延迟，同时防止长请求饥饿。
- KV-aware/cache-aware 路由：同时考虑历史长度、显存、缓存位置和计算负载。
- 必要时迁移：仅在预期节省的后续等待/执行成本足以覆盖搬运与同步时使用。
- Kernel/并行：长上下文小 batch 可评估 Split-KV 或 DCP；大 batch 不必一律切分。
- 投机解码：若接受率与验证成本合适，减少串行步数；不是无条件收益。

MoE 还要区分请求级均衡与 expert/rank 级均衡：请求分布均匀仍可能出现热门 experts，EPLB 等机制解决的是另一层问题。

## 关联知识

- [[../../../../wiki/concepts/Continuous Batching|Continuous Batching]]
- [[../../../../wiki/concepts/缓存感知路由|缓存感知路由]]
- [[../../../../wiki/concepts/Flash Decoding|Flash Decoding]]
- [[../../../../wiki/concepts/Expert Parallel Load Balancing|Expert Parallel Load Balancing]]
- [[../../../../wiki/concepts/Speculative Decoding|Speculative Decoding]]

## 参考来源与待核实

- 本页为系统设计归纳；未给出统一负载打分公式或迁移阈值，需真实 workload 压测。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
