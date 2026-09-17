# PD 分离中的 Prefill 与 Decode 实例配比如何确定？

- 整理状态：已整理
- 题目出处：用户提供的一面面经（SGLang / CUDA 系统专题）

## 30 秒回答

P:D 配比不能背固定数字，要按到达率和每请求两个阶段的平均 GPU 服务时间做容量规划。粗略地，若每秒到达 `λ` 个请求，平均 prefill GPU 时间为 `t_p`，平均 decode GPU 时间为 `t_d`，则两池需求比例先看 `λt_p : λt_d`，即 `t_p:t_d`；再通过负载测试加入 P99、突发流量、KV 传输和冗余余量。Prefill 受输入长度影响，decode 受输出长度、batch 和 TPOT 影响，所以真实配比必须随 workload 动态调整。

## 深入解释

容量估算时至少分别测：

- Prefill：输入长度分布、prefix-cache 命中率、chunk size、TTFT。
- Decode：输出长度分布、活跃序列数、TPOT、每步有效 batch。
- 交接：KV bytes、网络带宽、注册内存、排队和失败重试。
- SLA：P95/P99 TTFT、TPOT、超时率与目标 QPS。

如果 prefill 队列持续增长而 decode 空闲，应增加 P 或提高 prefix 命中；反之则增加 D。生产中通常需要自动扩缩容和 admission control，而不是永久固定比例。

## 关联知识

- [[../../../../wiki/concepts/PD分离|PD分离]]
- [[PD 解耦与物理分离如何实现，KV 和请求状态怎样交接|PD 状态交接]]

## 参考来源与待核实

- 本页公式是排队与容量规划的一阶近似，不包含组批非线性、并行效率和阶段间反压；最终配比必须由真实流量压测确定。

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/SGLang与CUDA系统一面|SGLang 与 CUDA 系统一面]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
