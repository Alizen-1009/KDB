# PD 分离中的 Prefill 与 Decode 实例配比如何确定？

- 整理状态：已整理
- 题目出处：用户提供的一面面经（SGLang / CUDA 系统专题）

## 其他问法

- PD 分离配比怎么定？Decode 多大 batch size 合适？怎么由此反推 Prefill 的数量？

## 30 秒回答

P:D 配比不能背固定数字，要按到达率和每请求两个阶段的平均 GPU 服务时间做容量规划。粗略地，若每秒到达 `λ` 个请求，平均 prefill GPU 时间为 `t_p`，平均 decode GPU 时间为 `t_d`，则两池需求比例先看 `λt_p : λt_d`，即 `t_p:t_d`；再通过负载测试加入 P99、突发流量、KV 传输和冗余余量。Prefill 受输入长度影响，decode 受输出长度、batch 和 TPOT 影响，所以真实配比必须随 workload 动态调整。

## 深入解释

### 英伟达五面补充

- 本次题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，AI Infra 与推理系统第 10 题；2026-11-05 收录，实际面试日期与具体轮次未提供。
- 本次原题：PD 分离配比怎么定？Decode 多大 batch size 合适？怎么由此反推 Prefill 的数量？

Decode 每请求每步生成一个 token 时，batch 为 B、步时为 t_step：

$$q_D(B)\approx\frac{B}{t_{step}(B)}$$

在 TPOT_P99≤SLA、显存和上下文长度约束下选 B，而不是追求最大 batch。投机解码、多 token step 等情形需要改用每步实际产生的有效 tokens。

设 λ 为 requests/s，平均输出长度为 L_o，每请求未命中输入长度为 L_miss，P/D 单实例有效吞吐分别为 q_P/q_D，则一阶容量模型为：

$$n_Dq_D\ge\lambda L_o,\qquad n_Pq_P\ge\lambda L_{miss}$$

因此在近似饱和且不计余量时：

$$\frac{n_P}{n_D}\approx\frac{L_{miss}q_D}{L_oq_P}$$

若已有 n_D，可先估 λ≈n_Dq_D/L_o，再推 n_P。实例规格必须明确；P/D 每实例卡数不同，还需转换为 GPU 数或成本，不能把实例比例当作卡数比例。

q_P 必须在相近历史长度、命中模式和输入分布下测量；相同未命中长度的 Attention 成本也会随前缀长度变化。实际还需加入队列、组批非线性、KV 传输、突发、反压、冗余及首 token 处理口径，不能从平均吞吐直接保证 P99。

容量估算时至少分别测：

- Prefill：输入长度分布、prefix-cache 命中率、chunk size、TTFT。
- Decode：输出长度分布、活跃序列数、TPOT、每步有效 batch。
- 交接：KV bytes、网络带宽、注册内存、排队和失败重试。
- SLA：P95/P99 TTFT、TPOT、超时率与目标 QPS。

如果 prefill 队列持续增长而 decode 空闲，应增加 P 或提高 prefix 命中；反之则增加 D。生产中通常需要自动扩缩容和 admission control，而不是永久固定比例。

## 关联知识

- [[../../../../wiki/concepts/Prefix Caching|Prefix Caching]]

- [[../../../../wiki/concepts/Continuous Batching|Continuous Batching]]

- [[../../../../wiki/concepts/PD分离|PD分离]]
- [[PD 解耦与物理分离如何实现，KV 和请求状态怎样交接|PD 状态交接]]

## 参考来源与待核实

- 本页公式是排队与容量规划的一阶近似，不包含组批非线性、并行效率和阶段间反压；最终配比必须由真实流量压测确定。

### 本次补充依据与边界

- 以上为容量规划近似，非固定最佳配比；目标 batch、q_P/q_D 和余量需要实测。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/SGLang与CUDA系统一面|SGLang 与 CUDA 系统一面]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
