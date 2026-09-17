# 如何估算推理系统理论上限，并定义 SLA 下的最大吞吐？

- 整理状态：已整理
- 题目出处：用户在本次对话中的面试追问

## 其他问法

- 有没有算过当前推理系统设计的理论性能上限？
- 如果达不到上限，可能受哪些因素影响？SLA 是什么意思？

## 30 秒回答

先固定模型、dtype、GPU/并行配置、batch 和输入输出长度分布，再分别算每步 `FLOPs/峰值算力`、`bytes/有效带宽`、`通信量/链路带宽`。理想完全重叠时，step time 下界是三者最大值；不能重叠时要加暴露部分。实测差距通常来自小 GEMM、KV 访存、launch、通信、动态 batching、tail、CPU 与功耗降频。生产上真正的上限是满足 P99 TTFT、TPOT、错误率等 SLA 时的最大 QPS/tokens/s，而不是离线无限 batch 的峰值。

## 深入解释

```text
T_compute = FLOPs / compute_peak
T_memory  = bytes / bandwidth
T_comm    = communication_bytes / link_bandwidth
T_ideal ≥ max(T_compute, T_memory, T_comm)  # 理想重叠
```

SLA 是对外服务等级协议；SLI 是实际测量指标，SLO 是内部目标。常见指标包括 P99 TTFT、P99 TPOT、成功率、可用性和超时率。

## 关联知识

- [[../../../../wiki/concepts/Roofline 模型|Roofline 模型]]
- [[../../../../wiki/concepts/Benchmarking|Benchmarking]]

## 参考来源与待核实

- 理论峰值必须与 dtype、稀疏/dense 口径和实际指令路径一致；kernel 上限不能直接代替端到端系统上限。

## 所属题单

- [[../sets/近期对话面试问题回顾|近期对话面试问题回顾]]
- [[../sets/量化与性能|量化与性能]]
- [[../README|秋招问题汇总]]
