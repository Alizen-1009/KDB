# DeepEP normal 与 low-latency 模式有什么区别？

- 整理状态：已整理
- 题目出处：用户提供的 Kimi AI Infra 一面复盘

## 30 秒回答

DeepEP 解决 MoE 的 token dispatch/combine。传统 normal（高吞吐）路径侧重训练和 prefill 的较大 token 批量，追求通信吞吐及与专家计算重叠；low-latency 路径侧重 decode 的小批量，重点是减少启动、同步与通信往返开销。Decode 时专家接收数量若要回到 CPU 才能决定输出 shape，会阻断流水和 CUDA Graph；可用预分配容量、GPU 侧计数及有效范围 mask 避免这次 CPU 同步。两条路径的取舍要按批量、拓扑和版本实测。

## 深入解释

- “normal / low-latency”是旧版常见称呼。当前 DeepEP V2.5 README 将高吞吐和低延迟操作统一到 `EPBuffer`；`do_cpu_sync=True` 可取精确输出大小，`False` 则用 GPU 侧计数和容量上限，专家 GEMM 必须只处理有效范围。
- “rank-major 去重”不是区分两模式的稳定定义。路由去重、发送布局、expert-major 展开及接收顺序要绑定具体版本和函数，不应仅凭这几个字推断算法。

## 关联知识

- [[../../../../wiki/entities/DeepEP|DeepEP]]
- [[../../../../wiki/concepts/Expert Parallelism|Expert Parallelism]]

## 参考来源与待核实

- [DeepEP README，V2.5 接口及训练、prefill、decode 示例](https://github.com/deepseek-ai/DeepEP/blob/main/README.md)。
- 待核实：面试官所指的 normal / low-latency 具体版本，以及“rank-major 去重”对应的数据布局或代码路径。

## 所属题单

- [[../sets/Kimi AI Infra一面|Kimi AI Infra 一面]]
- [[../README|秋招问题汇总]]
