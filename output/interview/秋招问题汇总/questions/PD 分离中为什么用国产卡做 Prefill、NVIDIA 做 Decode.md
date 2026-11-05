# PD 分离中为什么用国产卡做 Prefill、NVIDIA 做 Decode？

- 整理状态：已整理
- 题目出处：用户在本次对话提供的技术问题与面试追问

## 30 秒回答

按同一模型和流量的实测分工。如果国产卡的 Prefill 算子对大矩阵 workload 已有足够好的吞吐和成本，NVIDIA 的 Decode 路径在目标上下文及并发下有更好的尾延迟和 goodput，就可以这样组合。

## 深入解释

Prefill 通常能利用较大的矩阵，Decode 反复读权重及历史 KV，也可能卡在小 kernel、MoE 通信或调度。异构交接要核对 KV dtype、布局、TP 分片与传输路径，测出传输和重排成本。用同流量和同 SLO 比较两池资源配比、端到端延迟、达标吞吐与成本。

## 参考来源与待核实

- [[../../../../wiki/concepts/PD分离|PD分离]]。
- [vLLM PD 分离与 KV connector](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)。

## 所属题单

- [[../sets/推理并行、昇腾调优与 MLIR 面试追问|推理并行、昇腾调优与 MLIR 面试追问]]
- [[../README|秋招问题汇总]]
