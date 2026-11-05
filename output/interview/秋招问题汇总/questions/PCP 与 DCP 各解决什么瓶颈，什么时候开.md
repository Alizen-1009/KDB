# PCP 与 DCP 各解决什么瓶颈，什么时候开？

- 整理状态：已整理
- 题目出处：用户在本次对话提供的技术问题与面试追问

## 30 秒回答

PCP 沿 prompt token 分摊 Prefill 计算，重点看长输入的 TTFT。DCP 沿历史 token 分片 KV Cache，重点看每卡 KV 占用、可承载并发及 Decode 吞吐，同时盯住每步 collective 带来的 TPOT 和 ITL 成本。

## 深入解释

- **PCP**：每个 rank 负责部分 query，attention 仍需要访问对应的历史 KV。跨 rank 数据交换、因果 attention 的负载不均、额外 workers 都要计入成本。适合长输入、低前缀命中、Prefill 计算主导 TTFT 的负载。
- **DCP**：各 rank 读本地 KV，跨卡合并局部输出和 softmax 统计。适合高并发长上下文，尤其是 TP 下 KV head 少、存在复制的 GQA/MQA/MLA 模型。短上下文或 Decode 卡在权重、MoE 通信时先验证收益。
- **追问**：只有一个 KV head 时，TP8 可能复制 KV。DCP 可把历史 token 切分给已有 ranks。PCP 和 DCP 可按阶段分别配置，也可按后端支持组合。

## 参考来源与待核实

- [[../../../../wiki/concepts/Prefill Context Parallel|Prefill Context Parallel]]、[[../../../../wiki/concepts/Decode Context Parallel|Decode Context Parallel]]。
- [vLLM Ascend Context Parallel](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/context_parallel.html)。后端支持与拓扑约束按部署版本核对。

## 所属题单

- [[../sets/推理并行、昇腾调优与 MLIR 面试追问|推理并行、昇腾调优与 MLIR 面试追问]]
- [[../README|秋招问题汇总]]
