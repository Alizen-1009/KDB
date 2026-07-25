---
type: entity
entity_type: 项目
topic: 推理服务
sources: 1
updated: 2026-07-25
---

# vLLM AFD Plugin

## 一句话说明

`vLLM AFD Plugin` 是 vLLM 的实验性外部插件，为 MoE 推理提供 Attention-FFN Disaggregation，让 Attention 与 FFN/MoE 在独立服务和 rank 拓扑上执行。

## 类型

- 项目 / vLLM 外部插件

## 核心信息

- 通过 `vllm.general_plugins` 入口点和 `--additional-config` 接入，不要求修改 vLLM 源码树。
- Attention worker 保留调度、KV Cache、批处理、模型生命周期与采样；FFN worker 通过连接器接收 hidden states 和执行元数据，执行 `compute_ffn_output()` 后返回结果。
- 支持 NVIDIA GPU 与昇腾 NPU 后端，文章列出的连接器包括同步 `P2pNcclAFDConnector`、`CAMP2pAFDConnector` 和异步 `CAMAsyncAFDConnector`。
- 已注册 DeepSeek V2/V3 系列（含 DeepSeek V3.2）以及 GLM MoE DSA 的模型封装；同步路径面向 decode，CAM async 路径面向 PD 分离中的 prefill。
- 文章发布时目标 vLLM 版本为 `0.19.1`，仅支持 model runner v1；两种角色都加载完整权重，graph 路径仅限 decode，DBO 限定为两个 ubatch。

## 相关概念

- [[Attention-FFN 分离]]
- [[MoE]]
- [[Expert Parallelism]]
- [[Tensor Parallelism]]
- [[流水线并行]]
- [[PD分离]]

## 相关来源

- [[../sources/vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署]]

## 冲突与备注

- 项目明确处于实验阶段，需在完整模型、自然专家路由、真实在线负载和更多硬件拓扑上补充精度、延迟、吞吐与稳定性验证。
- 来源中的 decode 与 prefill benchmark 均有严格实验边界，不应作为任意 AFD 部署的通用性能承诺。
- 具体 vLLM 版本、model runner、连接器、graph 和并行拓扑支持范围可能快速变化，使用时应以对应 commit 的 README 与 recipe 为准。
