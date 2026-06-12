# 梦初AI Infra

## 一句话说明

AI Infra 技术文章作者 / 账号，当前知识库中主要作为 vLLM DCP 并行策略文章作者出现。

## 类型

- 人物 / 技术作者账号（待核实）

## 核心信息

- 在 `vllm并行策略之DCP(Decode Context Parallel)` 中，围绕 vLLM 的 decode context parallel 机制、使用方式、KV cache 分片和 prefill 兼容做了工程向梳理。
- 该来源使用 Helix Parallelism、vLLM RFC / PR 和 Flash Decoding 作为参照，强调 vLLM DCP 复用 TP group，与独立 CP group 设计存在差异。

## 相关概念

- [[../concepts/Decode Context Parallel]]
- [[../concepts/KV Cache]]
- [[../concepts/Tensor Parallelism]]
- [[../concepts/Flash Decoding]]

## 相关来源

- [[../sources/vllm并行策略之DCP(Decode Context Parallel)]]

## 冲突与备注

- 作者身份、机构归属和更多文章谱系尚未核实；当前仅按原始资料中的作者字段建页。
