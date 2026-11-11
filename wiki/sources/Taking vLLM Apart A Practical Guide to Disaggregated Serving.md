---
type: source
source_kind: 文章
topic: 推理服务
updated: 2026-11-10
---

# Taking vLLM Apart: A Practical Guide to Disaggregated Serving

## 来源信息

- 标题：Taking vLLM Apart: A Practical Guide to Disaggregated Serving
- 作者：Martin Hickey（IBM Research）
- 日期：2026-09-29（原文元数据）
- 类型：文章
- 原文链接：https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide
- 原始文件：[[../../raw/articles/Taking vLLM Apart A Practical Guide to Disaggregated Serving.md]]

## 2-3 条核心摘要

- vLLM 可分别拆分 Prefill/Decode 和 CPU frontend：前者通过 connector 交接 KV，后者以 render → token-only engine → derender 完成请求；两条切分轴可组合。
- PD 分离主要隔离长 Prefill 对 Decode 的干扰，提高同时满足 TTFT 和 ITL 目标的 goodput，而非保证峰值吞吐提升；传输、排队和额外服务会带来代价。
- 多轮双向 KV 传输可减少重算，但要求下一轮 prompt 与缓存一致；流式 derender 的 parser 重放可能显著增加 CPU 和网络开销。

## 值得关注的论断

- 原文双 L40S（无 NVLink、GPU P2P 不可用）、Qwen2.5-7B、约 8k 输入/256 输出、关闭 prefix caching 的实验中，0.4 req/s 时 PD 的 p99 ITL 为 29 ms、混部为 169 ms；0.2 req/s 时 PD 和混部 TTFT 中位数分别为 2.2 s 与 0.7 s。混部采用一个 DP=2 服务，作者指出两独立副本可能更好；结果不可泛化。
- 同一实验中约 470 MB KV 传输约 1.3 s；文章认为拷贝之外的开销（包括 decode 在 forward 间轮询完成状态）占了重要部分，不能单凭链路带宽推算实际传输延迟。
- 文章称特定 reasoning/tool parser 的流式 derender 在每个 chunk 重放历史、并可能反复扫描累计文本，长输出累计字符处理可达 O(n³)；该复杂度只适用于所述实现，不是所有 parser 的通性。
- 双向传输复用 Decode 缓存时，若下轮模板移除了先前 reasoning 内容，则 prompt 与缓存可能不对齐；文章称当时实现可能产生错误输出，关联 issue #43094，实际影响版本待核实。

## 关键概念

- [[../concepts/PD分离]]

## 相关实体

- [[../entities/vLLM]]

## 与现有 wiki 的关系

- 补充 [[../concepts/PD分离]] 的实际交接、传输开销、多轮复用和实验条件。
- 补充 [[../entities/vLLM]] 的 CPU frontend 与 token-only engine 切分。
- 未发现与现有 wiki 的直接冲突；实验结果强调不同 SLO 之间的权衡。

## 待确认

- 原文示例以 vLLM v0.31.0 或更高版本为前提；CLI、connector 配置、故障恢复策略与接口应按实际部署版本和源码验证。
- 文章提及的双向传输正确性问题、流式 parser 开销及相关修复状态可能随版本变化，尚未独立复核。
- 文中 frontend 和双向 KV 传输的示例及 benchmark 均未在本地运行。
