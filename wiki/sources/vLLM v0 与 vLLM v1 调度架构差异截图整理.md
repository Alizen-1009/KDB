# vLLM v0 与 vLLM v1 调度架构差异截图整理

## 来源信息

- 来源类型：用户截图转写 / 面试知识点整理
- 整理日期：2026-05-07
- 原始文件：[[../../raw/articles/vLLM v0 与 vLLM v1 调度架构差异截图整理]]
- 外部核对：vLLM V1 alpha release blog、vLLM V1 usage guide、vLLM optimization and tuning docs
- 关联实体：[[../entities/vLLM]]

## 2-3 条摘要

1. 截图的核心判断基本成立：`vLLM v0` 更偏阶段化调度，默认路径倾向先处理 prefill；`vLLM v1` 将 prompt tokens 与 output tokens 纳入统一 token-level scheduling decision，典型表示为 `{request_id: num_tokens}`。
2. `vLLM v1` 的统一调度器让 chunked prefill、prefix caching、speculative decoding 等功能更容易接入，因为调度器不再需要为每个功能分别适配 prefill/decode 两套路径。
3. 截图中也有几处需要更严谨：v0 在开启 chunked prefill 后也可混合调度；v1 的 “token quota” 不宜理解为长期固定配额；多 GPU 部分不能把 vLLM 简化成“主要只支持数据并行”。

## 值得关注的论断

- `vLLM v1` 的关键不是“彻底消灭 prefill/decode”，而是在调度决策层把两类 token 统一为每步要处理的 token 数。
- `vLLM v1` 移除 v0 的 CPU swapping 路径后，显存规划、KV cache 容量、请求准入和长上下文限流会更重要。
- 面试中可以把 `v0 -> v1` 的变化概括为：从“阶段中心的 scheduler”走向“token budget 中心的 scheduler”。

## 关联概念

- [[../concepts/vLLM V1 统一调度器]]
- [[../concepts/Continuous Batching]]
- [[../concepts/PagedAttention]]
- [[../concepts/KV Cache]]
- [[../concepts/Prefix Caching]]
- [[../concepts/Speculative Decoding]]

## 关联实体

- [[../entities/vLLM]]
- [[../entities/vLLM Team]]

## 待确认点

- `FCFS + 优先级调度`、chunked prefill 默认启用、swapping 移除等细节都和 vLLM 具体版本有关；引用时应带上版本或官方文档日期。
- 截图中的 `token_quota` 更像教学用变量名，公开配置项需按当前 vLLM CLI/API 文档核实。
- 多 GPU 场景需要拆开讨论服务层 replica 路由、tensor parallel、pipeline parallel 和跨节点执行，不能只用“数据并行”概括。

## 与现有 wiki 的关系

- 补足了 [[../entities/vLLM]] 条目中“v0/v1 调度架构差异”的空白。
- 与 [[../sources/Model Runner V2 A Modular and Faster Core for vLLM]] 互补：该来源讨论 vLLM 后续 runner/执行核心重构，本来源聚焦 v0/v1 scheduler 认知框架。
- 与 [[../concepts/Continuous Batching]]、[[../concepts/PagedAttention]]、[[../concepts/Speculative Decoding]] 形成面试回答链路。
