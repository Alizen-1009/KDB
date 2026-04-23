# LLM推理优化核心技术

## 来源信息

- 标题：LLM推理优化核心技术
- 作者：[[kason_zhang]]
- 日期：2026-04-12
- 类型：文章
- 原始文件：[[../raw/articles/LLM推理优化核心技术|LLM推理优化核心技术]]

## 2-3 条核心摘要

- 文章把生产级 LLM 推理优化组织成一条系统链路：`KV Cache / Prefix Caching` 打底，`FlashAttention / PagedAttention / Chunked Prefill` 解决长上下文处理，再往上才是 `TP / PP / EP` 和 `Prefill-Decode Disaggregation` 这类更重的系统优化。
- `Prefix Caching` 的收益不只取决于引擎实现，还强依赖 prompt 结构和缓存感知路由；稳定前缀前置、用户变化内容后置，是提升缓存命中率的直接工程手段。
- `PD 分离` 被文章明确定位成高投入架构优化，不是通用默认选项。只有在超大模型、超大流量、且 prefill-heavy 的场景下，才值得用架构复杂度换性能收益。

## 值得关注的论断

- `KV Cache` 不是可选优化，而是生产推理的基础设施。
- `Cache-aware routing` 是多副本环境中把 Prefix Caching 价值吃满的必要条件。
- `Tensor Parallelism` 主要服务低时延，`Expert Parallelism` 主要服务系统总吞吐，二者在 MoE 系统中是互补关系。
- `Conditional Aggregation` 比全量 PD 分离更符合真实流量形态。

## 关键概念

- [[KV Cache]]
- [[Prefix Caching]]
- [[缓存感知路由]]
- [[PagedAttention]]
- [[PD分离]]
- [[Tensor Parallelism]]

## 相关实体

- [[../entities/vLLM]]
- [[../entities/SGLang]]
- [[../entities/Nvidia Dynamo]]
- [[../entities/TensorRT-LLM]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`KV Cache`、`Prefix Caching`、`缓存感知路由`、`PagedAttention`、`PD分离`、`Tensor Parallelism`
- 会更新哪些实体页：`vLLM`、`SGLang`、`Nvidia Dynamo`、`TensorRT-LLM`
- 是否存在冲突：暂无直接冲突，但 `PD分离` 的适用边界需后续和更多生产案例交叉验证

## 待确认

- 文章中的部分实现细节和参数判断明显偏经验总结，后续需要结合官方文档或论文做二次校验
