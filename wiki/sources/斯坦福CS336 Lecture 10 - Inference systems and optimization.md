# 斯坦福CS336 Lecture 10 - Inference systems and optimization

## 来源信息

- 标题：斯坦福CS336 Lecture 10 - Inference systems and optimization
- 作者：[[../entities/Stanford CS336]]
- 日期：2025 Spring
- 类型：可执行课程讲稿
- 原始文件：[[../raw/articles/斯坦福CS336 Lecture 10 - Inference systems and optimization|斯坦福CS336 Lecture 10 - Inference systems and optimization]]

## 2-3 条核心摘要

- 这讲把 LLM 推理拆成了非常清楚的系统账本：`prefill` 更接近算力问题，`generation` 更接近内存带宽问题，因此推理优化和训练优化的主瓶颈并不一样。
- Stanford 把很多看似分散的 inference trick 统一到了一个主轴上：围绕 KV cache、内存带宽和动态 workload 管理来理解它们，而不是把它们当成孤立技巧。
- 这讲最值得保留的工程直觉是：推理系统不是只比模型精度，还要同时管理 `TTFT / latency / throughput / memory fragmentation / batching` 这些相互冲突的指标。

## 值得关注的论断

- generation 阶段的 attention 几乎天然 memory-bound，因此仅靠 batch 增大并不能像训练那样把 attention 轻松推到 compute-bound。
- KV cache 是推理效率的基础设施，但也会反过来成为内存和调度复杂性的主要来源。
- Continuous batching、PagedAttention 和 speculative decoding 代表了三类非常典型的系统优化思路：调度、内存管理和检查-生成分离。

## 关键概念

- [[KV Cache]]
- [[PagedAttention]]
- [[Continuous Batching]]
- [[Speculative Decoding]]

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/vLLM]]
- [[../entities/TensorRT-LLM]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`KV Cache`、`PagedAttention`、`Continuous Batching`、`Speculative Decoding`
- 会更新哪些实体页：`Stanford CS336`、`vLLM`
- 是否存在冲突：与现有 wiki 无直接冲突，但会把已有推理概念页从“静态机制说明”推进到“线上动态 workload”视角

## 待确认

- 后续可把这一讲涉及的 GQA / MLA / CLA / local attention、quantization、pruning 分拆挂到更系统的推理优化专题里
