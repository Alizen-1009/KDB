---
type: concept
topic: KV Cache
sources: 5
updated: 2026-06-12
---

# Prefix Caching

## 定义

在多个请求共享相同前缀时，直接复用该前缀已经计算出的 KV Cache，以跳过重复 prefill。

## 它解决什么问题

- 降低共享 prompt 的重复计算成本
- 缩短 TTFT，提升缓存命中场景下的吞吐和单位成本效率

## 核心机制

- 以序列起点开始匹配请求前缀
- 命中部分直接复用已有 KV
- 只对第一个新 token 之后的内容继续做 prefill 或 decode
- 在 `SGLang` 中，[[RadixAttention]] 用 radix tree 系统化保存 prompt 与生成结果的 KV cache，使 program 分支和共享系统 prompt 也能自动做前缀复用
- 在 [[../entities/RTP-LLM]] 中，Master 使用统一哈希映射聚合 worker 缓存键，做跨 worker 前缀匹配，并把匹配结果反馈给调度器
- 在 [[Decode Context Parallel]] 已启用时，prefix cache 命中的 KV 还需要符合 DCP 的分布式 cache layout；`vllm并行策略之DCP` 称 DCP 兼容 Prefix Cache，但该能力依赖具体 vLLM backend 和 cache manager 实现。
- 对 KDA/GDN 等递归线性注意力，token KV 命中不足以独立恢复执行：还需要对应边界的 Conv State 与矩阵状态 checkpoint。若 KV 命中更远但递归 checkpoint 较近，统一复用长度必须回退并重新 prefill 中间区间，详见 [[递归状态 Prefix Caching]]。

## 关键权衡

- 命中率高度依赖 prompt 结构
- 只能复用公共前缀，到第一个不相同 token 就停止
- 运行时缓存结构越复杂，越需要配合驱逐策略和缓存感知调度，否则显存占用可能吞掉收益
- 递归状态 checkpoint 引入额外权衡：快照越密重算越少，但需要保存所有相关层的 Conv State 和矩阵状态；快照越稀则 Prefix Cache 的实际可恢复边界越粗。

## 相关实体

- [[../entities/vLLM]]
- [[../entities/SGLang]]
- [[../entities/RTP-LLM]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/SGLang：LLM推理引擎发展新方向]]
- [[../sources/RTP-LLM]]
- [[../sources/vllm并行策略之DCP(Decode Context Parallel)]]
- [[../sources/SGLang的KDA管理与Prefix Cache难题]]

## 相关概念

- [[KV Cache]]
- [[RadixAttention]]
- [[缓存感知路由]]
- [[分层 KV Cache]]
- [[Decode Context Parallel]]
- [[递归状态 Prefix Caching]]
- [[线性注意力递归状态]]

## 研究备注

- 非前缀缓存仍是研究热点，后续可补 `CacheBlend`、`LMCache` 等实现
- 需要区分通用 `Prefix Caching` 概念与 `RadixAttention` 这类具体 runtime 组织方式
- RTP-LLM 的统一哈希映射属于系统级实现路线；缓存命中收益仍依赖 prompt 结构、hash 粒度、worker 负载和远程缓存读取成本。
