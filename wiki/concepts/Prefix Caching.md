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

## 关键权衡

- 命中率高度依赖 prompt 结构
- 只能复用公共前缀，到第一个不相同 token 就停止

## 相关实体

- [[../entities/vLLM]]
- [[../entities/SGLang]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]

## 相关概念

- [[KV Cache]]
- [[缓存感知路由]]

## 研究备注

- 非前缀缓存仍是研究热点，后续可补 `CacheBlend`、`LMCache` 等实现
