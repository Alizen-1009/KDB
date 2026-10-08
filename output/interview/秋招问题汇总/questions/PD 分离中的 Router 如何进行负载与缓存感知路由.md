# PD 分离中的 Router 如何进行负载与缓存感知路由？

- 整理状态：已整理
- 题目出处：用户提供的“千问 C 端 AI Infra 一面”面经

## 其他问法

- PD 分离中还有哪些先进的 Router 实现？
- Router 如何同时考虑队列负载、KV Cache 命中与传输成本？

## 问题背景

该题承接 PD 分离与 P:D 实例配比，需要区分请求入口路由、Prefill/Decode 阶段调度、缓存亲和性和跨实例 KV 迁移；具体实现与性能结论待绑定项目和版本。

## 30 秒回答

先为请求选择可用的 prefill/decode 资源池，再综合队列长度、可用 KV 容量、前缀缓存命中、KV 传输成本与目标 TTFT/ITL 分配实例。缓存命中可省 prefill，但若热点实例排队过长，负载均衡可能更快；需要定期更新状态并避免路由抖动。

## 深入解释

SGLang Model Gateway 的 cache-aware 策略是一个实现例子，利用前缀索引近似估算缓存命中，并结合负载阈值；具体 PD 跨池配对与版本细节须看部署配置。

## 参考来源与待核实

- [SGLang cache-aware 路由源码](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/policies/cache_aware.rs)

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
