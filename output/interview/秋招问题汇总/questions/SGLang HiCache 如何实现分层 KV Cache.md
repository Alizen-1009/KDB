# SGLang HiCache 如何实现分层 KV Cache？

- 整理状态：整理中
- 题目出处：用户提供的“千问 C 端 AI Infra 一面”面经

## 其他问法

- HiCache 是什么？它与普通 GPU KV Cache、RadixCache 有什么关系？

## 问题背景

面试中先追问 SGLang RadixCache 与 KV Cache，再继续追问 HiCache；后续整理答案时需要绑定具体 SGLang 版本，区分逻辑前缀索引、GPU/CPU 分层存储、传输调度和淘汰策略。

## 30 秒回答

HiCache 把 KV 分为 L1 GPU、L2 本机 CPU、L3 外部存储。HiRadixTree 记录前缀及本地 KV 的存储位置；请求先匹配 L1/L2，未命中的连续前缀可从 L3 预取，计算前把需要的 KV 恢复到 GPU。写回策略控制何时把数据下沉。

## 深入解释

L1/L2 通常是实例私有，跨实例复用依赖配置成共享的 L3 backend；不能说多机 CPU 内存自动组成一个共享 L2。预取是否等待、写回时机和后台传输要按部署配置权衡 TTFT、命中率与带宽。具体策略和代码细节随版本变化。

## 参考来源与待核实

- [[../../../../wiki/concepts/分层 KV Cache|分层 KV Cache]]
- [SGLang HiCache 设计文档](https://docs.sglang.io/docs/advanced_features/hicache_design)

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
