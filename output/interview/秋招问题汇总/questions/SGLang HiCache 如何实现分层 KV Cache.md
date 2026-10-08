# SGLang HiCache 如何实现分层 KV Cache？

- 整理状态：整理中
- 题目出处：用户提供的“千问 C 端 AI Infra 一面”面经

## 其他问法

- HiCache 是什么？它与普通 GPU KV Cache、RadixCache 有什么关系？

## 问题背景

面试中先追问 SGLang RadixCache 与 KV Cache，再继续追问 HiCache；后续整理答案时需要绑定具体 SGLang 版本，区分逻辑前缀索引、GPU/CPU 分层存储、传输调度和淘汰策略。

## 30 秒回答

HiCache 将可复用 KV 的驻留范围从 GPU 扩展到较慢层级：热前缀留 GPU，较冷但仍有复用价值的块可迁到主机内存或外部后端；请求命中时按层级恢复，再交给 GPU attention。要把前缀索引、存储层位置和异步搬运调度区分开。

## 深入解释

具体层级、淘汰策略、pinning 与 backend 支持随 SGLang 版本变化；此页先保留机制级短答，待绑定源码版本验证。

## 参考来源与待核实

- [[../../../../wiki/concepts/分层 KV Cache|分层 KV Cache]]

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
