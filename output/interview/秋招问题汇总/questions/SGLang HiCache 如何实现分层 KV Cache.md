# SGLang HiCache 如何实现分层 KV Cache？

- 整理状态：待整理
- 题目出处：用户提供的“千问 C 端 AI Infra 一面”面经

## 其他问法

- HiCache 是什么？它与普通 GPU KV Cache、RadixCache 有什么关系？

## 问题背景

面试中先追问 SGLang RadixCache 与 KV Cache，再继续追问 HiCache；后续整理答案时需要绑定具体 SGLang 版本，区分逻辑前缀索引、GPU/CPU 分层存储、传输调度和淘汰策略。

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
