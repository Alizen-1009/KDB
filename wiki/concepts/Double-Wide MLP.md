---
type: concept
topic: 模型架构
sources: 1
updated: 2026-04-23
---

# Double-Wide MLP

## 定义

通过扩大某些层 MLP 中间维度来补偿模型表达能力的一种结构设计。

## 它解决什么问题

- 在其它结构压缩或共享机制引入后，补偿表达能力损失
- 为特定层提供更强的非线性变换容量

## 核心机制

- 将目标层的 intermediate size 扩大
- 保持层的其余主结构不变
- 把更多参数预算投入到 MLP 变换能力上

## 关键权衡

- 表达能力增强，但参数量和部分计算成本会增加
- 是否值得采用，取决于它是否能有效补偿别处的结构约束

## 相关实体

- [[../entities/Gemma 4]]

## 相关来源

- [[../sources/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构]]

## 相关概念

- [[Per-Layer Embeddings]]
- [[Shared KV Cache]]

## 研究备注

- 文章将其解释为共享 KV 层的补偿机制，这一点值得后续用官方实现进一步验证
