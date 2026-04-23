# Dual RoPE

## 定义

针对不同层类型使用不同 [[RoPE]] 频率配置与旋转比例的设计，用于兼顾局部建模和超长上下文位置编码稳定性。

## 它解决什么问题

- 提升长上下文下的位置表示质量
- 避免单一 RoPE 配置同时承担局部层和全局层的全部需求

## 核心机制

- sliding attention 层使用标准 RoPE 配置
- full attention 层使用更长周期、更低 rotary 比例的配置
- 按层类型切换不同的 `rope_theta` 和 `partial_rotary_factor`

## 关键权衡

- 能更灵活地匹配不同层职责
- 增加了位置编码实现与调参复杂度

## 相关实体

- [[../entities/Gemma 4]]

## 相关来源

- [[../sources/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构]]

## 相关概念

- [[RoPE]]
- [[混合注意力]]

## 研究备注

- 它可以理解成“按层类型定制的 RoPE”，而不是一种与 RoPE 平行的新位置编码家族
- 后续可和长上下文 scaling、YaRN、NTK-aware RoPE 等方案做对比
