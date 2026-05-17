# Engram

## 一句话说明

DeepSeek 在 `Conditional Memory via Scalable Lookup` 中提出的记忆模块，用确定性 `N-gram` 查表为语言模型提供与 MoE 互补的条件记忆能力。

## 类型

- 项目 / 模块 / 论文实现

## 核心信息

- 它的目标是把“静态模式存储”从主干神经计算里剥离出来，让模型不必总靠早期 attention 和 FFN 去重建局部、固定、可查表的模式。
- 技术路径是：对局部后缀 `N-gram` 做 tokenizer compression、多头哈希检索、context-aware gating 和轻量卷积融合，再把结果残差注入主干。
- 它的大规模静态记忆表可以类比为“key 不是单 token，而是压缩后局部 `N-gram` 的超大 embedding table”：表项本身是可训练参数，训练时只有当前 batch 命中的 memory rows 被反传更新。
- 推理时的查找不是搜索或 ANN，而是 `N-gram -> hash index -> memory_table[index]` 的数组索引式访问；因为只计算固定数量的哈希地址，所以单次 lookup 是 `O(1)`。
- 论文强调它不仅是建模模块，也是系统友好的模块：训练时适合做大表分片与 `All-to-All` 行检索，推理时适合做 host-memory prefetch 与多级缓存。
- 官方 GitHub 仓库当前更像核心数据流演示与论文配套实现，而不是完整生产级训练栈。
- 当前 `engram_demo_v1.py` 这份 demo 进一步说明了几个关键实现落点：词表压缩先于 hash；不同层有各自的 hash multipliers 与质数表大小；检索后的记忆通过 RMSNorm 后的 query/key 计算标量 gate，再经短卷积扩展局部感受野。

## 相关概念

- [[Conditional Memory]]
- [[Sparsity Allocation]]

## 相关来源

- [[../sources/Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models]]
- [[../sources/engram_demo_v1]]

## 冲突与备注

- 需要明确区分：`Engram` 不是外部检索式 RAG，也不是服务层缓存；它是模型内部参数化静态记忆模块
- 也不要把它等同于普通 token embedding：普通 embedding 的 key 是单个 token id，Engram 的 key 是局部上下文 `N-gram` 经压缩与哈希后的 memory slot；代价是需要处理哈希碰撞、大表分片和热行访问。
