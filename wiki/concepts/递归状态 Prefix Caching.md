---
type: concept
topic: KV Cache
sources: 1
updated: 2026-07-25
---

# 递归状态 Prefix Caching

## 定义

递归状态 Prefix Caching 是为 GDN、KDA、Mamba 类固定状态模型保存特定 token 边界 checkpoint，使后续共享前缀请求能够恢复该边界的 [[线性注意力递归状态]]，而不必从序列开头重新递推。

它与普通 KV Prefix Cache 的核心差异是：KV Cache 保存彼此独立的 token/page 历史行，递归状态只保存“处理到当前边界后的聚合结果”。

## 为什么普通 Prefix Cache 不够

若缓存请求已处理 10 个 token，而新请求只共享前 7 个 token：

- MLA/full attention 可以引用前 7 个 token 对应的 KV 行；
- 若 KDA/GDN 只保存了第 10 步的状态 `S_10`，通常无法从它逆推出 `S_7`；
- 第 7 步继续执行还需要该边界的 Conv State，因为下一 token 的短卷积依赖最近窗口。

因此正确恢复点需要：

```text
checkpoint at boundary t
├── 每个递归线性层的 Conv State_t
└── 每个递归线性层的 Matrix State_t
```

checkpoint 是整个状态槽在该边界的快照，不是某一层的孤立矩阵。

## SGLang 的双账本管理

来源描述的混合 MLA/KDA 模型同时使用：

- `Token KV Pool`：按 token/page 保存 MLA 或 full attention 的 KV；
- `MambaPool`：按请求槽保存各 KDA/GDN 层的 Conv State 与递归矩阵状态；
- `UnifiedRadixCache`：在同一逻辑 Radix Tree 上协调 Full/MLA KV component 与 Mamba State component 的匹配、引用、锁、LRU 和驱逐。

“统一”指逻辑前缀树和生命周期协调，不表示两种状态被拼进同一个物理大向量。

## 命中与回退

当 token KV 命中到更远位置，但中间没有对应递归状态 checkpoint 时，统一可复用长度必须回退到最近一个同时拥有两类状态的边界：

```text
KV 命中到 token 10
递归 checkpoint 只到 token 8
=> 从 token 8 恢复状态
=> token 9..10 重新 prefill
```

若命中边界有 checkpoint，来源描述 SGLang 通过 copy-on-write 把快照恢复到新请求的活动状态槽，避免原地修改缓存快照。

## 显存与重算交换

- checkpoint 密：可在更细粒度边界恢复，重算少，但每个快照要保存所有相关层的 Conv State 和矩阵状态，显存高。
- checkpoint 稀：状态显存低，但 Prefix Cache 命中后要从最近 checkpoint 重算更长区间。
- 因而固定递归状态虽然消除了 KV 随上下文线性增长，却没有免费获得任意 token 边界复用能力。

## 与 speculative decoding

Draft token 会暂时推进递归状态。若 rejected token 已写入主状态，普通 KV slot 删除不足以恢复，因为矩阵更新通常不可逆。来源描述的 SGLang 路径先把 draft 对应状态写入暂存区，target 验证后再按实际接受长度提交正确状态。

机制上必须保证：

```text
主状态只反映已接受 prefix
rejected draft 不得污染 Conv State 与 Matrix State
```

精确的暂存布局、提交方式和 kernel 名称依赖版本，需按源码核实。

## 关键权衡

- 比从头重算共享前缀更快，但 checkpoint 占用可能显著，尤其矩阵状态按请求、层和 head 存储。
- 命中率不再只取决于 token 前缀，还取决于递归 checkpoint 是否存在且与 KV 边界对齐。
- KV 和递归状态应具有一致的引用、锁定与驱逐语义，否则可能出现一边仍可复用、另一边已释放的悬空命中。
- 模型、TP 布局或状态格式变化后，checkpoint 通常不可直接复用。

## 相关实体

- [[../entities/SGLang]]

## 相关来源

- [[../sources/SGLang的KDA管理与Prefix Cache难题]]

## 相关概念

- [[Prefix Caching]]
- [[RadixAttention]]
- [[KV Cache]]
- [[线性注意力递归状态]]
- [[Speculative Decoding]]
- [[Chunked Gated Delta Rule]]

## 研究备注

- `UnifiedRadixCache`、checkpoint 抽取频率、extra-buffer tracking 和 copy-on-write 属于版本相关实现细节，需绑定 SGLang commit。
- 可进一步量化 checkpoint interval、状态矩阵大小、命中分布与 recompute token 数之间的容量/TTFT 权衡。
