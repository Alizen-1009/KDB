# PageAttention代码走读

## 来源信息

- 标题：PageAttention代码走读
- 作者：zzk againAbove & Beyond
- 日期：2023-11-25
- 类型：文章 / 代码走读
- 原始文件：[[../raw/articles/PageAttention代码走读|PageAttention代码走读]]
- 原始链接：[知乎专栏](https://zhuanlan.zhihu.com/p/668736097)

## 2-3 条核心摘要

- 文章从源码角度走读了 `vLLM` 的 `PageAttention / PagedAttention` CUDA kernel：先介绍 block table 管理分散 KV cache 的动机，再解释 kernel launch、输入 shape、thread group、vectorized load、softmax 和 `logits @ V_cache` 的实现。
- 文章强调 `PagedAttention` 的核心不是改变 attention 公式，而是把一个 request 的 CacheKV 切成多个 block，并通过 `block_tables` 找到每个 sequence 对应的 physical blocks，从而避免按 `max_seq_len` 为所有请求连续预分配 KV cache 带来的显存浪费。
- 代码走读中最有价值的细节是两类访问模式的差异：K cache 用 `[num_blocks, num_kv_heads, head_size/x, block_size, x]` 方便 QK 阶段按 token 读取 head_dim chunk；V cache 用 `[num_blocks, num_kv_heads, head_size, block_size]` 方便 PV 阶段固定 head_dim 后沿 token 维读取。

## 值得关注的论断

- `CUDA thread block` 和 `KV cache block` 是两种不同 block：前者是 GPU 调度单位，后者是 KV cache 中一段 token 槽位。
- 在该版本 kernel 中，grid 形如 `(num_heads, num_seqs)`，一个 CUDA thread block 负责一个 sequence 的一个 head；warp/thread group 再分摊 KV blocks、tokens 和 head_dim 上的计算。
- attention softmax 是对历史 token / context length 维度做的，而不是对 `head_size` 或 `hidden_size` 做的。
- `NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE` 描述单个 token 的 `q · k` 中每个 thread 负责多少 head_dim 元素，不需要乘 `NUM_TOKENS_PER_THREAD_GROUP`；后者对应外层 token loop。

## 关键概念

- [[PagedAttention]]
- [[KV Cache]]
- [[Continuous Batching]]
- [[vLLM V1 统一调度器]]
- [[CUDA Kernel]]
- [[Online Softmax]]
- [[Warp Shuffle Reduce]]
- [[Prefix Caching]]

## 相关实体

- [[../entities/vLLM]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`PagedAttention`、`KV Cache`
- 会更新哪些实体页：`vLLM`
- 是否存在冲突：无直接冲突；本次主要把既有的 PagedAttention 概念从“内存管理抽象”补到“decode kernel 代码走读”层面。

## 待确认

- 原文发布时间为 2023-11-25，具体函数名、kernel 参数、支持的 block size 和 layout 可能已随 `vLLM` 版本变化；后续若要精确引用，应补具体 commit 或新版源码链接。
- 原文标题使用 `PageAttention`，wiki 概念页统一使用更常见的 `PagedAttention`。
