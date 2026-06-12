# 分层 KV Cache

## 定义

`分层 KV Cache` 是把推理运行时的 KV cache 从单一 GPU 显存扩展到多级存储层的系统设计，例如 GPU、本地 CPU、远程 CPU 和分布式存储。

## 它解决什么问题

- 单卡或单节点 GPU 显存无法长期保留所有可复用 KV cache
- 普通负载均衡容易让跨请求前缀复用失效
- 长上下文、多轮对话和生产高并发场景下，KV cache 既是性能资产，也是显存压力来源

## 核心机制

- 使用 token block hash 或前缀 hash 标识可复用 KV block。
- 调度器或 Master 维护全局缓存视图，把 hash key 映射到缓存块位置、worker 元数据和版本信息。
- 推理节点按从快到慢的顺序查找缓存：GPU memory、本地 CPU memory、远程 CPU memory、分布式存储。
- 缓存状态可以低频增量同步，而 worker 负载可以高频查询，避免每次调度都传输完整缓存键集合。
- 调度决策不只看负载，也看候选 worker 的缓存命中长度、缓存传输成本和准入容量。

## 关键权衡

- 分层缓存扩大了可复用 KV 的生命周期和范围，但引入了元数据管理、跨节点传输和一致性复杂度。
- 更高缓存命中率不一定自动降低端到端延迟；如果远程缓存读取或 KV 传输成本过高，收益可能被网络和调度开销抵消。
- 它与 [[PagedAttention]]、[[RadixAttention]]、[[Prefix Caching]] 不是同一层概念：PagedAttention 更偏单机分页式 KV 管理，RadixAttention 更偏 runtime 前缀树复用，分层 KV Cache 更强调跨节点和跨存储层级的系统缓存管理。

## 相关实体

- [[../entities/RTP-LLM]]
- [[../entities/vLLM]]
- [[../entities/SGLang]]

## 相关来源

- [[../sources/RTP-LLM]]

## 相关概念

- [[KV Cache]]
- [[Prefix Caching]]
- [[缓存感知路由]]
- [[PD分离]]
- [[PagedAttention]]
- [[RadixAttention]]

## 研究备注

- RTP-LLM 来源中提到的块大小、采样前缀哈希阈值、缓存同步周期和缓存评分公式都需要按论文或源码复核。
- 后续可以把 `GPU/CPU/offload/distributed storage` 的延迟、带宽和容量账本补成一张对比表，用来判断什么时候值得远程复用而不是重新 prefill。
