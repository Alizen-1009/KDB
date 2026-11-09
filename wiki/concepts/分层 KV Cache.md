---
type: concept
topic: KV Cache
sources: 3
updated: 2026-06-12
---

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

## 方案对照与选型边界

[[../sources/KV Cache 研习社 · 深潜  LMCache、Mooncake、FlexKV——三条路线谁在改写 KV 的命运？]] 将 [[../entities/LMCache]] 的独立缓存层、[[../entities/Mooncake]] 的集群 KV 池、[[../entities/FlexKV]] 的多级卸载作为不同侧重点；这些能力有重叠，不是互斥类别。

| 方案 | 定位与侧重点 | 选型时需核实 |
| --- | --- | --- |
| SGLang HiCache | SGLang 体系内的分层缓存，名称不属于 vLLM | 具体版本的存储后端、模型状态和部署支持 |
| FlexKV | 文章描述为相对独立的多级 KV 管理组件，支持框架集成 | 对应框架 connector、版本、数据布局和生产案例 |
| vLLM KV offloading | vLLM 运行时的卸载路径，以及外部 KV connector 集成 | 原生与外部后端在目标版本中的具体能力 |
| LMCache / Mooncake | 前者侧重缓存管理与共享，后者侧重集群缓存和传输 | 与引擎、网络和模型布局的实际兼容性 |

以上是粗粒度机制归纳，并非功能完备性、工业采用规模或性能排名。HiCache 的具体实现未由该文章验证；FlexKV 的腾讯云生产数字属于文章转述，待查一手资料。优先根据已有引擎、缓存命中率、读取成本、硬件和运维能力评估，而不是仅依据项目名称。

## 减少缓存层级的替代取舍

[[../sources/谈谈 DeepInfer]] 将 DeepSeek 方案描述为削减本地持久缓存层级，以分布式存储承载可复用 KV，从而减少计算节点的缓存位置依赖。该描述来自技术评论，生产实现与开源代码范围待核实。

- 对照既有 RTP-LLM 来源：多级缓存强调容量与本地复用，本文强调多种 PCIe/RDMA 流量争用及调度复杂度；两者属于架构取舍，不是相互否定。
- LLM 归纳：减少本地层级会提高对远端存储有效带宽、尾延迟和可用性的要求，应同时核算命中收益、传输与排队成本。
- 原文“仅保留 G3.5”“计算节点无状态”不能理解为执行期不需要 HBM KV 或请求状态；G1–G4 是作者的层级命名，不是统一标准。

## 相关实体

- [[../entities/DeepInfer]]
- [[../entities/LMCache]]
- [[../entities/Mooncake]]
- [[../entities/FlexKV]]

- [[../entities/RTP-LLM]]
- [[../entities/vLLM]]
- [[../entities/SGLang]]

## 相关来源

- [[../sources/谈谈 DeepInfer]]
- [[../sources/KV Cache 研习社 · 深潜  LMCache、Mooncake、FlexKV——三条路线谁在改写 KV 的命运？]]
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
