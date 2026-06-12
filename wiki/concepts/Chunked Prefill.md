# Chunked Prefill

## 定义

`Chunked Prefill` 是把长 prompt 的 prefill 阶段切成多个较小 token chunk，并在调度层与 decode token 或其他请求交错执行的推理优化。

## 它解决什么问题

- 避免单个超长 prompt 的 prefill 长时间占用 GPU，造成 decode 请求 ITL / TPOT 抖动。
- 降低长上下文 prefill 的单步峰值显存、workspace 和调度阻塞风险。
- 在统一 token budget 调度中，把长 prefill 从“一个大任务”改成“多个可插队的小任务”，方便和 decode、prefix cache、speculative decoding 等能力组合。

## 核心机制

- 调度器给每个 step 一个 `max_num_batched_tokens` 之类的 token budget。
- Decode 请求通常优先进入本轮 batch。
- 如果还有 token budget，再调度 pending prefill tokens。
- 如果某个 prefill 请求放不进当前 budget，就只处理其中一个 chunk，剩余 prompt tokens 留到后续 step。
- 在 [[Decode Context Parallel]] 已启用的 vLLM 语境中，来源称 Chunked Prefill 需要配合 DCP 的 interleaved KVCache 布局写入缓存；也就是说，chunk 调度粒度和 DCP cache 分片是两个层级，前者切 prefill 调度步，后者切后续 decode 可读取的 KV token shard。

## 和 PD 分离的关系

- [[PD分离]] 把 prefill worker 和 decode worker 物理或逻辑隔离，主要解决两阶段资源争用和 decode tail latency。
- `Chunked Prefill` 在同池部署中最直接的价值是让长 prefill 不阻塞 decode；因此在严格 PD 分离后，这部分价值会减弱。
- 但它没有失效：prefill pool 内部仍可能有长短 prompt 混部、P99 TTFT、公平性、峰值显存和 KV 传输流水化问题，chunking 仍可作为 prefill 侧调度粒度控制。
- 真实系统常用条件路由，不是所有请求都走完整 PD 分离；短 prompt、cache hit 或轻量请求可能仍在本地 mixed engine 中处理，此时 chunked prefill 仍然有价值。

## 关键权衡

- chunk 太大：更接近普通 prefill，可能继续阻塞 decode 或其他短 prefill。
- chunk 太小：调度次数、kernel launch、KV 管理和中间状态维护开销上升，也可能降低 prefill 本身的吞吐。
- PD 分离更像架构级隔离，chunked prefill 更像调度粒度控制；二者可以替代一部分场景，也可以组合。

## 面试回答模板

> PD 分离不会让 Chunked Prefill 完全没用。它们解决的问题层级不同：PD 分离是把 prefill 和 decode 放到不同资源池，减少两阶段互相干扰；Chunked Prefill 是把一个长 prefill 切成多个可调度的小块，控制单次 prefill 对 GPU、显存和队列的占用。在严格 PD 分离且 prefill 资源很充足时，Chunked Prefill 对 decode ITL 的价值会下降；但它在 prefill pool 内部仍能改善长短 prompt 公平性、P99 TTFT、峰值显存和调度粒度。真实系统里还常做条件路由，不是所有请求都走远端 prefill，所以 Chunked Prefill 仍是有用的补充，而不是被 PD 分离淘汰。

## 相关实体

- [[../entities/vLLM]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/vLLM v0 与 vLLM v1 调度架构差异截图整理]]
- [[../sources/量化剪枝推理瓶颈Nsight与异构集群面试整理]]
- [[../sources/vllm并行策略之DCP(Decode Context Parallel)]]

## 相关概念

- [[PD分离]]
- [[Continuous Batching]]
- [[vLLM V1 统一调度器]]
- [[KV Cache]]
- [[Prefix Caching]]
- [[Decode Context Parallel]]

## 研究备注

- vLLM 官方文档将 chunked prefill 描述为把 large prefills 拆小，并与 decode requests batch 在一起；V1 中在可能时默认启用。
- vLLM disaggregated prefilling 文档也指出，合适 chunk size 的 chunked prefill 可以控制 tail ITL，但实践中 chunk size 难调，disaggregated prefill 更可靠地控制 tail ITL。
- DCP 来源称它兼容 Chunked Prefill，但这是 backend 和版本相关实现能力；引用时需要核实具体 vLLM 版本、attention backend 与 cache manager 行为。
