# vLLM V1 统一调度器

## 定义

`vLLM V1` 的统一调度器是一种以 token budget 为中心的调度设计：调度器每一步用类似 `{request_id: num_tokens}` 的结构表示每个请求本轮要处理多少 token，而不是在调度决策层强行维护 prefill 与 decode 两套路径。

## 它解决什么问题

- 减少长 prompt prefill 对 decode 请求的阻塞，缓解 TTFT、TPOT 和吞吐之间的冲突
- 让 chunked prefill、prefix caching、speculative decoding 等能力更自然地组合
- 降低调度器、KV cache manager、worker、sampler 等核心组件之间的状态耦合
- 为多模态输入、异步 engine core 和后续 runner 重构留出更清晰的接口

## 核心机制

- 统一 token 视图：prompt tokens 和 generated tokens 在调度表示上都被看作本轮要处理的 tokens。
- 动态 token 分配：每个调度步根据全局 token budget、请求状态、KV cache 资源和调度策略，为请求分配 `num_tokens`。
- 自然支持 chunked prefill：长 prompt 可以被切成多个调度步，和 decode token 交错执行。
- 简化 feature 集成：prefix caching、speculative decoding 等功能可以围绕统一的 token scheduling decision 接入。
- 移除 v0 风格 swapping：v1 不再沿用把 KV cache 换出到 CPU 再换回 GPU 的路径，因此更依赖显存规划、请求准入和必要时的重计算/抢占策略。

## 常见误区

- 统一调度不等于 prefill/decode 在计算上完全一样。prefill 通常是大块 GEMM/attention，decode 常常是小 batch、强 KV cache 读的 memory-bound 路径；统一的是调度表示，而不是所有 kernel 的性能特征。
- `token quota` 不应理解成长期固定给某个请求的静态配额。更准确的说法是每个调度步动态决定每个请求本轮处理的 token 数。
- `vLLM v0` 不是完全不能混合 prefill/decode。开启 chunked prefill 后，v0 也可在 token budget 下交错部分 prefill 和 decode；但 v1 把这种统一抽象放进了核心架构。
- 多 GPU 服务不能只说“数据并行”。vLLM 同时涉及 replica 路由、tensor parallel、pipeline parallel、KV cache 分配和跨卡通信等不同层次。

## 调优关注点

- `max_num_batched_tokens`：每个调度步的 token budget，直接影响 chunked prefill、吞吐和延迟。
- `max_num_seqs`：控制并发序列数量，影响 decode batch 规模、KV cache 压力和 P99 延迟。
- `max_model_len` 与 `gpu_memory_utilization`：共同决定可用 KV cache 容量和长上下文可承载能力。
- scheduler policy / priority：用于在 FCFS、公平性、优先级和 SLA 之间做权衡，具体能力需按版本核实。

## 面试回答模板

可以这样回答：

> vLLM v0 的直觉是 prefill 和 decode 是两个阶段，默认调度会优先 prefill，长 prompt 容易阻塞 decode；虽然后续支持 chunked prefill，但整体上功能集成要适配两套阶段逻辑。vLLM v1 把调度决策抽象成每一步 `{request_id: num_tokens}`，prompt token 和 output token 都在同一个 token budget 下分配，因此 chunked prefill、prefix cache、speculative decoding 更容易组合。需要注意，统一调度只是 scheduler 表示统一，prefill/decode 的 kernel 特征仍不同；另外 v1 移除 v0 swapping 后，显存准入和 KV cache 容量规划更重要。

## 相关实体

- [[../entities/vLLM]]
- [[../entities/vLLM Team]]

## 相关来源

- [[../sources/vLLM v0 与 vLLM v1 调度架构差异截图整理]]
- [[../sources/Model Runner V2 A Modular and Faster Core for vLLM]]

## 相关概念

- [[Continuous Batching]]
- [[PagedAttention]]
- [[KV Cache]]
- [[Prefix Caching]]
- [[Speculative Decoding]]
- [[持久批处理]]
- [[Chunked Prefill]]

## 研究备注

- 后续可继续补 `vLLM v1 scheduler` 源码中的 waiting/running queue、KV cache allocation、preemption、encoder cache budget 和 async scheduler 细节。
- 若用于面试，应把 `vLLM v1`、`Model Runner V2`、`persistent batching` 三个概念分开：它们相关，但不是同一个层级的改动。
