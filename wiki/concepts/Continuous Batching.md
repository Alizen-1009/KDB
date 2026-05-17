# Continuous Batching

## 定义

一种面向在线推理请求流的调度策略，按 decode iteration 动态重组 batch，让新请求无需等待旧请求整段生成结束就能进入执行队列。

## 它解决什么问题

- 缓解静态 batch 只适合“同一时刻到达、同样长度请求”的限制
- 在保持较好延迟的同时提高 decode 阶段的整体硬件利用率

## 核心机制

- 把生成过程拆成一步一步的 iteration-level scheduling
- 每次 decode step 后都允许新请求加入、旧请求退出
- 对 attention 和非 attention 计算采用不同 batching 策略，以适应不同长度的 ragged requests
- 在 `vLLM` 的 `MRV2` 语境里，`Continuous Batching` 不再只是调度器策略，还和 `持久批处理`、GPU-side input preparation、async scheduling 紧密耦合，目的是减少每一步的 host-side bookkeeping 和 CPU/GPU 同步
- 在 `vLLM V1` 语境里，continuous batching 可以和统一 token-level scheduler 放在一起理解：调度器每步决定每个请求处理多少 token，使长 prompt 的 chunked prefill 可以和 decode token 更自然地交错执行

## 关键权衡

- 能显著改善真实流量下的吞吐与等待时间
- 调度器、状态管理和内存布局复杂度都会上升
- batch 越大并不意味着端到端延迟一定越低；在线服务里还要同时考虑 batching delay、queueing delay 和 decode 阶段的 memory-bound 特性
- 它还会让 batch 组成在相邻 step 间持续变化，这既是效率来源，也可能成为推理不可复现的来源之一，因为 kernel 的 tiling、规约顺序和算法选择可能随 batch 形态变化

## 相关实体

- [[../entities/vLLM]]
- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 10 - Inference systems and optimization]]
- [[../sources/Model Runner V2 A Modular and Faster Core for vLLM]]
- [[../sources/推理的非确定性运算及vLLMSGLang控制方式]]
- [[../sources/多卡GPU监控与SM执行模型面试整理]]
- [[../sources/vLLM v0 与 vLLM v1 调度架构差异截图整理]]

## 相关概念

- [[PagedAttention]]
- [[KV Cache]]
- [[持久批处理]]
- [[确定性推理]]
- [[vLLM V1 统一调度器]]
- [[Chunked Prefill]]

## 研究备注

- 后续可补 Orca、vLLM 等系统在 iteration-level scheduling 上的具体实现差异
- 面试里一个高频追问是“为什么 continuous batching 不等于无脑做大 batch”：关键原因是吞吐提升和 P99 延迟之间常常存在冲突，需要设置 `max_batch_size` 与 `max_wait_time` 等约束
- `MRV2` 提醒了一个容易被忽略的点：光有 iteration-level scheduling 还不够，如果输入准备、采样和状态更新仍主要依赖 CPU，小模型或高端 GPU 场景下 host-side overhead 仍可能成为瓶颈
- 新来源补充了一个反直觉但重要的视角：`Continuous Batching` 不只是吞吐优化，它也可能改变同一请求所处的 batch 上下文，从而触发不同数值路径
- 从 SM 利用角度看，continuous batching 的价值之一是缓解 decode 单步 batch 太小导致的 SM Active/Occupancy 不足，但它仍要和 KV cache 布局、P99 延迟和调度开销一起权衡。
- 面试里可以把 vLLM v1 的统一调度器理解成 continuous batching 的进一步架构化：它不只是在 decode iteration 之间插入请求，而是把每个调度步要处理的 prompt/output token 数变成统一决策。
