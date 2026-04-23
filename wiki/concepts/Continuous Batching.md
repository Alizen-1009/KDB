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

## 关键权衡

- 能显著改善真实流量下的吞吐与等待时间
- 调度器、状态管理和内存布局复杂度都会上升
- batch 越大并不意味着端到端延迟一定越低；在线服务里还要同时考虑 batching delay、queueing delay 和 decode 阶段的 memory-bound 特性

## 相关实体

- [[../entities/vLLM]]
- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 10 - Inference systems and optimization]]
- [[../sources/Model Runner V2 A Modular and Faster Core for vLLM]]

## 相关概念

- [[PagedAttention]]
- [[KV Cache]]
- [[持久批处理]]

## 研究备注

- 后续可补 Orca、vLLM 等系统在 iteration-level scheduling 上的具体实现差异
- 面试里一个高频追问是“为什么 continuous batching 不等于无脑做大 batch”：关键原因是吞吐提升和 P99 延迟之间常常存在冲突，需要设置 `max_batch_size` 与 `max_wait_time` 等约束
- `MRV2` 提醒了一个容易被忽略的点：光有 iteration-level scheduling 还不够，如果输入准备、采样和状态更新仍主要依赖 CPU，小模型或高端 GPU 场景下 host-side overhead 仍可能成为瓶颈
