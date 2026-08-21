---
type: source
source_kind: 文章
topic: GPU 编程
updated: 2026-08-21
---

# Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs

## 来源信息

- 标题：Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs
- 作者：Colfax Research
- 日期：2026-05-10
- 类型：文章 / GPU kernel 调度与实现分析
- 原始文件：[[../../raw/articles/Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs.md]]
- 原始链接：https://research.colfax-intl.com/dynamic-persistent-tile-scheduling-with-cluster-launch-control-clc-on-nvidia-blackwell-gpus/
- 主要参考：NVIDIA PTX ISA、CUDA Programming Guide 4.12、CUTLASS Blackwell CLC 文档与 CuTe DSL 示例

## 2-3 条核心摘要

- [[../concepts/Cluster Launch Control|Cluster Launch Control（CLC）]] 是 [[../entities/NVIDIA Blackwell]] 开始提供的硬件支持动态 persistent tile scheduling 机制：首批运行的 cluster 可以取消尚未启动的 cluster，并取得其 CTA 坐标来接手对应 work tile，从而把动态负载均衡与 persistent kernel 的初始化摊销、跨 tile 流水重叠结合起来。
- CLC 的 PTX 核心是 `clusterlaunchcontrol.try_cancel` 与 `clusterlaunchcontrol.query_cancel`：前者异步请求取消未启动 cluster 并写回 16 字节编码结果，后者先判断取消是否成功，再解码被取消 cluster 的首个 CTA 坐标。CuTe DSL 示例由每个 cluster 的单个 scheduler warp 发起请求，并通过 mbarrier 与 CLC pipeline 把 tile 信息同步给 TMA、MMA 和 epilogue warps。
- CLC 并非静态 persistent scheduling 的无条件替代。来源实验显示，它对 K 值差异明显的 grouped GEMM 能改善负载均衡；但在均衡 GEMM 上，CLC 与静态调度各有胜负，pipeline 深度、抢取时机、tile 顺序与 L2 局部性都可能改变结果。

## 值得关注的论断

- 与所有 cluster 竞争同一个 global atomic counter 的通用动态调度方案相比，CLC 不需要反复对单一全局计数器做原子操作，也不需要在每次 kernel launch 前清零该计数器。
- `try_cancel` 失败通常是正常调度信号，例如已经没有未启动 cluster，或需要为后启动的高优先级 kernel 让出资源；一旦 CTA 观察到失败，再次发起 `try_cancel` 属于未定义行为，应在完成当前工作队列后退出。
- CLC pipeline 加深可以预取多个 tile、隐藏调度延迟，但会让各 cluster 提前囤积任务，削弱动态均衡；极端情况下会逐渐接近静态 persistent scheduling。
- 来源在 B200 均衡 GEMM 实验中观察到，某些大 shape 下 CLC 的 L2 hit rate 低于静态调度，且总体性能更差；作者没有确认根因，因此不能把“负载更均衡”直接等价为“kernel 一定更快”。

## 调度策略对比

| 调度方式 | Grid 与任务分配 | 主要收益 | 主要代价 |
| --- | --- | --- | --- |
| Single-tile | 每个 work tile 对应一个 cluster | 硬件自然补充后续 wave，负载均衡较好 | 每个 tile 都承担初始化成本，难以跨 tile 重叠 epilogue 与 mainloop |
| Static persistent | 仅启动可并发驻留的 cluster，每个 cluster 按固定步长处理 tile | 摊薄初始化成本，支持跨 tile 流水重叠 | tile 开销不一致时容易负载不均；资源分配较僵化 |
| CLC dynamic persistent | 仍按完整问题 grid 启动，但活跃 cluster 取消未启动 cluster 并窃取坐标 | 动态均衡、保留 persistent 收益，并可为并发高优先级 kernel 让出资源 | 增加 scheduler warp、barrier/pipeline 开销；任务顺序和缓存局部性更难预测 |

## 实现约束

- 每次 `try_cancel` 只能由一个线程代表整个 cluster 提交；同一轮由多个线程提交会产生多个 cluster 取消请求。非平凡 cluster 通常由 leader CTA 的 scheduler warp 负责。
- `try_cancel` 的响应通过 shared memory 异步返回，固定为 16 字节，并用 transaction barrier 跟踪完成；若向整个 cluster multicast，发起时 cluster 内不能已有 CTA 退出。
- 必须先用 `query_cancel.is_canceled` 检查成功，再读取 CTA 坐标；取消失败时调用其他 `query_cancel` 形式属于未定义行为。
- `get_first_ctaid` 返回被取消 cluster 中首个 CTA 的 grid 坐标；cluster 内其他 CTA 需要加上自身 cluster-local 坐标，才能得到实际 work tile 坐标。
- CuTe DSL 示例使用单级 CLC pipeline；CUTLASS C++ kernel 可采用更深 pipeline，但预取深度需要在调度延迟与负载均衡之间调优。

## 关键概念

- [[../concepts/Cluster Launch Control]]
- [[../concepts/Tiling]]
- [[../concepts/Tail Effect]]
- [[../concepts/GPU执行模型]]
- [[../concepts/CuTe DSL]]
- [[../concepts/Occupancy]]

## 相关实体

- [[../entities/NVIDIA Blackwell]]
- [[../entities/Colfax Research]]

## 与现有 wiki 的关系

- 新建 [[../concepts/Cluster Launch Control]]，记录 single-tile、静态 persistent 与 CLC 动态 persistent 的机制边界。
- 更新 [[../concepts/Tiling]]、[[../concepts/Tail Effect]]、[[../concepts/GPU执行模型]] 和 [[../concepts/CuTe DSL]]，补充 work tile 调度、cluster-level 取消与 CuTe DSL pipeline 实现。
- 新建 [[../entities/NVIDIA Blackwell]] 与 [[../entities/Colfax Research]]。
- 未发现与现有 wiki 的直接冲突；该来源修正了“persistent 动态取活总会更快”的过度概括，并强调缓存局部性和预取深度可能抵消负载均衡收益。

## 待确认

- 均衡 GEMM 中 CLC 与静态调度出现 L2 hit rate 差异的根因，来源明确表示尚不清楚。
- 文中 benchmark 只适用于给定 B200、dtype、MMA tile、cluster shape、kernel 版本和测量设置，不能外推为所有 Blackwell workload 的固定结论。
- CLC pipeline 的最佳深度和首次 `try_cancel` 时机依赖 tile 耗时分布；variable-length attention、grouped GEMM 与规则 dense GEMM 需要分别测量。
- 来源提及 FlashAttention-4 的 CLC PR 作为 attention 案例线索，但当前资料没有完整展开其实现与 benchmark，后续需单独核实。
