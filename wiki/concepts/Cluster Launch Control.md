---
type: concept
topic: GPU 编程
sources: 1
updated: 2026-08-21
---

# Cluster Launch Control

## 定义

`Cluster Launch Control（CLC）` 是 [[../entities/NVIDIA Blackwell]] 开始提供的硬件支持动态 persistent tile scheduling 机制。已运行的 thread block cluster 可以取消尚未启动的 cluster，并取得后者的 CTA 坐标，从而接手对应 work tile。

## 它解决什么问题

- 静态 persistent scheduler 按固定规则给 cluster 分配 tile；grouped GEMM、变长 attention 等 workload 的 tile 成本不一致时，部分 cluster 会提前空闲，形成负载不均与 [[Tail Effect]]。
- 使用全局 atomic counter 的通用动态任务池需要所有 cluster 竞争同一个计数器、反复访问 global memory，并在 kernel launch 前初始化计数器。
- single-tile scheduler 虽能由硬件不断补充后续 cluster，但每个 tile 都要承担 pipeline、descriptor 等初始化成本，也难以让相邻 tile 的 mainloop 与 epilogue 重叠。

## 核心机制

1. Kernel 按完整问题的 work-tile grid 启动，而不是只启动一波 persistent cluster。
2. 首批活跃 cluster 先处理各自初始 tile；每个 cluster 由一个 scheduler thread 发出 `clusterlaunchcontrol.try_cancel`，尝试原子取消一个尚未启动的 cluster。
3. 硬件将 16 字节编码响应异步写入 shared memory，并通过 transaction barrier 通知完成。
4. 消费者先用 `clusterlaunchcontrol.query_cancel.is_canceled` 判断是否成功；成功后再用 `get_first_ctaid` 解码被取消 cluster 的首个 CTA 坐标，把它转换为新的 work tile 坐标。
5. 活跃 cluster 循环执行“处理 tile → 取消并接手未启动 cluster”。因此首波 cluster 可能持续到完成全部任务，而被取消的 cluster 从未真正启动。
6. 若取消失败，cluster 完成当前已取得的任务后退出；尚未执行的 grid 仍可在资源恢复后由后续 cluster 继续完成。

## 实现约束

- 一个 `try_cancel` 只能由一个线程代表整个 cluster 发起；多个线程调用会产生多个取消请求。
- 必须先检查 `is_canceled`。失败后继续解码响应，或由同一 CTA 再次发起 `try_cancel`，都属于未定义行为。
- 向整个 cluster multicast 响应时，发起操作前 cluster 内不能已有 CTA 退出。
- 返回坐标是被取消 cluster 的首个 CTA grid 坐标；其他 CTA 需叠加自身 cluster-local 坐标。
- CLC 只改变任务调度，不改变 GEMM、attention 等算子的数学定义。

## 与其他调度方式的关系

| 方式 | 任务取得方式 | 负载均衡 | 持久化收益 | 主要风险 |
| --- | --- | --- | --- | --- |
| Single-tile | 每个 cluster 固定一个 tile | 硬件动态补 wave | 弱 | 初始化成本高、跨 tile 重叠不足 |
| Static persistent | 一波 cluster 按静态步长遍历 tile | tile 成本不同时较差 | 强 | straggler 与尾部空闲 |
| Global atomic queue | cluster 原子领取 tile index | 强 | 强 | 单一 counter 竞争与 global-memory 往返 |
| CLC | 活跃 cluster 取消未启动 cluster 并继承坐标 | 强 | 强 | CLC pipeline、barrier 与缓存局部性需要调优 |

## Pipeline 深度与抢取时机

- 多级 CLC pipeline 能提前排队多个 work tile，隐藏调度延迟，适合单 tile 很短甚至为空的场景。
- pipeline 越深，cluster 越容易提前囤积不等量工作，动态调度会逐渐接近静态分配。
- 对少 wave 且 tile 成本差异很大的问题，即使只有一级 pipeline，也可能需要把首次 `try_cancel` 延后到当前 MMA mainloop 接近完成，以免高成本 cluster 又提前抢到高成本 tile。

## 并发 kernel 与资源让渡

较高优先级 kernel 在 CLC kernel 执行期间到达时，`try_cancel` 可能失败。当前 cluster 随后退出并让出资源；高优先级 kernel 完成后，原 grid 中尚未启动且未取消的 cluster 可以继续被调度。相比固定占满设备的一波 static persistent cluster，这使资源分配更灵活，但这里的“pre-emption”应按 CUDA Programming Guide 的 CLC 语义理解，不等同于任意指令位置的通用抢占。

## 关键权衡

- CLC 通常更适合 tile 耗时不规则的 grouped GEMM、变长 attention 等 workload；规则 dense GEMM 不保证获益。
- 更均衡的 tile 数或 FLOPs 分配不保证更高端到端性能；运行时 tile 顺序可能改变 L2 局部性。
- scheduler warp、shared-memory 响应区、mbarrier 和 pipeline 都会占用资源并引入同步成本。
- CLC、静态 persistent 和 single-tile 应保留为可调度候选，根据 shape、tile 成本分布、缓存行为和并发 kernel 情况实测选择。

## 相关实体

- [[../entities/NVIDIA Blackwell]]
- [[../entities/Colfax Research]]

## 相关来源

- [[../sources/Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs]]

## 相关概念

- [[Tiling]]
- [[Tail Effect]]
- [[GPU执行模型]]
- [[CuTe DSL]]
- [[Occupancy]]
- [[Megakernel]]
- [[Persistent Kernel]]

## 研究备注

- 来源在部分大 shape 均衡 GEMM 中观察到 CLC 的 L2 hit rate 低于静态调度，但没有确定根因，需结合 work-tile 顺序、swizzle 和缓存事件继续 profiling。
- 需要单独整理 CUTLASS C++ 多级 CLC pipeline 与 FlashAttention-4 CLC 实现，确认不同 workload 的 pipeline 深度和抢取时机。
