---
type: concept
topic: GPU 编程
sources: 0
updated:
---

# MegaMoE

## 定义

`MegaMoE` 是把 Expert Parallel MoE 的 dispatch、L1/up GEMM、activation、L2/down GEMM 与 combine 组织为细粒度融合流水，并按 expert wave 调度，以当前 wave 的计算覆盖后续/前序 wave 的通信。

```text
Wave 0: Dispatch -> L1 -> Act -> L2 -> Combine
Wave 1:          Dispatch -> L1 -> Act -> L2 -> Combine
Wave 2:                   Dispatch -> L1 -> Act -> L2 -> Combine
```

“All-to-All 消失”仅表示它可能从端到端关键路径上被隐藏，不表示通信逻辑或字节量消失。

## Wave 调度

Wave 是本 batch 命中 experts 的子集。切成多个 waves 后，已到达 token 的 wave 可以开始 GEMM，另一个 wave 同时 dispatch 或 combine。

增加 wave 数会加深流水、减少填充/排空 bubble，但也会让单 wave GEMM 的 `M` 更小、消息更碎，并增加调度与同步成本。最优配置需平衡通信覆盖、Tensor Core 利用率、消息粒度和 expert tail。

## 完全隐藏通信的条件

来源按 token-expert pair 估算：SwiGLU gate/up/down 为 `6*h*d_ff` FLOPs，FP8 dispatch 与 BF16 combine 为 `3*h` bytes，因此 workload 计算/通信比为：

```text
6*h*d_ff / (3*h) = 2*d_ff FLOPs/Byte
```

若硬件峰值/有效计算吞吐与互联带宽比 `C/B <= 2*d_ff`，计算时间有机会覆盖通信；若 `C/B` 更高，网络相对不足，仍有暴露通信。

这是必要的平衡直觉而非充分条件：实际还受有效吞吐、拓扑、消息粒度、wave 调度、负载不均和功耗/DVFS 影响。`6144` 只来自来源示例的 `d_ff=3072`。

## 与 DBO 的区别

[[Dual Batch Overlap]] 在 runtime 层切两个 microbatch，用 MB0/MB1 交错隐藏等待；MegaMoE 在单个 batch 内按 expert waves 融合五阶段。二者可概念叠加，但若 MegaMoE 已饱和 Tensor Core、HBM 和互联，DBO 可能只增加资源争用。

## 与 Megakernel 的关系

[[Megakernel]] 是跨算子链甚至整模型 forward 的广义模式；MegaMoE 是针对单个 MoE 数据流的具体融合流水。当前二手来源未明确其是单一 kernel、persistent kernel 还是协同 kernels，不能只因名称含 `Mega` 就断言与整模型 megakernel 实现相同。

## 关键权衡

- 优点：减少通信暴露时间、kernel/阶段空泡和中间调度开销。
- 代价：wave 过细会伤害 GEMM 与网络效率；融合实现、同步和容错复杂。
- 通信与计算并发可能争用 SM、HBM、L2、互联和功耗预算。
- 修改 SwiGLU 数学以降低 SFU 成本属于模型重设计，不是 MegaMoE runtime 的免费优化。

## 相关来源

- [[../sources/MegaMoE — 让 all-to-all 消失]]

## 相关概念

- [[MoE]]
- [[Expert Parallelism]]
- [[通信-计算重叠]]
- [[Dual Batch Overlap]]
- [[Megakernel]]
- [[算子融合]]
- [[Tail Effect]]

## 研究备注

- 性能数字和实现边界均待 DeepSeek-V4 一手报告或公开代码核实。
- Pull/Push 与 IBGDA 的术语应回到具体原语定义，避免沿用二手材料的宽泛类比。
