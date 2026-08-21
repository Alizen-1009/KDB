---
type: concept
topic: GPU 编程
sources: 3
updated: 2026-06-12
---

# Megakernel

## 定义

`Megakernel` 指把原本由许多 GPU kernel 串联完成的一整段计算，尽量合并到一个长期运行的大 kernel 内执行的设计方式；在 LLM 低延迟推理语境中，它可以指把整个模型 forward pass 放进单个 kernel。

## 它解决什么问题

- 避免大量小 [[CUDA Kernel]] 之间的 launch / teardown 成本。
- 减少 kernel 边界带来的全局顺序栅栏、[[Tail Effect]] 和权重加载断流。
- 在 batch size 1、模型较小、操作极短的 decode 场景中，让 GPU 尽量持续从 global memory 读取权重，逼近 [[Roofline 模型]] 给出的内存带宽上限。

## 核心机制

- 将 Transformer forward 拆成 kernel 内部的 instruction，而不是 CUDA runtime 层面的许多 kernel launch。
- 在 Python 侧提前生成每个 SM 的 instruction schedule；GPU 侧通过 interpreter 依次执行这些 instruction，并可复用同一 schedule 处理多个 forward pass。
- 对 shared memory 做页式管理：文章中的 H100 实现把前 `213 KiB` shared memory 切成 `13` 个 `16 KiB` page，instruction 显式申请和释放 page，释放后可尽早交给下一条 instruction 发起权重加载。
- 用 global memory 中的 counter 显式表达 instruction 依赖：某个 instruction 完成后递增 counter，后续 instruction 等待 counter 达到目标值再读取输入。
- 对 MLP 中间状态使用 chunk 级依赖，让 down projection 可以在对应 chunk 准备好后开始，而不是等待整个 hidden state 完成。

## 和普通算子融合的区别

- 普通 [[算子融合]] 多数聚焦相邻的 pointwise、norm、activation 或 epilogue；megakernel 试图跨越整层甚至整模型的 kernel 边界。
- Megakernel 的收益不只来自减少中间张量 HBM 往返，还来自消除 kernel 边界本身造成的调度、尾部和 pipeline bubble。
- 代价也更大：原本由 CUDA kernel launch 隐含保证的数据依赖，需要在 kernel 内显式同步和调度。

## 与 Persistent Kernel 的区别

[[Persistent Kernel]] 描述 worker 生命周期和任务取得方式：一个 CTA/cluster 长期驻留并连续处理多个同类 work tiles。Megakernel 描述融合范围：把多个算子、层甚至整个模型的执行放进一个 kernel。二者可以组合，但不是同义词：

- persistent GEMM/attention 可能只处理同一种算子，因此不是 megakernel；
- 整模型 megakernel 通常需要长期驻留 worker、内部 scheduler 和显式依赖，因而往往具有 persistent 特征；
- 两者仍需要至少一次 kernel launch。Persistent kernel 摊销后续 tile 边界，megakernel 则进一步减少算子/层之间的 kernel 边界。

## 与 MegaMoE 的关系

[[MegaMoE]] 是针对 MoE dispatch/FFN/combine 的 wave 级融合流水；本页的 Megakernel 是跨更长算子链、甚至整模型 forward 的广义模式。MegaMoE 是否由单一 persistent kernel 实现尚未被当前二手来源确认，因此不能仅凭名称把两者等同。

## 关键权衡

- 适合固定模型、固定 shape、低 batch、低延迟、memory-bound 的窄场景；不应直接推广为所有 LLM serving 的默认方案。
- 内部调度、shared memory 分配、同步和调试复杂度明显高于普通 fused kernel。
- 过度融合可能牺牲不同阶段各自最优的 tile、线程映射或 Tensor Core 路径；性能依赖硬件架构和 workload 假设。
- Kernel 内显式 counter / barrier 可以更细粒度地表达依赖，但会引入原子操作、barrier 和 consistency 开销。

## 相关实体

- [[../entities/HazyResearch]]
- [[../entities/Megakernels]]

## 相关来源

- [[../sources/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B]]
- [[../sources/MegaMoE — 让 all-to-all 消失]]
- [[../sources/译 NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell]]

## 相关概念

- [[CUDA Kernel]]
- [[算子融合]]
- [[CUDA内存层次]]
- [[GPU执行模型]]
- [[Tail Effect]]
- [[Roofline 模型]]
- [[Programmatic Dependent Launch]]
- [[Profiling]]
- [[MegaMoE]]
- [[通信-计算重叠]]
- [[Persistent Kernel]]

## 研究备注

- 后续应直接 ingest `HazyResearch/Megakernels` repo，核实其 instruction template、scheduler、counter 实现和 benchmark 复现实验。
- 需要进一步区分 megakernel、persistent kernel、CUDA Graphs、Triton/CUDA fused kernel、GEMM epilogue fusion 和 serving engine 调度优化之间的边界。
