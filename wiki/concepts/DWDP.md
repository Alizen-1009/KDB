---
type: concept
topic: 并行与分布式
updated: 2026-11-11
sources: 1
---

# DWDP

## 定义

`Distributed Weight Data Parallelism`（DWDP）是面向 MoE 推理的并行策略：在同组 GPU 上分布存放专家权重，但每个 rank 以数据并行方式独立处理请求，在执行某层 MoE 前从 peer GPU 拉取该层缺失专家，在本地完成计算。它以权重搬运换取推理关键路径上不使用跨 rank collective 同步。来源：[[../sources/DWDP Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72]]。

## 它解决什么问题

[[DP Attention]] + [[Expert Parallelism]]（论文称 DEP）通常需要 MoE dispatch/combine：各 rank 请求长度、缓存命中和专家负载不同，collective 同步会把局部慢 rank 转化为组内等待。DWDP 让各 rank 不必为其他 rank 的层进度等待，但仍须等待自身所需远程权重并共享物理互连资源。

## 核心机制

1. 每个 rank 完整复制 attention 权重；各 MoE 层专家分布在组内，允许按需冗余放置，不要求 expert 数整除组大小。
2. 计算第 `l` 层 MoE 与第 `l+1` 层 attention 时，用 copy engine 经 `cudaMemcpyAsync` 串行发起 P2P pull，预取 `l+1` 层缺失的远程专家。进入该层 MoE 前等待本地预取完成；远程缓冲区按层释放并用双缓冲流水。
3. 修改 grouped GEMM 直接消费本地和远程多个权重缓冲区，避免合并到连续缓冲区的 D2D copy。将远程拷贝切片并轮转调度，缓解多个目标同时从同一源拉取造成的争用。

与 EP 的典型区别是：EP 将 token/activation dispatch 到持有专家的 rank，再收回结果；DWDP 将缺失专家权重拉到处理请求的 rank。DWDP 不等于所有专家常驻每个 GPU，也不意味着远程读取和等待消失。

## 关键权衡

- **适用条件**：高速 GPU P2P 互连、足够的逐层计算窗口，尤其是 context/prefill；窗口太短时预取会暴露在关键路径上。
- **理想模型**：论文用 `T_DWDP = max(T_compute, T_prefetch)` 对比 `T_DEP = T_compute + T_all2all`。这是简化近似，忽略失衡和重叠干扰，不是部署延迟预测公式。
- **资源干扰**：即使 copy engine 不占 SM，P2P 仍可争用 HBM、L2、NoC 和功耗；论文附录的 attention 实验观察到功耗限制引起降频。
- **显存与带宽**：attention 权重复制、逐层远程权重缓冲和双缓冲占用显存；增加本地专家冗余可降低远程传输，但需要更多显存。
- **系统级指标**：context 单卡效率提升不必然改善端到端 TTFT；削减 context GPU 可能增加排队，需联合评估吞吐、TTFT、TPOT、资源配比和 SLO。

## 论文实验边界

作者使用 GB200 NVL72、TensorRT-LLM commit `3a89495`、DeepSeek-R1 NVFP4 和 FP8 KV cache，PD 分离中只改变 context 配置。8K/1K、20–100 TPS/user 的端到端实验报告相近 TPS/user 下输出 TPS/GPU 平均提升 8.8%；表 6 的所有配对点中位 TTFT 均增加，20–30 TPS/user 从 2538 ms 到 8314 ms，而 170–180 TPS/user 的 TPS/GPU 为基线的 0.97×。端到端结果默认未包含争用缓解优化。不得将上述数字视为其他拓扑或模型的保证。

## 相关实体

- [[../entities/TensorRT-LLM]]
- [[../entities/DeepSeek-AI]]

## 相关来源

- [[../sources/DWDP Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72]]

## 相关概念

- [[DP Attention]]
- [[Expert Parallelism]]
- [[PD分离]]
- [[通信-计算重叠]]

## 研究备注

- 论文称上游 PR #12136 集成进行中；当前实现及性能需按版本核实。
- 在目标 workload 中同时测量预取暴露时间、copy engine 争用、时钟频率、端到端 TTFT 与 goodput。
