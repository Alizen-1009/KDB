---
type: concept
topic: 性能分析
sources: 3
updated: 2026-04-23
---

# Benchmarking

## 定义

通过可重复的计时实验测量某段代码或某个算子的端到端执行时间，以比较实现差异和观察性能随规模变化的趋势。

## 它解决什么问题

- 判断两个实现到底谁更快
- 识别性能是否按预期随 batch、dim、layers 或 sequence length 扩展

## 核心机制

- 先 warmup，避免首次执行的编译、缓存和初始化噪声
- 对 GPU 代码在计时边界前后进行 `synchronize`
- 进行多次 trial，观察均值和波动，而不是只看单次结果

## 在线推理 Benchmark 口径

比较 serving 或 speculative decoding 时，除模型和硬件名称外，至少要固定或报告：

- 框架版本 / commit、容器 image digest、模型 revision 和 attention backend。
- 实际 GPU 数、TP/PP/DP 拓扑；“容器可见 4 卡”不等于模型一定使用 4 卡。
- 并发、请求到达模式、输入/输出长度分布、数据集版本和采样参数。
- warmup、trial 次数、均值/P50/P95/P99 与方差，而不是只记录一次 tok/s。
- speculative decoding 还应报告 acceptance length、draft/verify 分段耗时、verify batch shape 和 Baseline 的完全相同配置。
- accuracy 对比需要区分模型/算法变化与 dynamic batch 引起的数值非确定性；temperature=0 也不自动等于 batch-invariant。

## Sparse Attention 评测口径

DeepSeek-V3.2-Exp 的资料补充了两个常见混杂因素：

- **能力分数与输出长度**：GPQA、HLE、HMMT 2025 的差距伴随 V3.2-Exp 生成更少 reasoning tokens；相近输出长度的中间 checkpoint 会缩小差距。比较 sparse/dense 模型时应同时报告 reasoning length、token budget、停止条件和 checkpoint。
- **服务成本与价格假设**：论文 Figure 3 来自 H800 集群实际服务 benchmark，并按 `$2/GPU-hour` 换算；短序列 prefill 还使用 masked MHA mode 模拟 DSA。没有精确表格时，不应从曲线估读并固化美元数字，也不能脱离 GPU 价格、batch 和服务实现比较。

## 关键权衡

- Benchmarking 给你“快不快”，但不直接告诉你“为什么快”或“慢在哪”
- 如果 workload、硬件、库版本或输入形状变化，历史 benchmark 很可能失效

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/vLLM]]
- [[../entities/DeepSeek-V3.2-Exp]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]
- [[../sources/并行投机解码(DFlashDSpark)的快速理解与vLLM实测]]
- [[../sources/DeepSeek-V3.2-Exp：Boosting Long-Context Efficiency with DeepSeek Sparse Attention]]

## 相关概念

- [[Profiling]]
- [[Roofline 模型]]
- [[Speculative Decoding]]
- [[确定性推理]]
- [[DeepSeek Sparse Attention]]

## 研究备注

- DFlash/DSpark 来源中的 `latest` 镜像、未显式 TP 参数和不完整 workload 是典型复现风险；来源数字应绑定原测试，不应用作跨版本通用结论。
