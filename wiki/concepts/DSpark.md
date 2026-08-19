---
type: concept
topic: 投机解码
sources: 1
updated: 2026-08-17
---

# DSpark

## 定义

`DSpark` 是在 [[DFlash]] 式并行 drafter 基础上加入顺序 token 建模、条件 confidence 和 Hardware-Aware Prefix Scheduler 的 [[并行投机解码]]方法。它允许 drafter 先产生较长候选，再只选择预计能提高吞吐的前缀交给 target model 验证。

## 它解决什么问题

- 固定长并行草稿的后缀接受率低，却仍占用 target verification 算力。
- 不同请求的候选质量不同，不应强制使用相同验证长度。
- Target verify 的效率随总 token batch `B` 变化，因此“多验 token”不一定提高系统吞吐。

## 核心流程

1. Target 对 prefix forward，产生 anchor token 和可供 drafter 使用的中间 context。
2. Parallel block 根据 anchor 与 mask positions 并行计算多个未来位置的中间 logits。
3. Sequential block 从左到右生成候选 token，并输出每个位置的条件 confidence `c_k`。
4. Scheduler 根据 confidence 和离线/预先 profile 的硬件 `SPS(B)` 曲线，为 batch 中每个请求选择验证长度 `l_r`。
5. 只有选中的候选前缀进入 target verification；低价值后缀直接丢弃。
6. Target 从左到右连续接受，通过第一个失败位置后丢弃剩余候选，并产生替代或 bonus token。

## Hardware-Aware Prefix Scheduler

### 前缀存活概率

如果 `c_k` 表示“前面候选都通过时，第 `k` 个 token 仍能通过 target 验证”的条件概率，则验证到位置 `j` 时整个前缀存活的估计为：

```text
a_j = ∏_{i≤j} c_i
```

越靠后的 `a_j` 通常越低，因此新增一个验证 token 的期望收益递减。

### 硬件性能模型

文章用 `SPS(B)` 表示 target 一次处理 `B` 个验证 token 时每秒可运行的 forward steps。若本轮请求 `r` 的验证长度为 `l_r`：

```text
B = Σ_r (1 + l_r)
Θ = τ · SPS(B)
```

其中 `τ` 是估计的每步期望接受 token 数，`Θ` 是吞吐估计。多验证一位会提高 `τ`，也会增大 `B` 并可能降低 `SPS(B)`。

### 贪心扩展

Scheduler 将“为某请求再增加一位验证”的候选按前缀存活概率排序，从短前缀开始逐项扩展：

- 如果更新后的 `Θ` 提升，则保留扩展。
- 如果 `Θ` 不再提升，则早停。

最终每个请求可以得到不同的验证长度。

## 关键权衡

- Confidence 必须足够校准；过高估计会浪费验证计算，过低估计会过早截断可接受候选。
- `SPS(B)` 会随硬件、并行拓扑、kernel、CUDA Graph、并发和版本变化，需要可靠 profiling 与更新策略。
- 全局贪心调度比固定长度复杂，还要考虑调度开销、公平性和 tail latency。
- Sequential block 恢复候选间依赖，但也重新引入一部分串行成本；收益取决于该成本是否显著低于传统 autoregressive drafter。

## 来源 Benchmark

来源在 Qwen3-4B、A800 和其所称 vLLM `0.26.0` 环境中报告：

- `num_speculative_tokens=7`：`584.07 tok/s`，约为 Baseline 的 `2.55x`。
- `num_speculative_tokens=4`：`561.11 tok/s`，约为 Baseline 的 `2.45x`。
- 两组配置下相对 DFlash 为 `1.17x–1.30x`。

这些数字属于来源 benchmark，环境与方法配置尚不足以独立复现或外推。

## 相关实体

- [[../entities/vLLM]]

## 相关来源

- [[../sources/并行投机解码(DFlashDSpark)的快速理解与vLLM实测]]

## 相关概念

- [[并行投机解码]]
- [[DFlash]]
- [[Speculative Decoding]]
- [[Continuous Batching]]
- [[Benchmarking]]

## 研究备注

- 需要阅读 DSpark 原论文 `arXiv:2607.05147`，核对 sequential block 的具体结构、confidence 训练/校准方式、scheduler 最优性边界和 sampling correctness。
- 来源称 vLLM 可直接以 `method=dspark` 部署，但正式支持版本、CLI、CUDA Graph 与 TP/DP 限制待源码验证。
