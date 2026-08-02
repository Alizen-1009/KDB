---
type: concept
topic: 模型架构
sources: 2
updated: 2026-07-25
---

# LatentMoE

## 定义

`LatentMoE` 是在模型主干 hidden space 与 routed expert 路径之间增加低维潜在空间的 MoE 结构。Token 进入 MoE block 时仍是模型 hidden size `d`，只在 routed experts 内部临时压缩到 latent dimension `ℓ`，计算完成后再恢复到 `d`。

```text
x: [T, d]
-> down projection: d -> ℓ
-> Router / Top-k / Routed Experts in latent space
-> up projection: ℓ -> d
-> y: [T, d]
```

它不是把整个模型 hidden size 改成 `ℓ`，也不是简单缩小 expert intermediate size。

## 三种维度

- `d`：模型主干 hidden size / residual stream width。Attention 输出、MoE 输入输出和下一层接口通常都保持 `[T, d]`。
- `ℓ`：LatentMoE 潜在维度。Routed expert 接收和返回的 token activation 宽度。
- `m`：单个 expert FFN 的 intermediate size，即 SwiGLU/MLP 内部扩展宽度。

传统 routed expert：

```text
[T, d] -> d -> m -> d -> [T, d]
```

LatentMoE routed expert：

```text
[T, d]
-> project d -> ℓ
-> expert ℓ -> m -> ℓ
-> project ℓ -> d
-> [T, d]
```

若使用 SwiGLU，传统 expert 的 gate/up/down 权重规模粗略为 `3 × d × m`，潜在 expert 的相应规模粗略为 `3 × ℓ × m`，但整个 block 还要加入潜在空间上下投影和其他路径。

## 它解决什么问题

传统 MoE 提高 Expert 总数或 Top-k 时，成本会受到两类约束：

- 激活 expert 时要读取与主干 hidden width `d` 相关的 expert 权重；
- [[Expert Parallelism]] 需要在 ranks 间发送 `[tokens, d]` activation，All-to-All payload 随 `d × Top-k` 增长。

LatentMoE 先把 routed path 压到 `ℓ`，使 expert 权重和 EP activation 的主导尺寸与 `ℓ` 相关，从而在相同系统预算下探索更多 experts、更高 Top-k 或不同 expert 组合。

## 权重读取与通信账本

忽略 SwiGLU 常数、metadata 和投影开销时：

```text
传统 expert 输入/输出相关权重量级  ~ d × m
Latent expert 对应权重量级          ~ ℓ × m

传统 EP activation payload          ~ tokens × d × Top-k
Latent EP activation payload         ~ tokens × ℓ × Top-k
```

因此主导子项的理论尺寸比例约为：

```text
Latent / Traditional ≈ ℓ / d
```

例如 `d=4096, ℓ=1024` 时，`d/ℓ=4`，expert 权重读取与 All-to-All 主 activation payload 的尺寸可粗略降到四分之一。

但这不代表：

- 整个 Expert 参数量精确缩小 4 倍；
- All-to-All 时间精确缩短 4 倍；
- MoE block 或端到端推理快 4 倍。

实际还包括：

- `d -> ℓ` 与 `ℓ -> d` 投影计算、权重和中间激活；
- Router、shared expert、gate 与 residual 路径；
- Expert intermediate size `m` 和 Top-k；
- Token permutation、index、scale、padding 和对齐；
- Collective latency、同步、负载不均与小 batch GEMM；
- Attention 和其他非 MoE 层。

## 与 Expert Intermediate Size 的区别

直接缩小 `m` 会减少 expert 内部容量；LatentMoE 保留独立的 `m`，主要压缩 expert 与模型主干相连的输入/输出空间。其目标是在减少每次 expert 访问系统成本的同时，保留或重新分配 expert 内部容量。

但 `ℓ` 太小仍可能形成信息瓶颈；最终质量取决于投影学习能力、Expert 数、Top-k、`m`、训练预算和路由稳定性，不能只按参数形状判断。

## 与 Sparsity Allocation 的关系

LatentMoE 增加了新的预算轴：

```text
latent dimension ℓ
<-> total experts E
<-> active Top-k K
<-> expert intermediate m
<-> projection overhead
<-> EP communication budget
```

降低 `ℓ` 可以为更多 Expert 或更高 Top-k 释放权重/通信预算，但会增加潜在瓶颈风险。它把 MoE 扩展问题从“只堆 Expert 参数”转成“设计每个 Token 访问 Expert 的表示宽度和组合数量”。

## 关键权衡

- 优点：减少 routed expert 的权重带宽和 EP activation payload，可能扩大专家组合空间。
- 代价：增加上下投影，并可能引入信息瓶颈、额外同步或小矩阵开销。
- Top-k 增长仍会增加计算和通信，只是每次 expert 访问的 payload 更窄。
- 端到端收益高度依赖 routed experts 是否为主要瓶颈，以及 projection 是否能被融合、摊薄或复用。
- 质量结论必须在等参数、等 active FLOPs、等训练 token 和一致 benchmark 下比较。

## 面试口径

一句话：LatentMoE 保持模型主干 hidden size `d` 不变，只把 routed expert 路径投影到 `ℓ` 维；`m` 仍是 expert 的 intermediate size。这样 expert 权重和 EP activation 的主导尺寸从 `d` 变为 `ℓ`，再用节省的预算增加 experts 或 Top-k。

## 相关实体

- [[../entities/Moonshot AI]]

## 相关来源

- [[../sources/2026 年MoE 架构正在发生一次关键变化]]

## 相关概念

- [[MoE]]
- [[Expert Parallelism]]
- [[Sparsity Allocation]]
- [[混合精度训练与推理]]
- [[Tensor Parallelism]]

## Kimi K3 官方实例

Kimi K3 官方技术报告确认其 Stable LatentMoE 使用 `d=7168`、latent dimension `ℓ=3584`、单 expert hidden dimension `m=3072`，共有896个 routed experts、每 token激活16个，并有2个 full-width shared experts。routed path 执行 `d→ℓ` 后路由和 expert 计算，再经 RMSNorm 与 `ℓ→d` 返回主干；expert weights 使用 MXFP4、输入 activation 使用 MXFP8，非 expert 模块保持更高精度。

Serving 优化中，latent down-projection 与 router GEMM 融合，latent weights 跨 TP ranks 分片，并把 output AllGather 融入 GEMM epilogue；这说明 projection 开销是否可融合是 LatentMoE 端到端收益的关键。详见 [[../entities/Kimi K3]] 与 [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)。

vLLM的K3 Preview还记录：SiTU已接入MXFP4 TRTLLM-Gen与DeepGEMM路径，AMD侧使用FlyDSL MLIR的A16W4/A8W4融合算子；16-GPU `DP16+EP16`只明确完成Optimized Backend选择与Correctness Check，不能当作吞吐Benchmark。

## 相关来源补充

- [[../sources/A Preview of Production-Scale Kimi K3 Support on vLLM]]

## 研究备注

- LatentMoE 的通用质量/性能结论仍需等参数、等 active FLOPs、等训练 token 和一致硬件 benchmark；Kimi K3 单例不能证明所有模型都受益。
- Nemotron 3 Super 的具体配置仍主要来自二手来源，需官方报告核实。
