---
type: concept
topic: 推理服务
sources: 2
updated: 2026-07-25
---

# Attention-FFN 分离

## 定义

`Attention-FFN Disaggregation`（AFD）是面向 MoE 推理的算子角色分离架构：把 Transformer 层中的 Attention 路径与 FFN/MoE 专家路径部署到不同 worker/rank 集合，通过连接器逐层传递中间激活和执行元数据，使两侧能够采用不同拓扑并独立扩缩容。

AFD 不是把模型重排成“先执行所有 Attention，再执行所有 FFN”。Transformer 的层间依赖仍然成立：

```text
Attention[l] -> FFN[l] -> Attention[l+1]
```

因此每个被拆分层都要完成 Attention 到 FFN、再从 FFN 返回 Attention 的数据交换。

## 它解决什么问题

- Attention 容量主要受请求状态、序列长度、KV Cache 和调度压力影响；MoE FFN 容量主要受 token 路由、专家负载、Grouped GEMM 与 All-to-All 通信影响。共用一套 worker 拓扑时，两类负载无法独立扩容。
- Attention 侧需要调度器、KV Cache、批处理和采样；FFN 侧只需要当前层激活、路由/执行元数据和返回通道。角色分离可让 FFN 侧成为更轻量的专家计算服务。
- GPU/NPU 的通信库、graph runtime 和 MoE 算子不同。后端中立的连接器契约可以稳定模型流程，同时允许后端实现各自的数据通路。

## 逐层执行机制

Attention 服务通常保留：

- 请求调度和批处理
- KV Cache 生命周期
- Attention 计算
- 模型层推进与采样

FFN 服务通常负责：

- 接收 `layer_id`、hidden states 与执行元数据
- 执行对应层的 Router、token dispatch 和 MoE/FFN
- 把 FFN 输出返回 Attention 服务

以三层模型为例：

```text
Attention 服务                         FFN 服务

Layer 0 Attention
      ├──── hidden states + 元数据 ────> Layer 0 FFN/MoE
      <────────── FFN output ───────────┤
Layer 1 Attention
      ├──── hidden states + 元数据 ────> Layer 1 FFN/MoE
      <────────── FFN output ───────────┤
Layer 2 Attention
      ├──── hidden states + 元数据 ────> Layer 2 FFN/MoE
      <────────── FFN output ───────────┤
LM Head / Sampling
```

FFN 服务不是“一层部署一个服务”。通常是一组 FFN ranks 承担所有切分层的专家计算，通过 `layer_id` 选择当前层。Norm、Residual 和 Router 的精确归属取决于模型封装，但不会改变逐层 `A -> F -> A` 的依赖。

## 传输什么

每个切分层通常有两次主要传输：

1. `A -> F`：FFN 输入 hidden states，以及层号、ubatch、data-parallel、graph 等执行元数据。
2. `F -> A`：当前层的 FFN/MoE 输出，供残差合并并进入下一层。

若 decode step 的有效 token 数为 `B`、hidden size 为 `H`、激活元素占 `s` 字节、切分 MoE 层数为 `L`，只计算双向主激活时，单 step 的边界通信量可粗略写为：

```text
2 * L * B * H * s
```

实际通信还受 TP shard 布局、rank 映射、padding、元数据和额外 gather/scatter 影响。Decode 单次激活较小但 step 多且对延迟敏感；Prefill 单次激活较大，但更容易通过 ubatch 形成流水和较大的专家 GEMM。

## 同步与异步执行

### 同步 AFD

单个 ubatch 严格执行：

```text
Attention[l]
-> send
-> FFN[l]
-> return
-> Attention[l+1]
```

同一 ubatch 的下一层 Attention 不能在上一层 FFN 返回前开始。

### 异步 AFD

异步执行不会打破同一 ubatch 的模型依赖，而是交错不同 ubatch：

```text
时间      t0        t1        t2        t3
A ranks   U0-Attn   U1-Attn             U0-next-Attn
F ranks             U0-FFN    U1-FFN
```

当 F 侧处理 `U0` 时，A 侧可以处理 `U1`，从而重叠 Attention、激活传输与专家计算。收益取决于两侧阶段是否均衡；一侧明显更慢时仍会形成 bubble 和排队。

## 与 Pipeline Parallelism 的区别

AFD 具有算子级两阶段流水的特征，但不等于传统 [[流水线并行]]。

| 维度 | 传统 PP | AFD |
| --- | --- | --- |
| 切分轴 | 模型深度、连续层范围 | 每层内部的 Attention/FFN 角色 |
| Stage 持有内容 | 一段完整 Transformer 层 | 所有切分层的 Attention 或 FFN |
| 激活流向 | 通常沿 PP stages 单向前进 | 每个切分层在 A/F 之间往返 |
| 主要目的 | 分摊模型权重与深度计算 | 两类负载独立扩缩容和后端优化 |
| 主要风险 | Pipeline bubble、stage 不均衡 | 高频激活通信、A/F 配比失衡 |

因此更准确的表述是：AFD 是一种**可流水化的算子角色分离**，而不是新的按层 PP。

## 与其他并行方式组合

AFD 只定义 Attention 与 FFN 的服务边界，不替代两侧内部的并行策略。

### 与 Expert Parallelism

[[Expert Parallelism]] 自然位于 F 侧：

```text
A ranks
  -> AFD connector
  -> F ranks: Router
  -> EP dispatch All-to-All
  -> Local experts / Grouped GEMM
  -> EP combine All-to-All
  -> AFD connector
  -> A ranks
```

需要区分两类通信：AFD 通信跨 A/F 服务传 hidden states 与 FFN output；EP 通信在 F 侧按专家位置分发和收回 token。

### 与 Tensor Parallelism

[[Tensor Parallelism]] 可以分别作用于 A/F 两侧：A 侧切 Attention projection/head，F 侧可在 EP 之外继续切单个 expert。两侧 TP size 不必概念上相等，但连接器可能需要完成 TP shard 到 EP/TP 布局的重分布。是否支持具体组合取决于插件版本、连接器和已验证 recipe，不能把“可组合”理解成任意拓扑均已开箱验证。

### 与数据并行和 DP Attention

A 侧可以按请求和 KV Cache 组织多个数据副本，多个 Attention replica 再共享或映射到独立的 FFN/EP pool。这样 Attention 容量可按请求数、上下文长度和 KV Cache 扩展，FFN 容量可按 token 数、专家负载和通信能力扩展。

### 与 PD 分离

AFD 与 [[PD分离]] 是不同切分轴，可以叠加：

```text
Prefill pool
├── Prefill Attention ranks <-> Prefill FFN ranks
└── 完成后把 KV Cache / 请求状态交给 Decode

Decode pool
└── Decode Attention ranks <-> Decode FFN ranks
```

两种边界传输不同：

| 维度 | PD 分离 | AFD |
| --- | --- | --- |
| 边界 | Prefill -> Decode | Attention <-> FFN |
| 主要数据 | 各层 KV Cache 与请求状态 | 当前层 hidden states 与 FFN output |
| 频率 | 以请求阶段交接为主 | 每个 forward step 的每个切分层 |
| 生命周期 | KV Cache 在后续 decode 中长期保留 | 中间激活主要服务当前层/当前 step |

## 与 Wide-EP 的边界

[[Wide Expert Parallelism]] 在同一 serving 拓扑内以 Attention DP replicas 共享宽 expert pool；AFD 则把 Attention 和 FFN 提升为独立服务角色，可分别设置 rank 数和扩缩容策略。两者都需要 A/F 激活交换，也都可使用 EP、DBO 等内部优化，但 AFD 的服务边界和部署自由度更强、逐层连接器复杂度也更高。

## 关键权衡

- **资源专用化与通信开销**：独立扩缩容可能提高设备利用率，但要覆盖逐层双向激活传输和布局转换成本。
- **A/F rank 配比**：分离本身不保证提速。来源实验中较低 Attention/FFN 配比落后于 EP 基线，提高 Attention ranks 后才取得更高归一化吞吐。
- **流水重叠与单请求依赖**：不同 ubatch 可以交错，但同一 ubatch 仍必须等待当前层 FFN 完成。
- **部署灵活性与系统复杂度**：独立拓扑带来更多 rank 映射、连接器、故障恢复、版本兼容和性能调参问题。
- **权重驻留**：当前 vLLM AFD Plugin 的两个角色都加载完整权重，角色分离尚不等于理想化的权重分片或显存减半。

## 面试口径

一句话：AFD 是把每个 MoE Transformer 层的 Attention 与 FFN 部署在不同 rank 集合上，逐层传 hidden states 和 FFN output，以高频通信换取两类资源的独立扩缩容。

与 PD 分离的区别：PD 主要在 prefill 完成后把 KV Cache 交给 decode；AFD 则在每个切分层、每个 step 都要在 Attention 和 FFN 之间往返传递临时激活。

与 PP 的区别：PP 按连续层切模型深度，AFD 按每层内部的算子角色切分；AFD 可以用 ubatch 做流水，但不是传统按层 PP。

## 相关实体

- [[../entities/vLLM AFD Plugin]]
- [[../entities/vLLM]]
- [[../entities/NCCL]]

## 相关来源

- [[../sources/vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署]]
- [[../sources/vLLM Large Scale Serving DeepSeek @ 2.2k toksH200 with Wide-EP]]

## 相关概念

- [[MoE]]
- [[Expert Parallelism]]
- [[Tensor Parallelism]]
- [[流水线并行]]
- [[PD分离]]
- [[DP Attention]]
- [[集合通信]]
- [[KV Cache]]
- [[Wide Expert Parallelism]]
- [[Dual Batch Overlap]]

## 研究备注

- 项目仍处于实验阶段，当前性能数字来自特定昇腾拓扑、模拟逻辑规模、强制均衡路由或裁剪模型，不能外推为通用生产结论。
- 后续需要结合仓库源码核实 connector 在不同 TP/EP 布局下是否执行额外 gather/scatter，以及 Norm、Residual、Router 在各模型封装中的精确归属。
- 值得继续跟踪按角色只加载所需权重、更多 ubatch、完整 graph、真实路由精度与多节点故障恢复能力。
