# 从 FlashKDA 到 GDN：Chunkwise 递归注意力的 GPU 优化方法

## 目标与范围

本文分析 FlashKDA v1 的前向实现，并提炼可用于 Gated Delta Network（GDN）/ Gated Delta Rule Kernel 的优化方法。重点不是把 FlashKDA 机械改名为 FlashGDN，而是回答三个问题：

1. FlashKDA 为什么选择两阶段 Kernel，而不是把所有计算融合进一个 Kernel？
2. 哪些优化来自递归注意力的共性，能够迁移到 GDN？
3. GDN 的 scalar per-head gate 更简单，哪些 KDA 中间量和 HBM 往返有机会被消除？

本文讨论的是公开仓库 commit `d2ff19a` 的 forward/prefill 路径。公开接口当前要求 `torch.inference_mode()`，仓库中没有 backward Kernel，因此不能仅据此推断训练 backward 的实现。

## 一句话结论

FlashKDA 最值得迁移的不是某条 PTX 指令，而是它对**并行性边界**的处理：

> 把可按 `(sequence, head, chunk)` 大规模并行的准备阶段，与只能按 `(sequence, head)` 串行推进状态的 recurrence 阶段拆开；前者追求 occupancy，后者通过 warp specialization、异步流水和寄存器复用压缩每个 chunk 的关键路径。

GDN 可以复用这套调度框架，但不应默认复制 FlashKDA 每 chunk `13,824` 字节的 workspace。由于 scalar gate 显著简化 channel-wise decay，GDN 更有机会采用“轻量两阶段”或按 shape 动态选择 fused persistent Kernel。

---

## 1. 计算结构：并行阶段与递归阶段

### 1.1 Chunkwise delta-rule 的抽象

对一个 chunk，可将计算粗略分为两部分：

```text
Chunk-local prepare（不依赖进入 chunk 的 state）
├── Q/K normalization
├── gate activation 与 cumulative decay
├── chunk 内三角变换
└── 构造后续 GEMM 需要的矩阵

State-dependent recurrence（依赖前一 chunk 的 state）
├── 用进入 chunk 的 state 修正 pseudo-value
├── 计算当前 chunk 输出
└── 更新离开 chunk 的 state
```

只要 prepare 的输出不依赖进入 chunk 的状态，不同 chunk 就可以独立计算；但同一 sequence/head 的状态满足：

```text
S_0 -> chunk 0 -> S_1 -> chunk 1 -> S_2 -> ...
```

这条依赖链不能由普通并行 launch 消除。

### 1.2 单 Kernel 融合为什么可能更慢

若一个 CTA 同时负责 prepare 和 recurrence，grid 通常只能按 `(N, H)` 展开：

```text
grid ≈ number_of_sequences × number_of_heads
```

单条超长序列即使有很多 chunk，也只有 `H` 条独立 recurrence chain。可并行的 chunk-local 工作被迫排在每条 recurrence chain 内串行执行，导致大量 SM 空闲。

FlashKDA 的早期单 Kernel 原型遇到了这一问题。官方 deep-dive 记录：拆成两个 Kernel 后端到端至少提升约 `15%`。该数字只对应作者的实现与实验，不应外推为所有 GDN shape 的固定收益。

---

## 2. FlashKDA v1 的两阶段实现

### 2.1 K1：token/chunk-parallel prepare

Launch 形状：

```text
grid  = [total_chunks, H]
block = 256 threads
```

每个 CTA 处理一个 `(chunk, head)`，完成：

```text
Q/K L2 Norm
-> gate activation
-> cumulative decay
-> q_decayed / k_decayed / k_restored
-> L 与 Mqk
-> 16×16 triangular transform
```

若 `T=8192, H=96, chunk=16`，固定长度单序列的 K1 约有：

```text
8192 / 16 × 96 = 49,152 CTAs
```

因此 K1 有足够的 grid 并行度填满 GPU。

### 2.2 K2：head-parallel recurrence

Launch 形状：

```text
grid  = [N, H]
block = 192 threads = 6 warps
```

每个 CTA 负责一个 `(sequence, head)`，按顺序消费该序列的所有 chunk：

```text
chunk 0 -> state update
chunk 1 -> state update
...
```

K2 的六个 warp 分工为：

```text
warp 0-3：MMA / recurrence compute
warp 4：TMA load producer
warp 5：output store consumer
```

并配置：

```text
3-stage input pipeline
2-stage output pipeline
```

于是可以让 chunk `t+1`/`t+2` 的输入搬运、chunk `t` 的计算和 chunk `t-1` 的输出写回发生流水重叠。

### 2.3 两阶段并不跨 Kernel 并发

公开 v1 在同一个 CUDA stream 上先 launch K1，再 launch K2：

```text
K1 全部完成
-> workspace 可见
-> K2 开始
```

因此这里的主要收益是**并行度解耦**，不是 K1 与 K2 同时占用不同 SM。K2 内部才有显式的 load/compute/store overlap。

---

## 3. 为什么 FlashKDA 使用 16-token chunk

### 3.1 数值范围

Kimi K3 使用 lower-bounded log-decay，单步 gate 被限制在约：

```text
g ∈ (-5, 0)
```

当 `C=16` 时：

```text
sum(g) ∈ (-80, 0)
```

`exp(sum(g))` 及其倒数仍可落在 BF16 动态范围内，避免更大 chunk 所需的分层 rescaling。

### 3.2 小矩阵与 Tensor Core

`C=16` 使 chunk 内三角矩阵为 `16×16`，可以直接映射到 warp-level `m16n8k16` MMA。FlashKDA 使用 SM80 MMA atom 完成小矩阵乘法，但整体数据搬运依赖 SM90 TMA，因此当前公开实现要求 SM90 及以上，而不是能直接运行于 SM80。

### 3.3 有限 Neumann 展开

严格下三角 `16×16` 矩阵满足：

```text
L^16 = 0
```

因此三角变换可以通过有限乘积计算，无须通用 LU inverse：

```text
(I + L)^-1
= (I - L)(I + L^2)(I + L^4)(I + L^8)
```

公开代码以 FP16 完成该展开，最后转为 BF16。作者给出的理由是逆矩阵元素范围受限，FP16 动态范围足够，同时比 BF16 提供更多有效尾数位。

### 对 GDN 的启示

`C=16` 是一个强基线，但不是应永久硬编码的结论。GDN 应分别测试：

```text
C ∈ {16, 32, 64}
```

并同时观察：

- gate 累积的数值范围；
- 小矩阵变换成本；
- recurrence 循环次数；
- K1 grid 数量；
- workspace/HBM 流量；
- K2 pipeline 的计算与访存比例。

GDN 的 scalar gate 可能使 rescaling 与数据布局更简单，允许它在某些 shape 上使用比 KDA 更大的 chunk；这是候选方向，不是已验证结论。

---

## 4. 可直接迁移到 GDN 的优化

## 4.1 按自然并行度拆分 Kernel

这是优先级最高的迁移项。

建议先实现保守版：

```text
GDN K1: grid = [total_chunks, H]
GDN K2: grid = [N, H]
```

K1 只产生 K2 必需的最小中间量。不要一开始追求单 Kernel，因为是否融合应由 recurrence 并行度决定，而不是由“减少 launch 数”单独决定。

适合两阶段的典型场景：

- 单条或少量长序列；
- `N × H` 小，K2 recurrence 并行度不足；
- chunk-local prepare 占比显著；
- prepare 可以跨大量 chunk 独立执行。

## 4.2 Warp specialization 与异步 pipeline

即使 GDN 的数学更简单，K2 仍需顺序推进 state。可以复用：

```text
load warp + compute warps + store warp
```

以及：

```text
多级 input buffer + 多级 output buffer
```

关键验证问题不是“是否用了 TMA”，而是：

- load warp 能否把下一 chunk 的输入准备好；
- compute warp 是否仍因 state dependency 出现空洞；
- store warp 是否真正隐藏输出写回；
- shared-memory stage 数增加是否降低 occupancy。

## 4.3 BF16 resident state + FP32 accumulator

FlashKDA 的 working state 常驻 shared memory，格式为 BF16；GEMM/FMA 使用 FP32 accumulator，每个 chunk 更新后 round 回 BF16。

收益：

- `128×128` state 从 64 KiB FP32 降为 32 KiB BF16；
- state 可直接作为 Tensor Core operand；
- 给 pipeline stage 留出更多 shared memory；
- 避免每个 chunk 的 FP32→BF16 critical-path 转换。

迁移到 GDN 时应至少比较三种策略：

| 策略 | 状态存储 | 更新累加 | 风险 |
|---|---|---|---|
| A | BF16 | FP32 | 性能最好候选，chunk 边界有 BF16 rounding |
| B | FP32 | FP32 | 显存与 shared memory 压力高 |
| C | BF16 + 周期性 FP32 checkpoint | FP32 | 实现复杂，可能折中长期误差 |

必须用长序列测试累计误差，不能只测试单个 chunk。

## 4.4 shared-memory 生命周期复用

FlashKDA K1 使用 `union` 复用不重叠的生命周期：

```text
Phase A: Q / K / gate
Phase B: decayed Q/K / triangular matrices
```

K2 又让 input/output pipeline buffer 与 FP32 state 转换 buffer 共用空间，因为二者分别出现在 recurrence loop 内外。

这类优化与 KDA/GDN 公式无关，可直接复用。实施前应画出每个 tensor 的：

```text
首次写入 -> 最后一次读取
```

再决定 union，而不是仅按 shape 相近复用。

## 4.5 寄存器级 transpose 与 fragment 复用

FlashKDA 中间量 `U` 连续服务于：

```text
INV @ V_tilde
Mqk @ U
K_restored^T @ U
```

代码使用 `MOVM_T` 在寄存器文件内转换 MMA fragment 布局，避免：

```text
register -> shared memory -> transpose load -> register
```

对 GDN，凡是同一中间 tile 连续作为不同 MMA operand 使用，都应优先检查：

- 是否能保留在 registers；
- 是否能通过 fragment reinterpretation 或 `MOVM_T` 转换；
- 是否真的需要 shared-memory round trip。

这通常比单独减少一两条标量指令更重要。

## 4.6 复用 state tile 完成多个 GEMM

FlashKDA K2 在同一个 K-loop 中同时计算：

```text
K_decayed @ state
Q_decayed @ state
```

state tile 载入一次，被两个 GEMM 使用。GDN 若也同时需要 correction term 与 output term，应维持这种 dual-GEMM 结构，而不是拆成两个独立 pass。

## 4.7 base-2 exponent 与近似 sigmoid

FlashKDA 将指数基换成 2，并使用：

```text
ex2.approx.ftz.f32
```

sigmoid 则通过：

```text
sigmoid(x) = 0.5 * tanh(x/2) + 0.5
```

调用 `tanh.approx.f32`。

GDN gate 更简单，适合在同一 Kernel 内融合：

```text
raw gate -> activation -> cumulative gate -> decay factor
```

但需要对照训练/参考实现确认近似指令的误差是否符合模型要求，尤其是长序列状态累计后的偏差。

---

## 5. KDA 特有、不能机械复制的部分

## 5.1 Channel-wise gate

FlashKDA 的 gate 形状近似：

```text
g_kda: [T, H, D]
```

每个 key channel 有独立 decay，因此每 chunk 要产生：

```text
g_total: [D]
```

并对 Q/K 的每个 channel 分别缩放。

当前 benchmark 中的 FLA GDN 则使用 scalar per-head gate：

```text
g_gdn: [T, H]
```

若目标 GDN 实现也是 scalar gate，则：

- `g_total` 可从 `[D]` 缩成一个 scalar；
- cumulative gate 的工作量显著下降；
- state decay 变成整块统一缩放；
- 不必为每个 channel 存储/加载 decay vector；
- K1 的部分计算可能不再值得单独物化。

如果目标 GDN 版本使用 vector gate，则需重新分类，不能套用上述简化。

## 5.2 FlashKDA workspace 不应原样照搬

FlashKDA 每 `(chunk, head)` workspace 为：

| 中间量 | 字节 |
|---|---:|
| `k_decayed` | 4096 |
| `q_decayed` | 4096 |
| `k_restored` | 4096 |
| `g_total` | 512 |
| `INV` | 512 |
| `Mqk` | 512 |
| 合计 | `13,824` |

在 `T=8192, H=96` 时约为：

```text
512 chunks × 96 heads × 13,824 bytes ≈ 648 MiB
```

这是两阶段拆分的主要代价。GDN 设计时应逐项问：

1. `q_decayed`、`k_decayed` 是否必须同时落 HBM？
2. scalar `g_total` 能否作为 metadata 而非完整 vector 保存？
3. `k_restored` 是否可在 K2 从 `k_decayed` 和 scalar decay 重建？
4. `Mqk` 与 triangular transform 是否可合并或按需重算？
5. 某些小矩阵重算是否比 HBM 写回再读取更便宜？

对 GDN，减少 workspace 很可能比继续优化 K1 的标量 gate 指令更有价值。

## 5.3 三角变换必须按 GDN 公式重新确认

FlashKDA 的 `L`、`Mqk`、`INV` 来自 KDA 的 UT transform。GDN 虽然同属 delta-rule recurrence，但具体：

- beta 的位置；
- gate 与 key/value 的缩放方式；
- 三角矩阵的符号；
- causal diagonal 是否保留；
- 是否需要相同的 inverse 展开；

必须以目标 GDN 实现为准。可迁移的是“小型严格三角矩阵可用有限展开和 warp MMA 处理”的方法，不是直接复制 FlashKDA 的矩阵公式。

---

## 6. GDN 的三种候选 Kernel 方案

## 6.1 方案 A：保守两阶段

```text
K1: prepare all chunks
-> global workspace
K2: recurrence per sequence/head
```

### 优点

- 最接近已验证的 FlashKDA 调度；
- K1 并行度最高；
- K1/K2 可分别 profile；
- 适合先建立高性能 correctness baseline。

### 缺点

- workspace 容量大；
- K1 输出产生额外 HBM 写读；
- 两次 Kernel launch；
- GDN prepare 较轻时，拆分收益可能不抵 IO 成本。

## 6.2 方案 B：轻量两阶段

K1 只保存难以在 K2 低成本重算的中间量：

```text
K1 output candidate:
├── triangular transform / INV
├── 必需的小型 chunk matrix
└── scalar cumulative gate
```

K2 重新从 Q/K 和 scalar gate 构造部分 decayed tensor。

### 适用条件

- GDN scalar gate 让重算很便宜；
- HBM workspace 是主要瓶颈；
- K2 仍有足够计算空间隐藏少量重算；
- Q/K 原始输入本来就必须读取。

### 关键权衡

```text
额外 Tensor Core / ALU 重算
vs.
减少 workspace 写入、读取和容量
```

应通过 roofline 与 NCU 验证，不能只比较 FLOPs。

## 6.3 方案 C：融合 persistent recurrence Kernel

一个 CTA 负责 `(sequence, head)`，在内部完成每个 chunk 的 prepare 与 recurrence：

```text
for chunk in sequence:
    load raw Q/K/V/gate
    prepare chunk
    update state
    write output
```

### 优点

- 几乎不需要大 workspace；
- 中间量留在片上；
- 减少 HBM 往返和 Kernel launch。

### 缺点

- grid 只有 `N×H`；
- 单长序列并行度低；
- prepare 无法跨 chunk 展开；
- CTA 生命周期长；
- register/shared-memory 压力可能降低 occupancy。

### 适用条件

- `N×H` 足够大；
- 多 request 或 packed varlen batch；
- 每条 sequence 不太长；
- GDN prepare 明显轻于 KDA；
- workspace 带宽比 recurrence compute 更昂贵。

---

## 7. 推荐：按 shape 动态派发

不建议试图用一个 Kernel 覆盖所有场景。候选派发逻辑为：

```text
if N × H 足够大 and 每条序列较短:
    fused persistent GDN
elif 单条/少量超长序列 and K1 prepare 较重:
    two-stage GDN
else:
    lightweight two-stage GDN
```

需要通过 benchmark 决定阈值，而不是手填固定经验值。

### 需要覆盖的 shape

```text
T_total: 512, 2K, 8K, 32K, 128K
N:       1, 2, 8, 32, 128, 1024
H:       8, 16, 32, 64, 96
D:       64, 128, 256（若目标模型支持）
state:   None / BF16 / FP32
layout:  fixed / packed varlen
```

特别要把总 token 数相同但 sequence 分法不同的 case 放在一起：

```text
1 × 8192
8 × 1024
64 × 128
```

它们能直接揭示 recurrence chain 数量对 GPU 利用率的影响。

---

## 8. 性能模型与诊断指标

## 8.1 K1 应观察什么

- achieved occupancy；
- blocks per SM；
- registers per thread；
- shared memory per CTA；
- Tensor Core utilization；
- TMA throughput；
- register spilling；
- workspace store bandwidth；
- 小矩阵 MMA 是否被标量前后处理拖住。

FlashKDA K1 使用 `__launch_bounds__(256, 8)`，接受少量 spilling 换取更多驻留 CTA。GDN 不应照抄 `8`，而应结合自身 shared memory 与寄存器占用重新扫描。

## 8.2 K2 应观察什么

- 单序列时活跃 CTA 数；
- load pipeline stall；
- compute warp 的 dependency stall；
- store warp 是否落后；
- state shared-memory bandwidth；
- state GEMM 的 Tensor Core 利用率；
- input stage 从 2 增至 3 是否有收益；
- output double buffer 是否足够；
- workspace load 在总时间中的比例。

## 8.3 端到端必须同时报告

```text
K1 latency
K2 latency
workspace allocation/reuse策略
总 forward latency
峰值 workspace
输出与 state 误差
```

只报告 K2 或只报告某个 MMA microbenchmark，无法说明端到端 GDN 是否更快。

---

## 9. 精度验证方案

### 9.1 参考层级

建议保留三层 reference：

1. FP64 recurrent reference：验证数学；
2. FP32 state reference：评估低精度状态误差；
3. 现有 FLA GDN：验证接口和生产路径一致性。

### 9.2 测试维度

- chunk 边界前后；
- 非 16 倍数 tail；
- 长序列累计误差；
- 极端 gate，接近全保留或快速遗忘；
- beta 接近 0/1；
- state in/out 的 BF16、FP32、None 组合；
- packed varlen；
- 不同 sequence 排列是否影响结果；
- 多次连续 prefill，前一次 final state 作为下一次 initial state。

### 9.3 不能只看单步 max error

递归模型还应记录误差随位置的曲线：

```text
position -> output error
position -> state error
```

若 BF16 resident state 出现缓慢漂移，短序列单测可能看不出来。

---

## 10. 推荐实施顺序

### 第 1 步：建立 GDN 算法账本

明确每个 chunk 中：

- 哪些量只依赖当前 chunk；
- 哪些量依赖 incoming state；
- 每个中间量 shape/dtype；
- 每个中间量首次产生和最后使用位置；
- 哪些量可以重算。

### 第 2 步：实现保守两阶段 baseline

先复用 FlashKDA 的并行边界：

```text
K1 [chunks, heads]
K2 [sequences, heads]
```

保证 correctness，再测清 K1/K2 占比。

### 第 3 步：裁剪 GDN workspace

优先处理：

- vector `g_total` -> scalar；
- 可由 scalar decay 重建的 tensor；
- 只使用一次、且重算便宜的中间量；
- 可以保留在寄存器中的 U/fragment。

### 第 4 步：加入 K2 warp specialization

按实际负载测试：

```text
4 compute + 1 load + 1 store
```

以及其他组合，不假定 FlashKDA 的六 warp 配置必然最优。

### 第 5 步：实现 fused persistent 变体

针对 `N×H` 足够大的 varlen/multi-request shape，验证减少 workspace 是否超过 prepare 串行化代价。

### 第 6 步：建立 runtime dispatch

用实际 benchmark 数据确定两阶段、轻量两阶段和 fused 变体的派发边界。

---

## 11. 对 benchmark 的克制解读

FlashKDA 官方报告在 H20、`T=8192, D=128` 上，相对 FLA `chunk_kda` 报告约 `1.85×–2.31×`；GB200 上部分 varlen case 达到约 `3.27×`。

但相对更简单的 FLA GDN，收益明显较小。例如 GB200、`H=64`、固定单序列 case：

```text
FlashKDA: 0.9247 ms
FLA GDN:  0.8857 ms
相对速度: 0.96×
```

这说明：

- FlashKDA 成功消化了 KDA 的额外复杂度；
- 现有 GDN Kernel 本身已经较快；
- GDN 优化目标必须绑定具体瓶颈和 shape；
- “移植 FlashKDA”不是天然等价于“GDN 一定更快”。

特别是 GDN prepare 更轻时，两阶段 workspace 的额外 HBM 流量可能比它释放的并行度更昂贵。

---

## 12. 最终设计原则

1. **先按依赖关系划分模块，再决定 Kernel 边界。**
2. **融合不是越多越好；并行度不同的阶段可能应拆开。**
3. **递归状态固定大小，不代表 recurrence 有足够 GPU 并行度。**
4. **GDN 应利用 scalar gate 简化 workspace，而非照搬 KDA 数据流。**
5. **小矩阵重算可能比写入并重新读取 HBM 更便宜。**
6. **中间 tile 的寄存器复用通常比标量指令微调更重要。**
7. **BF16 state 必须用长序列误差曲线验证。**
8. **单长序列与多短序列应使用不同 Kernel 或调度策略。**
9. **性能结论必须同时报告延迟、workspace、精度和实验 shape。**

---

## 来源与可追溯性

### 官方源码与文档

分析基于本地官方仓库 `/Users/alizen/Dev/FlashKDA`，commit `d2ff19a`：

- `README.md`：支持范围、FLA dispatch 条件与接口。
- `docs/20260420-flashkda-v1-deep-dive.md`：chunk size、两阶段拆分、精度和底层优化说明。
- `csrc/smxx/fwd_launch.cu`：K1/K2 launch、pipeline stage 与 workspace layout。
- `csrc/smxx/fwd_kernel1.cuh`：prepare Kernel、shared-memory union、gate/decay 和三角变换。
- `csrc/smxx/fwd_kernel2.cuh`：recurrence Kernel、warp specialization、TMA pipeline 与寄存器复用。
- `csrc/smxx/utils.cuh`：MMA、Neumann 展开、`MOVM_T` 与精度转换。
- `tests/torch_ref.py`：与 Kernel 对齐的数学 reference。
- `BENCHMARK_H20.md`、`BENCHMARK_GB200.md`：官方 forward benchmark。

### 技术报告

- `raw/papers/k3_tech_report.pdf`：Kimi K3 架构、lower-bounded KDA、FlashKDA 与 KDA 推理系统背景。

### 相关知识库页面

- [[../../wiki/concepts/Chunked Gated Delta Rule]]
- [[../../wiki/concepts/线性注意力递归状态]]
- [[../../wiki/concepts/Tiling]]
- [[../../wiki/concepts/Occupancy]]
- [[../../wiki/concepts/CUDA内存层次]]
- [[../../wiki/concepts/内存合并访问]]

## 来源事实、机制推导与待核实边界

### 来源事实

- FlashKDA v1 使用 `CHUNK=16`、K1/K2 两阶段、BF16 resident state、FP32 state update accumulation、FP16 triangular transform。
- K1 grid 为 `[total_chunks, H]`，K2 grid 为 `[N, H]`。
- K2 使用四个 compute warp、一个 load warp、一个 store warp，以及 3-stage input / 2-stage output pipeline。
- 当前公开仓库仅实现 forward，固定支持 `D=128`，运行要求 SM90+。

### 机制推导 / 工程归纳

- GDN scalar gate 有机会显著缩小 workspace。
- GDN 可能适合轻量两阶段或 fused persistent 变体。
- 应根据 `N×H` 和每条 sequence 长度动态派发 Kernel。
- 某些 K1 中间量可能重算比 HBM materialization 更便宜。

这些是基于源码数据流的设计建议，不是 FlashKDA 作者已经报告的 GDN 实验结论。

### 待核实

- 目标 GDN 版本的精确 gate 形状、UT transform 和 beta 位置。
- GDN 使用更大 chunk 是否在精度和性能上更优。
- 哪些 K1 中间量可以在 K2 重算而获得端到端收益。
- fused persistent、轻量两阶段与完整两阶段的实际派发阈值。
- Kimi K3 内部训练 backward 的 FlashKDA 实现是否与公开 forward 采用相同拆分。
