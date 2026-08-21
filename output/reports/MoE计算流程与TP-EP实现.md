# MoE 计算流程与 TP/EP 实现

## 核心数学

设输入 `X:[T,D]`，共有 `E` 个 routed experts，每个 token 选择 `K` 个 expert。Router 产生：

```text
logits       = X @ W_router       [T,E]
topk_ids     = TopK(logits)       [T,K]
topk_weights = normalize(...)     [T,K]
```

将每个 `(token, selected expert)` 称为一条 route，因此 route 总数：

```text
R = T × K
route_id = token_id × K + kth
```

每个 expert 通常是 SwiGLU FFN：

```text
gate_e = X_e @ W_gate_e
up_e   = X_e @ W_up_e
mid_e  = SiLU(gate_e) ⊙ up_e
out_e  = mid_e @ W_down_e
```

最后按原 token 合并：

```text
Y[token] = Σ_k topk_weights[token,k] × out(route(token,k))
```

## 单卡/单 rank 的完整执行流程

```text
X [T,D]
  │
  ├─ Router GEMM → logits [T,E]
  │
  ├─ softmax/top-k → topk_ids、topk_weights
  │
  ├─ Count routes per expert
  │
  ├─ Prefix sum + padding/alignment → expert_offsets
  │
  ├─ Populate route maps
  │
  ├─ Dispatch / gather tokens by expert
  │
  ├─ Grouped C1 GEMM: gate + up
  │
  ├─ Epilogue: SwiGLU；NVFP4 时再 quantize/pack/scale
  │
  ├─ Grouped C2 GEMM: down projection
  │
  ├─ Finalize: inverse-permute/scatter + gate-weighted sum
  │
  └─ Y [T,D]
```

高性能实现不一定物化每个中间张量。例如 C1 loader 可根据 route map 直接 gather 原始 `X`，把 token permutation 融进 GEMM load；C1 epilogue 可直接生成 C2 所需的 NVFP4 data/scale layout。

## Populate 是什么

`populate` 不是 MoE 数学算子，而是 dispatch 元数据构建阶段。它把“每条 route 属于哪个 expert”变成“按 expert 连续排列的物理 row 对应哪条原始 route”。

### 三步结构

1. Count：统计每个 expert 收到多少 routes：`expert_counts[e]`。
2. Prefix sum/alignment：得到每个 expert 在 permuted buffer 中的起点 `expert_offsets[e]`，并可按 GEMM tile 做 padding。
3. Populate：为每条 route 找到 expert 内 slot，填写映射表。

```text
permuted_row = expert_offsets[expert] + slot_in_expert
permuted_idx_to_expanded_idx[permuted_row] = route_id
```

示例：

```text
原始 routes：t0e0, t0e2, t1e0, t1e1, t2e1, t2e2, ...
按 expert 排列：
[e0 routes][e0 padding][e1 routes][e1 padding][e2 routes]...
```

`permuted_idx_to_expanded_idx[p] = r` 表示 expert GEMM 的物理 row `p` 来源于 route `r`；再由：

```text
token_id = route_id / top_k
kth      = route_id % top_k
```

找到原 token 和 routing weight。

### 它可能只构建 map，而不复制 activation

- Materialized dispatch：根据 map 把 `X[token]` 真正复制到 expert-contiguous buffer。
- Fused gather：只保存 map，GEMM loader 运行时直接从原 `X` gather；避免一次独立 permute buffer，但 mainloop 多了间接寻址。

用户文档中的 `populate_contiguous_mapping_kernel` 属于后者所需的 metadata builder。

## Preshuffle 是什么

`preshuffle` 是对静态 expert 权重和 scale 做的**离线物理布局变换**。它与 token dispatch/permutation 不是一回事。

逻辑权重可能写成：

```text
W_gate_up: [E, 2H, D]
W_down:    [E, D, H]
```

但 TMA/tcgen05/CUTLASS kernel 希望看到按 tile、K-major、swizzle、interleave 和量化 packing 组织的字节。Preshuffle 会在模型加载、量化或权重转换阶段完成：

```text
逻辑 [E,N,K]
→ 分块/转置
→ FP4 nibble packing
→ scale interleave
→ BlockMajorK/MajorK
→ TMA/tcgen05 所需 swizzle/alignment
→ kernel-specific physical bytes
```

目的：

- 让 TMA 连续搬运目标 tile；
- 让 SMEM layout 直接满足 Tensor Core descriptor；
- 避免每个 forward 都 transpose/repack；
- 让 FP4 data 与 scale 按 kernel 消费顺序排列。

Preshuffle 是 ABI/layout contract。同一逻辑权重经过不同 preshuffle 后，物理 bytes 不可互换。若一套部署同时使用 TensorRT kernel 与自研 kernel，二者要么消费相同 preshuffle，要么维护两份权重/增加转换；这就是文档强调“不做 PD 分离时要兼容 TRT preshuffle”的原因之一。

## Epilogue 是什么

GEMM 分为：

```text
Mainloop：反复加载 K tiles 并执行 MMA accumulator
Epilogue：消费最终 accumulator，做后处理并输出
```

普通 GEMM epilogue 可能只做：

```text
C = alpha × accumulator + beta × residual
→ cast
→ store
```

Fused MoE 的 epilogue 更重。

### C1（Gate-Up）Epilogue

```text
TMEM/RF gate accumulator
TMEM/RF up accumulator
        ↓
SiLU(gate) × up
        ↓
若 C2 使用 NVFP4：
按 16-element group 求 absmax/scale
        ↓
FP4 convert + pack
        ↓
写 FP4 data + scale，作为 C2 输入
```

这相当于把 `do_act + nvfp4_quantize` 融入 C1 GEMM 尾部，避免先写 BF16 中间激活、再单独读回量化。

### C2（Down）Epilogue

通常包括：

- accumulator scale/cast；
- 可选 routing weight 乘法；
- 写入 permuted expert output；
- 或直接 inverse-scatter/atomic-combine 到 token output。

有些实现仍保留独立 `finalize`，因为多个 routes 需要汇聚回同一 token，且 scatter/reduction 与 GEMM tile layout 不完全一致。

## TP 条件下的 MoE

这里假设只启用 TP，不启用 EP。每个 TP rank 都覆盖全部 experts，但每个 expert 的矩阵沿 intermediate 维切片。

### 数据与权重

通常每个 TP rank 都有完整输入：

```text
X: [T,D]
router/top-k：每 rank 相同
expert list：每 rank 都有全部 E 个 expert 的权重 shard
```

对完整 expert：

```text
W_gate/up: [D,H]
W_down:    [H,D]
```

TP=P 后：

```text
W_gate/up_rank: [D,H/P]      # column parallel
W_down_rank:    [H/P,D]      # row parallel
```

### 执行流程

```text
每个 TP rank：
X [T,D]
  → 复制执行 Router/Top-k
  → 复制执行 Count/Populate
  → 对同一批 routes gather X
  → C1 local GEMM，产生 mid shard [R,H/P]
  → C1 epilogue/SwiGLU/可选 NVFP4 quantize
  → C2 local GEMM，产生 out_partial [R,D]
  → inverse-permute + weighted combine（可前后调整）
  → TP all-reduce/reduce-scatter 求和
  → 完整 Y [T,D]
```

由于 down projection 按 H 输入维切开，各 rank 的 `[T,D]` 只是 partial sum，必须跨 TP ranks 求和。

### TP 下 populate/preshuffle/epilogue

- Populate：每 rank 通常构建相同 route map；它是重复控制开销，但不需要 token All-to-All。
- Preshuffle：每 rank 只 preshuffle 自己的 expert weight shard，physical layout 必须匹配本 rank 的 local `H/P` shape。
- C1 epilogue：只处理本 rank 的 intermediate shard；若量化为 NVFP4，scale 通常对应 local shard。
- C2 epilogue/finalize：产生 partial output，之后需要 TP collective。

### TP 的主要瓶颈

- 每个 rank 都保存所有 experts 的一部分，expert 总权重仍铺在整个 TP group；
- Router/populate 重复执行；
- 每层 C2 后有低延迟敏感的 all-reduce/reduce-scatter；
- decode 下 payload 不大但 collective 高频；
- TP=1 时应专用化掉 TP collective 和为 TP 通用布局保留的冗余步骤。

## EP 条件下的 MoE

这里假设 EP、每个 expert 不再额外 TP。不同 ranks 持有不同 experts 的完整权重。

```text
Rank 0: experts 0..m
Rank 1: experts m+1..2m
...
```

### 执行流程

```text
每个 source rank 的本地 tokens X_local
  → Router/Top-k（expert id 是全局 id）
  → Count/Populate send map：按目标 rank/expert 分桶
  → Dispatch All-to-All：发送 activation + route metadata
  → destination rank 接收来自所有 source ranks 的 routes
  → 再按本地 expert populate/reorder/pad
  → 对本地 experts 执行完整 C1 → epilogue → C2
  → Combine All-to-All：expert outputs 发回 source rank
  → source rank inverse-permute + routing-weighted sum
  → Y_local
```

Dispatch 发送的核心是 token activation，不是 expert 权重；expert 权重常驻 owner rank。

### EP 下 populate/preshuffle/epilogue

- Populate 可能出现两次：source 侧构建 destination send layout；destination 侧把收到的数据整理成本地 expert-contiguous rows。
- Preshuffle：每 rank 只处理自己持有的完整 experts；换 placement 时权重与 preshuffled scale 必须随 expert 一起迁移。
- Epilogue：本地 C1/C2 不需要 TP partial-sum all-reduce，但 C2 输出必须携带 source token/route metadata 经 combine All-to-All 返回。
- Finalize：通常在 token 原属 rank 上完成 inverse-permute 和 top-k weighted sum。

### EP 的主要瓶颈

- 两次动态多对多通信：dispatch 与 combine；
- token/expert 负载不均导致 rank straggler；
- EP 越宽，单 rank expert batch 可能越碎；
- Prefill payload 大但更容易形成大 GEMM；decode 消息小、频率高且 expert M 很薄；
- 需要 topology-aware placement、EPLB、wave/pipeline 和通信—计算重叠。

## TP 与 EP 对比

| 维度 | TP-only MoE | EP-only MoE |
| --- | --- | --- |
| Expert 放置 | 每 rank 有全部 experts 的权重 shard | 每 rank 有部分完整 experts |
| Token 是否跨卡找 expert | 否 | 是 |
| Router/Populate | 各 TP rank 常重复 | source/destination 围绕 All-to-All 构建 |
| Expert GEMM | 每 rank 算 partial | owner rank 算完整 expert |
| 主要 collective | C2 partial output All-Reduce/RS | Dispatch + Combine All-to-All |
| Preshuffle | 每个 expert 的 local tensor shard | 本 rank 持有的完整 experts |
| 主要风险 | 每层同步、复制控制开销 | 负载不均、消息碎片、跨节点带宽 |
| 典型用途 | 降低单 expert GEMM/模型宽度压力 | 分散海量 expert 参数，提高系统容量 |

## EP+TP 混合

大模型经常同时使用：

```text
先按 EP 把 experts 分到不同 expert groups
再在每个 expert 内按 TP 切 W_gate/up/down
```

此时一次 MoE 层同时具有：

```text
EP dispatch All-to-All
→ local expert C1/C2 partial compute
→ TP all-reduce/reduce-scatter
→ EP combine All-to-All
```

通信顺序、group 拓扑和是否能融合/重叠取决于框架；`EP size` 与 `TP size` 没有数学上必须相等的要求。

## 对当前 SM103 NVFP4 文档的映射

```text
populate
  = 给 grouped expert GEMM 构建 route→permuted-row 元数据

preshuffle
  = 提前把静态 expert FP4 权重/scale 变成 TRT/CuTe kernel 的物理 layout

epilogue
  = C1 MMA 后执行 SwiGLU+NVFP4 quantize，或 C2 MMA 后执行 scale/store/combine
```

v1/v2/v5 主要优化 Router/Populate/Top-k 控制面，v3 优化 gather 与 C1 epilogue，v4 用 2CTA+TMA multicast 优化 C1 权重复用。它们共同决定端到端 Fused MoE，而不只是 GEMM 峰值。

## 相关页面

- [[../../wiki/concepts/MoE|MoE]]
- [[../../wiki/concepts/Tensor Parallelism|Tensor Parallelism]]
- [[../../wiki/concepts/Expert Parallelism|Expert Parallelism]]
- [[../../wiki/concepts/MegaMoE|MegaMoE]]
- [[Fused MoE NVFP4 v1-v5优化复盘|Fused MoE NVFP4 v1-v5 优化复盘]]
