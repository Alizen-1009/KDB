# LPT 在 Causal Attention 中的调度优化

## 问题

Attention 已经在 GPU 上并行执行，causal mask 也会屏蔽未来 token，为什么还需要 `LPT（Longest Processing Time First）`？它应该如何用于 Atrex FA4？

## 一句话答案

Attention 的并行性解决“多个 Q tiles 能否同时计算”，LPT 解决“这些并行任务耗时不同时，应该先把哪些任务交给 SM”。高效 causal FlashAttention 会直接跳过未来 KV tiles，因而越靠近序列尾部的 Q tile 通常要扫描越多 KV tiles；LPT 让这些长任务先启动，再用短任务填补执行尾部，以减少 [[../../wiki/concepts/Tail Effect|Tail Effect]]。它只改变独立 Q work tiles 的调度顺序，不改变 causal mask 和输出归属。

## 先区分三个层级

### 1. Causal mask：定义数学可见范围

标准 causal self-attention 中，第 `i` 个 query token 只能关注 `0..i`：

```text
Q0  ■ □ □ □ □ □
Q1  ■ ■ □ □ □ □
Q2  ■ ■ ■ □ □ □
Q3  ■ ■ ■ ■ □ □
Q4  ■ ■ ■ ■ ■ □
Q5  ■ ■ ■ ■ ■ ■
```

`■` 是允许计算的位置，`□` 是未来 token。真实 kernel 通常在 softmax 前把被 mask 的 logits 视为 `-inf`；softmax 后对应概率为 0，剩余位置重新归一化。把已经归一化的 attention weights 直接涂灰但不重新归一化，只适合作为示意。

### 2. FlashAttention 并行：定义谁负责输出

典型 fused prefill kernel 把工作分成多个 `(batch, Q head, Q tile)`：

```text
不同 Q tiles / heads / requests：跨 CTA/SM 并行
一个 Q tile 内：沿 KV tiles 流式推进 online softmax
一个 KV tile 内：warp/thread/MMA 并行
```

一个 Q tile 的多个 KV tiles 共享 `m/l/O_acc` online-softmax 状态，所以常规路径通常由同一个 CTA 按迭代顺序处理。完整映射见 [[Prefill Attention 的 CUDA 并行映射]]。

### 3. LPT：定义独立任务的领取顺序

LPT 是调度启发式：先执行预计耗时最长的任务。它不增加数学并行度，也不改变 mask，只希望让多个 SM 更接近同时完成。

## Causal mask 为什么会产生不同长度的 CTA

设 Q/K tile 分别为 `BLOCK_M/BLOCK_N`。在标准等长 causal self-attention 中，第 `q_tile` 个 Q tile 能看到的 KV tile 数可粗略估算为：

```text
q_end          = min(sequence_length, (q_tile + 1) * BLOCK_M)
valid_k_tiles  = ceil_div(q_end, BLOCK_N)
estimated_cost = valid_k_tiles * cost_per_k_tile + fixed_overhead
```

若 `BLOCK_M = BLOCK_N` 且忽略边界，工作量近似为：

```text
Q tile:        0  1  2  3  4  5
KV iterations: 1  2  3  4  5  6
```

因此“所有 Q tiles 都能并行”与“所有 Q tiles 耗时相同”是两件事。

> [!note] 机制推导
> `valid_k_tiles` 是第一版 cost model，不是精确时钟预测。对角 mask、partial tile、TMA、Softmax、epilogue、2-CTA cluster、head dimension 和 pipeline 深度都会影响单次迭代成本；Tensor Core 仍可能计算完整的边界 tile，不能简单用有效元素数估时。

## 如果已经有 CUDA 动态 block 调度，LPT还有什么意义

GPU 会在 SM 释放资源后继续派发尚未启动的 CTA，但默认调度器并不知道哪个逻辑 Q tile 更长。若短任务先启动、长任务晚启动，最后可能只剩少量 SM 处理最长 Q tiles。

用 4 个 SM、8 个耗时分别为 `1..8` 的任务作粗略示意：

```text
短任务优先：第一批 1,2,3,4；长任务 5,6,7,8 较晚开始
LPT：       第一批 8,7,6,5；后续 4,3,2,1 用于填缝
```

理想化 list-scheduling 下，前者 makespan 可到 12 个时间单位，后者约为 9。这里的数字只是调度示例，不是 attention benchmark。

仅反转 CUDA `blockIdx` 不能保证硬件严格按该顺序启动；若需要稳定控制，应由 persistent scheduler、显式 work queue 或已有 FA4 scheduler 把 `scheduler_rank` 映射到原始 work tile。

## 为什么高效 causal kernel 反而更需要考虑负载均衡

有两种实现边界：

### 完整方阵计算后再 mask

如果所有 Q tiles 都扫描完整 KV 范围，只在 score 上施加 mask，那么任务长度接近一致，LPT 收益很小；但上三角做了大量无效计算。

### 直接缩短 causal KV loop

高效 FlashAttention 会让早期 Q tile 提前停止 KV loop。它减少了无效计算，却将规则方阵变成不等长任务集合。LPT处理的是这种“省掉无效工作后暴露出的调度不均”。

## Atrex FA4 的候选实现

> [!warning] 待实现核实
> 当前讨论没有检查 Atrex FA4 的源码、scheduler interface、cluster shape 或现有 work-tile encoding。以下是实现路线，不代表仓库中已经具备对应能力。

### 方案 A：固定长度的隐式 reverse-Q 顺序

对等长 causal self-attention，cost 随 `q_tile` 基本单调增加，可省去显式排序：

```cpp
int rank = scheduler.next();
int original_q_tile = num_q_tiles - 1 - rank;
```

优点：

- 无排序和额外 metadata；
- 映射简单，适合固定 shape；
- 可作为验证 LPT 是否值得的 V0。

限制：

- 只近似适用于规则 causal self-attention；
- 跨 batch/heads 时仍需定义全局 work-item 编码；
- CUDA 原生 block launch 顺序不可靠，最好放进显式 scheduler 映射。

### 方案 B：显式 cost-ordered work list

把实际调度单位编码为：

```text
WorkItem {
    batch_or_sequence,
    q_head_or_head_group,
    q_tile,
    original_output_offset,
    estimated_k_tiles,
}
```

按 `estimated_k_tiles` 降序排列，scheduler 每次领取下一个 item：

```text
scheduler_rank
    → lpt_worklist[scheduler_rank]
    → original (batch, head, q_tile)
    → 按原始坐标写回 O/LSE
```

适用于 varlen、packed sequence、chunked prefill 或不同 mask/window 导致的非单调成本分布。

### 方案 C：LPT 与 persistent/CLC scheduler 组合

LPT 与动态取活并不冲突：

```text
实际 mask/sequence metadata
        ↓
估算每个 logical work tile 的成本
        ↓
LPT logical ordering
        ↓
persistent queue / CLC 领取 scheduler rank
        ↓
rank 映射回原始 output tile
```

在 [[../../wiki/concepts/Cluster Launch Control|Cluster Launch Control]] 路径中，被继承的是 cluster/grid 坐标。可考虑让该坐标先表示 LPT rank，再通过映射表取得真实 Q tile。若 FA4 使用 2-CTA cluster，必须以整个 cluster 为调度单位，不能让两个 CTA 独立领取不同 Q tiles。

LPT负责“长任务先进入候选队列”，CLC/动态任务池负责“哪个空闲 cluster 接下一个任务”。两者可能互补，也可能因动态 stealing 已足够有效而收益重叠，需要实测。

## 生产 cost model 不能只写成 `q_tile + 1`

### Varlen batch

不同 request 的长度不同，应该使用每条序列的实际边界：

```text
cost(sequence, q_tile) ≈ 实际可见 KV tiles 数
```

排序应跨所有可独立调度的 sequences/heads/Q tiles，而不是分别反转每个 batch 后假设成本一致。

### Chunked prefill 与已有 prefix

当前 chunk 中较早的 Q token 仍可看到已经存在的 prefix KV。此时可见范围不是简单的本地 `q_tile + 1`，应复用 kernel 的真实 mask/offset 计算逻辑来估算 `n_block_max`。

### Sliding-window/local attention

窗口达到上限后，各 Q tiles 的 KV loop 长度可能趋于相同，LPT收益会缩小；只按绝对 q index 排序会高估序列后部工作量。

### GQA/MQA 与 head packing

多个 Q heads 可能共享 KV head，或一个 CTA/cluster 同时处理多个 heads。cost model 应对应真实 work-item 粒度，而不是机械按单个 Q head 复制成本。

### Boundary tiles

如果 partial tile 仍执行完整 MMA，只是通过 predicate/mask 丢弃部分元素，成本更接近“执行了几个物理 KV tiles”，而不是“有多少有效 token pairs”。

## 正确性约束

LPT只重排彼此独立的 Q output tiles：

- causal mask 仍由原始 token 坐标计算；
- 每个 work item 仍写回原来的 `O[b,h,q_range,:]` 和 LSE 位置；
- 不允许用 scheduler rank 代替原始 q index 计算 mask；
- dropout RNG、Philox counter 或确定性要求必须绑定原始逻辑坐标，而不是动态领取顺序；
- 训练 backward 若复用 forward metadata，也必须保存原始 tile identity；
- 一个 Q tile 内 KV loop 是否重排属于另一项设计，不应与 Q-tile LPT 混为一谈。

从数学上说，prefill/training 的不同 Q rows 输出互不依赖，因此可重排计算。普通 `q_len=1` decode 没有多个长短不同的 Q tiles，这种 triangular LPT 通常不适用。

## 可能的收益

- 长序列 causal prefill/training 中减少 straggler Q tiles；
- 少 wave、低 `batch×heads` 时缩短 kernel tail；
- varlen batch 中把长 sequence/深 Q tiles 提前；
- 与 persistent scheduler 结合时，提高一波常驻 workers 的任务均衡度。

## 风险与反例

- **排序/metadata 开销**：动态 varlen 每次构建并排序列表可能抵消 kernel 收益；固定 shape 应优先用闭式映射或 bucket。
- **缓存局部性**：改变 Q tile、batch 或 head 顺序可能降低 K/V 的 L2 reuse；FLOPs 更均衡不保证端到端更快。
- **已有动态调度已足够**：work stealing、CLC 或大量 waves 可能已经隐藏成本差异，LPT增益很小。
- **任务足够多**：当 `B×H×Q_tiles` 远大于满载波次，尾部占总时长比例可能有限。
- **短序列/固定开销主导**：KV iterations 差异小于 descriptor、setup、epilogue 成本时，LPT预测不准。
- **非 causal attention**：所有 Q tiles 通常扫描相同 KV 范围，没有天然三角工作量差异。
- **局部 attention**：进入稳定窗口后，后部 tiles 不再持续变长。

## 推荐 Atrex 实验顺序

### V0：证明存在不均衡

在当前 causal FA4 上记录或推导每个 work tile 的：

```text
(batch, head/group, q_tile, valid_k_tiles)
```

用 Nsight Systems/Compute 或 kernel 内低扰动时间戳检查：

- 不同 Q tiles 的实际 duration 是否随 `valid_k_tiles` 增长；
- kernel 尾部是否只剩少数 SM/cluster；
- tail 占总 kernel 时长比例。

### V1：零 metadata 的 reverse-Q

只针对固定长度 causal 路径，把 scheduler rank 映射为 descending Q tile。保持非 causal、decode、local-window 和不支持 shape 的原路径不变。

### V2：Varlen LPT/bucket

若 V1 有收益，再为 varlen 构建 cost bucket，例如按 `valid_k_tiles` 分桶而不是每次完整 `O(n log n)` 排序。需要验证 CUDA Graph capture、host/device metadata 构建与 serving 调度开销。

### V3：与 persistent/CLC 叠加

比较：

1. 原始 FIFO/static；
2. reverse-Q；
3. LPT worklist；
4. dynamic queue/CLC；
5. LPT + dynamic queue/CLC。

只有第 5 项稳定优于第 4 项，才能说明 LPT 在现有动态 scheduler 上仍有增量价值。

## Benchmark 矩阵

至少覆盖：

| 维度 | 建议取值 |
| --- | --- |
| Mask | causal / non-causal / local-window |
| Sequence | 短、中、长；固定与 varlen |
| Grid | 小/大 `batch×heads`，少 wave 与多 wave |
| Head | MHA/GQA/MQA；实际支持的 head dimension |
| Scheduler | FIFO、reverse-Q、LPT、dynamic、组合 |
| 输出 | latency、吞吐、tail duration、SM Active、Tensor Active、L2 hit rate、scheduler overhead |

正确性必须覆盖 output/LSE、边界 tile、varlen、dropout/determinism 和 2-CTA cluster 协同。性能结果需绑定 GPU、dtype、shape、tile、软件 commit 和测量方法。

## 结论

Causal mask 不会自动完成负载均衡。高效 kernel 因跳过未来 KV tiles，使不同 Q tiles 形成从短到长的任务分布；LPT通过“长任务先发、短任务填尾”减少 SM 尾部空闲。它最值得在长序列、少 wave、低 `batch×heads`、varlen 或 persistent FA4 路径中验证，但不是必然加速：动态调度、排序开销和 K/V cache locality 都可能削弱收益。

## 来源与证据边界

### 已有知识库事实

- [[../../wiki/concepts/FlashAttention]]：典型 fused prefill 中，一个 CTA 负责 Q output tile，并沿 KV tiles 流式维护 online-softmax 状态；causal 路径可缩短未来 KV 范围。
- [[Prefill Attention 的 CUDA 并行映射]]：Q-tile grid、CTA 内 KV loop 与 split-KV 边界。
- [[../../wiki/concepts/Tail Effect]]：尾波 block/cluster 不足和 persistent task pool 的作用。
- [[../../wiki/concepts/Cluster Launch Control]]：Blackwell cluster-level 动态 persistent tile scheduling、负载均衡与缓存局部性权衡。
- [[../../wiki/sources/PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化]]：FA4 pipeline 依赖具体 shape、TMEM/SMEM 与 2-CTA layout。

### 机制归纳

- 将 causal Q tile 的预计成本近似为可见 KV tiles 数；
- 对固定长度 causal 路径使用 descending Q tile 近似 LPT；
- 将 LPT logical ordering 与 persistent/CLC scheduler 组合；
- 上述实现与实验路线来自本次讨论的工程推导，不是已有来源宣称的 Atrex FA4 功能。

### 待核实

- Atrex FA4 当前 work-tile 编码、scheduler API、2-CTA/cluster shape 与 causal loop 边界；
- 默认 scheduler 是否已经采用 reverse-Q、cost-aware ordering、work stealing 或 CLC；
- LPT 在目标生产 shape 上相对现有动态调度的增量收益；
- 重排对 K/V L2 locality、deterministic dropout、CUDA Graph 和 metadata 构建成本的影响。
