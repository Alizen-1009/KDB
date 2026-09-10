# Fused MoE NVFP4 v1-v5 优化复盘

## 范围与证据边界

- 对象：`flashinfer.trtllm_fp4_block_scale_moe`，SM103 NVFP4 fused MoE，prefill 场景。
- 流程：`route → topk_gate_softmax → gemm13 → do_act → nvfp4_quantize → gemm2 → finalize`。
- 本报告依据用户提供的钉钉文档摘录进行机制审阅，并参考 KernelWiki 的 Blackwell TMA/TMEM/2-SM/Fused-MoE 页面与上游 PR 线索。
- `10us/3us/10us/32us/2us` 的含义、完整 shape、commit、时钟和测量方法未给出，因此均视为来源报告值，不能直接相加或外推。

## 最重要的结论

### 1. 这是一次“控制面→数据面→计算面”的纵向优化

五轮并非只在优化 Tensor Core：

| 层次 | 版本 | 主要瓶颈 |
| --- | --- | --- |
| 路由元数据 | v1 | 重复 atomic、过宽索引 |
| Top-k 控制与寄存器 | v2、v5 | spill、冗余状态、runtime specialization |
| Gather 与 epilogue | v3 | mapping 重读、无效 scale、FP4 中间量 live range |
| GEMM/cluster 数据复用 | v4 | 两个 M tiles 重复加载同一 expert 权重和 scale |

这解释了为什么早期 C1 GEMM 的收益会掩盖 Top-k 落后：端到端优化必须同时测量 orchestration、routing、GEMM 和 epilogue，不能只看主 GEMM TFLOPS。

### 2. 通用方法是“让信息在第一次产生时就留下来”

v1 中 `atomicAdd(expert_counts[e], 1)` 的返回旧值天然就是 route 在 expert 段内的 slot。保存 `route_slots[route]` 后，populate 阶段不再重新执行同一计数。

这是可迁移原则：

> 如果后续阶段需要 producer 已经计算过的顺序、offset、mask 或 scale，优先把它作为紧凑 metadata 向后传递，而不是在 consumer 重算或重新竞争。

但新增 `route_slots` 也有一次 global store/load，收益来自用顺序内存流量替换 contended atomic；应分别测热门 expert、高度均匀 routing 和不同 token 数。

### 3. 编译期 specialization 是贯穿 v3/v5 的统一主题

- `unit_scale` 专用 kernel 跳过 scale load/multiply；
- `top_k=10` constexpr 消除 runtime guard，并允许完整 unroll、constant propagation 和更精确的 live-range 分析；
- v1 的 `int64→int32` 也是把实际输入边界写入数据契约。

正确工程形式不是把所有参数硬编码，而是建立有限 specialization key：

```text
(top_k, unit_scale_c1, unit_alpha_c2, tile_m, use_2cta, sm_target)
```

再控制 variant 数量、JIT/cache 命中和 dispatch 开销。

### 4. NVFP4 的格式粒度应成为 epilogue 调度粒度

v3 从 whole-subtile 量化改为每 16 元素 scale-vector group 流式处理：

```text
16 values: SwiGLU → absmax → scale → FP4 convert → pack/store → 释放
```

这把软件临时量生命周期与 NVFP4 scale block 对齐。一般原则是：

> 量化格式以多大 group 共享 scale，epilogue 就优先以多大 group 产生、归约、转换和提交数据。

它减少同时存活的 `up/gate/tCompute/abs/scale/packed` fragments，并将一条长依赖链拆成多个独立短链，提高 ILP。

### 5. v4 的本质是“按复用域组织 cluster”，不只是把 M128 改成 M256

两个 CTA 满足：

```text
相同 expert、N tile、K tile
不同 M/token rows
```

因此 B 权重和 SFB scale 是 cluster 内共享数据，A/token rows 是 CTA 私有数据。TMA multicast 让一次上游读取服务 cluster 内多个 CTA，并填充各 CTA 的 SMEM destination；2-SM `tcgen05.mma` 再让两个 CTA 协作一个 M256 work item。

这条优化成立的前提是 scheduler 能把具有相同 `(expert, N, K)` 的两个 M tiles 配对。若 expert token 数较少或高度不均，M256 可能增加 padding、减少独立 work items，并放大尾部，因此需要 M128/M256 hybrid heuristic，而不是全 shape 固定 2CTA。

## 分版本审阅

### v1：Atomic 返回值复用与 int32 metadata

**成立的知识：**

- `atomicAdd` 返回旧值可作为 segment-local slot；
- 把 count 与 slot acquisition 合并，可消除 populate 阶段第二次 contended atomic；
- `topk_ids` 和 route index 若有严格上界，int32 可减少 metadata 带宽、cache footprint 和 64-bit address arithmetic。

**需要验证：**

- `token_count × top_k`、expert offsets 和 padded route 总数均不会溢出 int32；
- 外部 API、TensorRT/FlashInfer ABI 是否要求 int64；
- atomic 决定的 expert 内 row 顺序可能不稳定。若后续 finalize 使用原子累加或结果要求 bitwise determinism，需要验证顺序变化是否影响输出；
- `expert_counts`、prefix sum、populate 三阶段之间必须有明确 kernel/stream ordering。

来源称 `populate_contiguous_mapping_kernel` 从 `14.5us` 降至 `2.8us`；该值需绑定 shape 与 commit。

### v2：删除可重算数组、分布式持有 Top-k 结果

**成立的知识：**

- `lane_ids[i] = i*32+lane` 属于低成本可重算状态，不值得与 `lane_vals` 一起长期占寄存器；
- 每 lane 保存完整 `selected_vals[10]/selected_ids[10]` 会复制 warp-wide 状态；让 lane `k` 只拥有第 `k` 个结果，可显著减少 per-thread state；
- 编译器不能把动态索引数组完全标量化时，数组很容易落到 local memory，删除后收益可能主要来自消除 spill。

**正确性重点：**

- 每轮 winner 必须从后续候选中排除；
- tie-breaking、NaN、`-Inf` 和相同 logits 行为要与 baseline 完全一致；
- 最终 softmax denominator、routing weight 和输出写回需要从分布在不同 lanes 的 top-k 值正确归约；
- “省掉 `__shfl_sync(best, 0)`”只有在 warp reduction 已让每个需要它的 lane 得到 winner 时才成立。

### v3：Mapping invariant hoist、unit-scale variant、group-streaming quantization

**Mapping cache：**把 K-loop 内重复读取的 route→token mapping 提升到 CTA 前置阶段，并缓存“已经 decode 的 token index”而不只是 raw route，可同时减少 global loads 与重复整数除法。512-byte SMEM 成本很小，但仍需检查 bank conflict 和 barrier 是否被 L1/L2 命中收益抵消。

**Unit-scale：**最好由 host/config metadata 保证并选择专用 kernel；若每次运行前扫描整个 scale tensor 判断是否全 1，检测成本可能超过收益。

**量化生命周期：**来源给出的 register count 约 `165→161`，若 ptxas 分配档位仍是 168，则理论 occupancy 没变。此时收益更可能来自：

- spill/local-memory 减少；
- live range 缩短；
- dependency chain 变短；
- ILP 提升。

不能把它归因为 occupancy 提升。

### v4：2CTA M256 + TMA multicast

**成立的知识：**

- B/scale 沿 M 方向复用，非常适合 cluster-M TMA multicast；
- 每个 CTA 仍有自己的 SMEM storage，multicast 减少上游 GMEM/L2 transaction，不表示两个 CTA 共享同一块物理 local SMEM；
- 官方 CUTLASS 2-SM 示例由 leader CTA issue MMA，两个 CTA 的 operand partitions 和 TMEM partitions共同参与，随后双方等待 completion 并各自消费输出分块。

**必须增加的 heuristic：**

```text
if expert_rows 足够且 B-load pressure 高：M256 / 2CTA
else：M128 / 1CTA
```

还应将 expert token 分布、cluster occupancy、有效 M 利用率、TMA bytes、L2 hit rate 和 tail duration 纳入选择，而不是只看 C1 GEMM 平均时间。

### v5：`top_k` constexpr

这项优化合理，但原文有两处需要修正：

1. `if (kth < top_k)` 对整个 warp 通常是 warp-uniform 条件，因此不构成 lane-level warp divergence。收益主要来自消除 uniform branch/predicate、允许 dead-code elimination、完整 unroll、constant propagation 和缩短 live range。
2. runtime `top_k` 本身多占一个标量寄存器通常不是主要成本；真正的成本是它让多个迭代和 conditional paths 必须保留。

若产品需要多个 top-k，应为常见值生成有限 variants，并保留 generic fallback。

## 关于 TMEM 的重要修正

“TMEM 使很重的 epilogue 不影响 MMA”过于绝对。TMEM 确实把 accumulator 从普通 RF 移出，使 MMA producer 与 epilogue warps 更容易解耦；但只有满足以下条件时才能充分重叠：

- 有双缓冲或足够 TMEM partition；
- epilogue 不过早占满 TMEM/RF/SMEM；
- completion、fence 和 mbarrier 正确；
- store/quantize 吞吐不形成 backpressure；
- 同一 CTA 的 issue bandwidth 和 CUDA Core 资源未成为瓶颈。

因此应测 `Tensor Active`、epilogue duration、TMEM buffer wait、local spill 和 store throughput，而不是根据架构图直接假设完全重叠。

## SM103 是独立的兼容性风险

该工作目标是 SM103，不能把 SM100 成熟实现直接视为可用。KernelWiki 收录的上游记录已经出现：

- vLLM PR #30484：专门加入 SM103/GB300 支持；
- SGLang PR #9807：`cvt.e2m1x2` guard 原先只覆盖 SM100a，导致 SM103a FP4 quantize 失败；
- 另有 SM100 attention kernel 在 SM103 hang 或精度失败后被临时禁用的记录。

因此 specialization key 必须包含 exact SM target，CI 至少覆盖：正确性、极端 routing、CUDA Graph、不同并发度和长时间稳定性；“同属 SM10x”不足以证明兼容。

## 当前文档最缺的证据

建议把 v0-v5 做成可审计表：

| 版本 | Commit | GPU/SM | Shape | Stage latency | E2E latency | Regs | Lmem | SMEM/TMEM | Tensor Active | L2/DRAM | 数值误差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

并同时保留：

- cumulative：v0、v1、v2、v3、v4、v5；
- ablation：在同一最终代码上分别关闭每项优化；
- token/expert 分布：均匀、热点 expert、长尾、空 expert；
- prefill 与 decode；
- 对比 TensorRT 时统一 preshuffle layout、workspace、CUDA Graph、warmup 和计时边界。

各轮收益存在交互，不能直接把 `10+3+10+32+2us` 相加。

## 对 Agent 优化流程的提炼

文档里最值得保留的不是“让 Agent 再想一些优化”，而是下面的闭环：

1. **先解释数据流和不变量**：每张 mapping 表表示什么、谁生产、谁消费、生命周期多长。
2. **限制 review 范围**：一次只审 route/top-k、metadata、epilogue 或 GEMM scheduler，降低注意力稀释。
3. **参考实现迁移前先核对假设**：TP、EP、prefill/decode、SM target、layout 和 deterministic requirement。
4. **独立模型/人工复核**：专门寻找重复工作、错误归因和被 baseline 大收益掩盖的退化。
5. **维护实验账本**：每个优化种子记录 hypothesis、metric、patch、result、keep/revert，而不是只保存建议文本。
6. **人类专家守住系统契约**：算法语义、Preshuffle/layout 兼容、生产场景、benchmark 公平性和上线边界。

## 最值得继续尝试的方向

1. 对每个 expert 按 token count 在 M128/1CTA 与 M256/2CTA 间动态或分桶选择。
2. 将 top-k、unit-scale、SM target、2CTA 组成受控 autotune/specialization key，避免 variant 爆炸。
3. 检查 C1→C2 的 FP4 数据和 scale 是否可以保持最接近 C2 TMA/tcgen05 消费布局，消除额外 slice/repack；SGLang PR #15731 提供了消除 unpadded-output slice 的上游线索。
4. 对 v3 epilogue做 stall attribution，确认收益来自 spill、dependency 还是 store backpressure。
5. 单独优化 finalize/combine 与 routing weight 读取，避免 C1 优化后瓶颈迁移到尾部。

## 相关资料

- KernelWiki：`wiki/kernels/fused-moe.md`
- KernelWiki：`wiki/hardware/tma.md`、`tmem.md`、`2sm-cooperative.md`、`nvfp4.md`
- CUTLASS 2-SM CuTe DSL 示例：`examples/python/CuTeDSL/blackwell/tutorial_gemm/`
- vLLM PR #30484：SM103 support
- SGLang PR #9807：SM103 FP4 quantize guard
- SGLang PR #15731：消除 `trtllm_fp4_block_scale_moe` slice
