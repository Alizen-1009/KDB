---
type: concept
topic: 注意力机制
sources: 5
updated: 2026-07-18
---

# FlashAttention

## 定义

一种围绕 GPU 内存层级重新组织 exact attention 计算的数据流算法。它通过 `Q / K / V` 分块、[[Online Softmax]] 和 kernel 融合，尽量把中间状态留在寄存器或 SRAM 中，从而显著减少 HBM 读写。

## 它解决什么问题

- 避免标准 attention 显式物化大尺寸 score matrix 或概率矩阵带来的 `O(N^2)` 中间存储与带宽压力
- 缓解 safe softmax 多次遍历完整 score 的 IO 成本，让 attention 不至于长期被 memory wall 拖住
- 让长序列 attention 更接近 GPU 友好的访问模式，而不是在 HBM 和 SRAM 之间频繁搬运中间结果

## 核心机制

- 对 `Q / K / V` 和中间 attention 计算做 tile-wise 分块
- 对每个 `Q` 行块维护逐行的 `m / d / O` 状态，其中 `O` 更适合理解为“未归一化分子”的累积值
- 使用 [[Online Softmax]] 支持按块归一化，并避免显式保存完整概率矩阵
- 通过融合与重计算减少中间张量物化和反复访存，本质上是用少量额外计算换更低的 `HBM` 读写

## Prefill CUDA 并行映射（典型 FA2 风格）

> [!note] 实现归纳
> 以现代 fused FlashAttention forward kernel 为例；具体 tile size、warp 分工和 grid 维度顺序会随 FlashAttention 版本、GPU 架构、head dimension 和 backend 改变。

- 设 `Q: [B, Hq, Nq, D]`，典型 grid 可理解为 `ceil(Nq / Br) × B × Hq`。一个 CUDA thread block / CTA 负责一个 `(batch, Q head, Q row tile)`，最终只写它所属的 `O[b, h, q0:q0+Br, :]`。
- block 先把一块 `Q` 搬入 shared memory / registers，并为该 tile 的每个 query 行初始化 `m=-inf`、`d=0` 和未归一化输出累加器 `O_acc=0`。
- 随后该 block 沿 `K/V` 序列分块循环：协作搬运 `K_j/V_j`，用 Tensor Core / MMA 计算 `S_j = Q_i K_j^T`，施加 scale 和 mask，做逐行归约，再用 [[Online Softmax]] 更新 `m/d/O_acc`，并计算 `P_j V_j`。
- `K/V` tile 之间存在 online softmax 状态依赖，因此常规 prefill 路径由同一 block 顺序扫过它所需的 `K/V` tiles；但 tile 内的拷贝、MMA、行归约和 `PV` 由多个 warp / thread 并行，并可用异步拷贝和双缓冲重叠搬运与计算。
- 遍历完所有可见 `K/V` tiles 后，block 做一次 `O_acc / d`，把输出 tile（以及需要时的 log-sum-exp）写回 HBM。因果 mask 下，只需遍历当前 `Q` tile 可见的 `K/V` 范围。
- block 由 GPU 动态调度到某个 [[GPU执行模型|SM]]；它一旦驻留就不跨 SM 迁移，但一个 SM 可以同时驻留多个 block。实际数量受线程数、寄存器和 shared memory 用量限制，并不是“每个 SM 固定只跑一个 block”。
- 当 `B × Hq × ceil(Nq/Br)` 太小、难以喂饱所有 SM 时，某些 backend 会再沿 `KV/context` 维做 split-KV，由多个 block 产生局部结果，再用 log-sum-exp / online-softmax 规则合并；这是增加并行度的特化路径，不是上述常规映射的必选步骤。

## Causal Q tile 负载与 LPT 候选调度

> [!note] 机制归纳
> 本节是基于上述 Q-tile/KV-loop 映射的调度推导，不是现有来源宣称的通用 FlashAttention 实现特性。

高效 causal kernel 会直接缩短每个 Q tile 的 KV loop：序列前部 Q tile 只扫描少量 KV tiles，后部 Q tile 扫描更多历史，因此 CTA 虽然可以并行，耗时却不相同。`LPT（Longest Processing Time First）` 可把“可见 KV tiles 较多”的 Q work tiles 优先交给 persistent worker/cluster，再用短任务填补尾部，以减少 [[Tail Effect]]。

LPT 只改变独立 Q output tiles 的调度顺序；causal mask、online-softmax KV loop 和输出写回仍使用原始 `(batch, head, q_tile)` 坐标。固定等长 causal self-attention 可用 descending Q tile 近似 LPT；varlen、chunked prefill、local window、GQA/head packing 和 boundary tile 则需要按真实 mask iterator 估算成本。默认 CUDA block 调度不保证严格按 block index 启动，稳定实现通常需要显式 scheduler 映射或 persistent work queue。

该优化最可能适合长序列、少 wave、低 `batch×heads` 或 varlen prefill；non-causal、普通 `q_len=1` decode、大量独立 work tiles，或已有充分动态 stealing 的路径可能收益很小。完整实现与验证清单见 [[../../output/reports/LPT在Causal Attention中的调度优化|LPT 在 Causal Attention 中的调度优化]]。

## FA1 与 FA2 的差异

- `FlashAttention-1` 更接近“外 KV 内 Q”：同一个 `KV` block 会反复驱动多个 `Q` block 更新，因此输出状态更容易频繁回写 HBM
- `FlashAttention-2` 更接近“外 Q 内 KV”：固定一个 `Q` block，让所有 `KV` block 流过本地状态，等这块输出完全算完后再一次性写回
- 两者数学上等价，但 `FA2` 的 work partition 更符合输出归属，也更容易把 `O / m / d` 留在本地缓存里

## FA4 HeadDim=256 的 Blackwell 专用流水

PAI-FA 来源给出一个 shape 改变后必须重做 pipeline 账本的案例：`head_dim` 从 128 增至 256 后，`S` tile 形状不变，但 `O/dQ/dK/dV` 等 [[Tensor Memory|TMEM]] accumulator footprint 翻倍，原有多 stage 方案会超出片上容量。

- Forward 维持 `128×128×256` 大 tile，但因单次 MMA 工作量翻倍、Softmax 工作量基本不变，将掩盖关系从约“2 MMA 对 1 Softmax”改为约“1 MMA 对 1 Softmax”。Q stage 从 2 降到 1，只保留一个 O tile，并让同一 Q 下的不同 K tile ping-pong。
- Backward 拆为 `dQ kernel` 与 `dKdV kernel`：前者使用 `128×128`、Outer-Q/Inner-K，后者使用 `128×64`、Outer-K/Inner-Q。拆分把 MMA 数从约 5 增至约 7，但避免 `dQ/dK/dV` 同时占用 TMEM。
- 2-CTA/DSMEM 用于分摊 SMEM、共享操作数和扩大协作 tile。该收益依赖 Blackwell 数据路径、具体 dtype、mask、GQA、序列长度和实现版本，不是所有 FlashAttention backend 的通用配置。

## 为什么它通常更快

- 真正省下来的往往不是 FLOPs，而是中间矩阵的物化与全局显存往返
- `FA2` 会把最终除法推迟到所有 `KV` block 处理结束之后，减少内层循环中的慢操作
- 在 causal mask 场景下，`外 Q 内 KV` 的执行顺序还更自然支持对未来块提前停止
- 这种收益在 `prefill` 阶段通常更明显，因为长序列 attention 的大头更接近中间矩阵 IO，而 `decode` 阶段则更容易被缓存读写主导

## 与 PagedAttention V1 的区别

- `PagedAttention V1` 在 decode 场景中常以一个 CUDA thread block 计算一个 `sequence + head` 的输出行，并把这一行的 `QK^T` logits 放入 shared memory 后做 softmax。
- `FlashAttention` 更强调二维 tile-wise 数据流：在 `Q/K/V` 分块间维护在线 softmax 状态，避免完整 score/probability 矩阵物化。
- 因此二者可以放在不同层面理解：`PagedAttention` 的核心是 paged KV cache 管理和 decode kernel 间接寻址，`FlashAttention` 的核心是 exact attention 的 IO-aware tiling 与 online softmax。

## 与 FlashMLA 的关系

- [[FlashMLA]] 可视为 FlashAttention 思路在 DeepSeek [[MLA]] decode 后端上的特化：同样强调减少 HBM 往返和中间矩阵物化，但输入/cache 形态变成 latent KV、paged cache 与变长序列 metadata。
- 因此不要把 FlashMLA 简化为“直接套用 FlashAttention”；它还要处理 MLA 的矩阵吸收、latent cache layout、Split-KV 合并和 Hopper-specific kernel 优化。

## 关键权衡

- 能显著降低 memory traffic 并提升长序列吞吐
- 实现复杂度高，性能收益依赖硬件特性、序列长度、tile 设计与 kernel 质量
- 在 PyTorch 级别的分块模拟里，收益常会被 Python 调度和 tensor 临时写回掩盖，真实优势通常要到底层 CUDA / Triton kernel 才能完全体现

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/vLLM]]
- [[../entities/TensorRT-LLM]]
- [[../entities/NVIDIA Blackwell]]
- [[../entities/阿里云 PAI 团队]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 5 - GPUs]]
- [[../sources/Flash Attention 详细解释推演与Pytorch代码实现]]
- [[../sources/vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现]]
- [[../sources/陈巍：DeepSeek 开源Day（1）-FlashMLA 深入分析（收录于：DeepSeek技术详解系列）]]
- [[../sources/PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化]]

## 相关概念

- [[Online Softmax]]
- [[Tiling]]
- [[算子融合]]
- [[重计算]]
- [[Roofline 模型]]
- [[PagedAttention]]
- [[Flash Decoding]]
- [[FlashMLA]]
- [[Tensor Memory]]
- [[Tail Effect]]
- [[Cluster Launch Control]]

## 研究备注

- 不要把 FlashAttention 简化成“更快的 softmax”；更准确的说法是“围绕 IO 瓶颈重排 exact attention 的数据流”
- 后续可补 FlashAttention v1/v2/v3 的实现差异，以及它和推理引擎中 attention kernel 设计的关系
- 和 `PagedAttention V1` 对比时，应避免说成二者互斥：vLLM 语境下 prefill/decode 可能使用不同 attention backend，区别主要来自阶段、KV cache 管理和并行划分。
- [[Flash Decoding]] 可以视为 decode 场景下沿 `KV/context` 维扩展并行度的 FlashAttention-family 思路；关键不是近似计算，而是 Split-KV 后用 online softmax / log-sum-exp 正确合并局部结果。
