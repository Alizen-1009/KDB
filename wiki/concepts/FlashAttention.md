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

## FA1 与 FA2 的差异

- `FlashAttention-1` 更接近“外 KV 内 Q”：同一个 `KV` block 会反复驱动多个 `Q` block 更新，因此输出状态更容易频繁回写 HBM
- `FlashAttention-2` 更接近“外 Q 内 KV”：固定一个 `Q` block，让所有 `KV` block 流过本地状态，等这块输出完全算完后再一次性写回
- 两者数学上等价，但 `FA2` 的 work partition 更符合输出归属，也更容易把 `O / m / d` 留在本地缓存里

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

## 相关来源

- [[../sources/斯坦福CS336 Lecture 5 - GPUs]]
- [[../sources/Flash Attention 详细解释推演与Pytorch代码实现]]
- [[../sources/vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现]]
- [[../sources/陈巍：DeepSeek 开源Day（1）-FlashMLA 深入分析（收录于：DeepSeek技术详解系列）]]

## 相关概念

- [[Online Softmax]]
- [[Tiling]]
- [[算子融合]]
- [[重计算]]
- [[Roofline 模型]]
- [[PagedAttention]]
- [[Flash Decoding]]
- [[FlashMLA]]

## 研究备注

- 不要把 FlashAttention 简化成“更快的 softmax”；更准确的说法是“围绕 IO 瓶颈重排 exact attention 的数据流”
- 后续可补 FlashAttention v1/v2/v3 的实现差异，以及它和推理引擎中 attention kernel 设计的关系
- 和 `PagedAttention V1` 对比时，应避免说成二者互斥：vLLM 语境下 prefill/decode 可能使用不同 attention backend，区别主要来自阶段、KV cache 管理和并行划分。
- [[Flash Decoding]] 可以视为 decode 场景下沿 `KV/context` 维扩展并行度的 FlashAttention-family 思路；关键不是近似计算，而是 Split-KV 后用 online softmax / log-sum-exp 正确合并局部结果。
