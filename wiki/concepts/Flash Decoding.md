# Flash Decoding

## 定义

`Flash Decoding` 是面向自回归 decode 阶段的 attention 并行化与 IO-aware kernel 思路：当每个请求本轮通常只有 1 个 query token 时，把历史 `KV Cache` 沿 context/token 维切成多个 split，并行计算局部 attention，再通过 [[Online Softmax]] / log-sum-exp 统计合并为精确的全局输出。

## 它解决什么问题

- Decode 阶段 `Q` 很短，batch 小时仅按 `sequence + head` 并行容易吃不满 GPU。
- 长上下文下每步都要读取大量历史 `K/V`，容易受 HBM 带宽、SM 利用率和 tail effect 影响。
- 需要在不改变 attention 数学结果的前提下，增加 `KV` 维度上的并行粒度。

## 核心机制

- 将历史 `K/V` 按 context length 切成多个 `KV split`。
- 每个 split 内可用 [[FlashAttention]] 式分块计算，得到局部输出、局部最大值与局部分母/log-sum-exp。
- split 之间不能直接把局部输出相加，必须先根据全局最大值修正各块的 softmax 尺度，再合并分母和输出分子。
- 在多 GPU / context parallelism 口径下，`KV split` 可以位于不同设备；此时还需要交换 `Q`、局部输出和 log-sum-exp 统计。

## 和 FlashAttention 的关系

可以把 `Flash Decoding` 理解成 **decode 场景下的 Split-KV FlashAttention**，但这个说法要加边界：

- 相同点：都依赖 tile-wise attention、[[Online Softmax]] 和减少中间 score/probability 矩阵物化。
- 差异点：[[FlashAttention]] 的经典主战场是 prefill / training 这类 `Q` 长度较大的二维 attention；`Flash Decoding` 的关键是 `Q` 极短时沿 `KV/context` 维再切分，增加并行度。
- 实现上，Flash Decoding 往往需要额外的 split 合并 kernel 或跨设备归约逻辑；它不只是把 FlashAttention API 换个参数。

## 当前实现中的普及程度

`Split-KV` 已经可以看作现代 LLM serving decode attention 后端里的常见实现技巧，尤其用于长上下文、小 batch、`Q length = 1` 或接近 1 的场景。不同系统未必叫 `Flash Decoding`：可能叫 `split-KV`、`KV sequence parallelism`、`multi-block attention`、`recursive / cascade attention`，或在多 GPU 上进一步变成 [[Decode Context Parallel]] / context parallel 的一部分。

但它不是“所有 attention 都默认 Split-KV”：

- prefill / training 阶段 `Q` 长度较大，通常已经能沿 batch、head、query block 等维度提供足够并行度，不一定需要沿 KV 再拆。
- decode 阶段如果 batch 足够大、context 较短或 head 数足够多，普通 paged attention / FlashAttention-family kernel 可能已经够用。
- `Split-KV` 会引入局部输出、max / log-sum-exp 和最终合并的额外开销；是否启用通常由 backend 的 heuristic、固定 split size、上下文长度、硬件和 cache layout 决定。

## 和相邻概念的区别

- 与 [[PagedAttention]]：PagedAttention 重点是 paged KV cache 管理与间接寻址；Flash Decoding 重点是对长 `KV` 维做并行切分与 softmax 合并。二者可以组合。
- 与 [[Chunked Prefill]]：Chunked Prefill 切的是 prefill 阶段的 prompt 调度粒度；Flash Decoding 切的是 decode attention 读取的历史 KV。
- 与 [[Decode Context Parallel]]：DCP 可视为多 GPU serving 口径下的 decode context/KV 分片；它和 Flash Decoding 一样需要正确合并局部 softmax 统计，但重点是减少跨 rank KV cache 重复。`vllm并行策略之DCP` 直接把 DCP 类比为“分布式 Flash Decoding”，同时提示二者在 TP group 复用、KVCache 布局和通信方式上仍有实现差异。
- 与 [[FlashMLA]]：FlashMLA 是 DeepSeek [[MLA]] decode 后端的特化 kernel；它也会用 Split-KV/IO-aware 思路，但输入形态是 latent KV 和 MLA metadata。

## 面试回答模板

> 基本可以说 Flash Decoding 是 FlashAttention 在 decode 场景下的 Split-KV 版本。普通 decode 每个请求只有一个 query token，如果只按 batch/head 并行，长上下文小 batch 时 GPU 并行度不够。Flash Decoding 把历史 KV cache 沿 context 维切成多个 split，每个 split 内算局部 attention，同时保存局部 max 和 log-sum-exp，最后用 online softmax 的合并公式把多个局部结果还原成全局精确 attention。所以它不是近似 attention，也不是 speculative decoding；核心是 KV 维并行和正确的 softmax 归约。

## 相关来源

- [[../sources/vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现]]
- [[../../raw/articles/推理长序列利器：ChunkedPrefill&FlashDecoding原理详解]]
- [[../sources/vllm并行策略之DCP(Decode Context Parallel)]]

## 相关概念

- [[FlashAttention]]
- [[Online Softmax]]
- [[KV Cache]]
- [[PagedAttention]]
- [[Chunked Prefill]]
- [[Decode Context Parallel]]
- [[FlashMLA]]

## 研究备注

- `split-KV` 增加并行度的同时也引入合并开销，收益依赖 context length、batch size、num heads、cache layout、硬件和 kernel 实现。
- 面试里不要把它说成“FlashAttention 的另一个名字”；更准确是“decode 小 Q 场景下，沿 KV/context 维扩展并行度的 FlashAttention-family 实现思路”。
- 面试里也不要把 `Split-KV` 说成罕见技巧；现代 serving 后端已经普遍吸收了这个思路，只是会按 workload 条件选择是否启用。
