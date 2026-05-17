# vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现

## 来源信息

- 标题：vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现
- 作者：[[../entities/方佳瑞]]
- 日期：2023-12-28
- 类型：文章 / CUDA kernel 代码实现解析
- 原始文件：[[../raw/articles/vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现]]
- 原始链接：[知乎专栏](https://zhuanlan.zhihu.com/p/673284781?share_code=HttD5x9CDmV1&utm_psn=2015858798674338445)

## 2-3 条核心摘要

- 文章从 CUDA 并行算法设计视角解释 `vLLM` 的 `PagedAttention V1`：一个 CUDA thread block 负责一个 `sequence + head` 的输出行，内部通过 warp / thread group 遍历 paged KV cache，完成 `QK -> softmax -> PV`。
- 文章补强了 `KCache` 与 `VCache` layout 差异的解释：`KCache [num_blocks, num_kv_heads, head_size/x, block_size, x]` 服务于 QK 阶段的 thread group 协作和 logits 写入；`VCache [num_blocks, num_kv_heads, head_size, block_size]` 服务于 `softmax(QK^T) @ V` 阶段沿 token 维读取。
- 文章把 `PAv1` 与 `FlashAttention / FlashDecoding` 的任务划分区别讲清楚：PAv1 通常每个 block 写一行 output，并保留整行 `QK^T` 中间结果；FlashAttention 则更强调二维 tiling、online softmax 和减少 score/probability 中间矩阵物化。

## 值得关注的论断

- `PagedAttention` 的页式 KV cache 管理能提升吞吐和显存利用率，但间接寻址也可能增加单请求延迟，因此 kernel 实现质量会直接影响 vLLM 的综合表现。
- 作者称 `PAv1` 在 `MQA/GQA` 下没有充分减少 KV cache 读取次数，读 K/V 时主要只是 `head_idx -> kv_head_idx` 映射，可能重复读取相同 KV head；该判断需要按具体 vLLM 版本源码核实。
- 作者称 `PAv1` 对很长序列适应性有限，因为没有沿 `context length` 或 batch 维度做进一步切分；`PAv2` 借鉴 FlashDecoding 的思路，通过 sequence 维切分增加并行粒度。

## 关键概念

- [[PagedAttention]]
- [[KV Cache]]
- [[FlashAttention]]
- [[CUDA Kernel]]
- [[Online Softmax]]
- [[Warp Shuffle Reduce]]

## 相关实体

- [[../entities/vLLM]]
- [[../entities/方佳瑞]]

## 与现有 wiki 的关系

- 会更新哪些概念页：`PagedAttention`、`KV Cache`、`FlashAttention`
- 会更新哪些实体页：`vLLM`、`方佳瑞`
- 是否存在冲突：无直接冲突；本来源主要把既有 `PagedAttention` CUDA 走读从代码行级补充到并行算法设计视角。

## 待确认

- 原文记录的是 2023 年底附近的 `vLLM PagedAttention V1` 实现视角；具体启发式阈值、kernel 名称、layout 和 MQA/GQA 优化程度需要按具体 vLLM commit 或新版本源码核实。
- 原文称 `V1` 适合长度小于 `8192` 或 `num_seqs * num_heads > 512` 的情况，这应作为版本相关实现备注保留，不宜写成长期稳定规则。
