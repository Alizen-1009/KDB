---
type: concept
topic: 注意力机制
sources: 1
updated: 2026-05-17
---

# CSA 与 HCA

## 定义

根据当前 DeepSeek V4 官方 Transformers 文档，`CSA` 与 `HCA` 是两种不同的长程压缩注意力层，而不是同一个结构的两个名字：

- `CSA`：`Compressed Sparse Attention`，先做低倍率压缩，再用 Indexer 稀疏选择压缩条目。
- `HCA`：`Heavily Compressed Attention`，先做高倍率压缩，不使用 Indexer，而是对全部可见压缩条目做 dense attention。

## 共同骨架

- 两者都建立在共享 `K=V` 的 MQA 路径上，并保留独立的 local sliding-window `K=V` 支路，以维护细粒度局部依赖。
- 长程 compressor 产生的条目与 local branch 一起进入核心 attention；区别主要在压缩强度、窗口是否重叠，以及是否经过稀疏选择。
- 因此它们既不同于只压缩 KV 表示的 [[MLA]]，也不同于直接在原始 token 上做 top-k 的 [[DeepSeek Sparse Attention|DSA]]。

## CSA：低压缩后再稀疏选择

- 官方 Transformers 默认压缩率为 `m=4`，使用 overlapping windows 形成较细粒度的 compressed pool。
- Lightning Indexer 对 query 与压缩条目打分，每个 query 只 gather top-`index_topk` 条目进入长程核心 attention。
- 其状态除 local sliding-window KV 外，还包括 compressor overlap state、compressed pool 与 Indexer state。

## HCA：重压缩后的 Dense Attention

- 官方 Transformers 默认压缩率为 `m'=128`，使用 non-overlapping windows 形成更粗粒度的 compressed entries。
- HCA 没有 Indexer；每个 query 可以关注全部因果可见的压缩条目，并同时使用 local sliding-window branch。
- 它通过更激进的压缩直接缩短长程序列，换取更粗的远程分辨率。

## 早期 RoPE 解析中的补充线索

现有来源页来自官方结构公开前的二手解析，其中以 `C128A` 为例讨论：压缩后为每个块选择位置标尺再施加 RoPE，并对共享 K/V 场景下的输出逆旋转进行推导。这些内容可作为位置编码实现思路，但具体 `C128A` 命名、起始位置 `128*t` 与输出逆旋转仍需绑定正式模型版本和源码，不能覆盖上述官方 CSA/HCA 定义。

## 关键权衡

- CSA 保留更细的远程分辨率，但需要 overlapping compressor、Indexer、top-k gather 和对应 cache state。
- HCA 移除 Indexer、状态更简单，但高倍率压缩会折叠更多 token 级细节；对 compressed entries 的 dense attention 仍随条目数增长。
- 两者都依赖 local branch 为近期 token 保底；不能只按长程压缩率推断端到端质量或速度。
- 压缩条目的 RoPE 标尺、cache layout、dtype 与层调度都可能随模型版本变化，应绑定 checkpoint config 和实现核实。

## 相关实体

- [[../entities/DeepSeek-AI]]
- [[../entities/DeepSeek V4]]

## 相关来源

- [[../sources/DeepSeekV4中RoPE设计解析]]

## 官方资料

- [DeepSeek V4 Transformers 文档](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)
- [先进大模型架构知识图谱](../../output/reports/先进大模型架构知识图谱.html)

## 相关概念

- [[RoPE]]
- [[MLA]]
- [[KV Cache]]
- [[DeepSeek Sparse Attention]]
- [[mHC]]

## 研究备注

- 当前正式定义已按 DeepSeek V4 官方 Transformers 文档更新；旧来源中的 `C128A` 与 RoPE 推导仍作为二手实现线索保留。
- `m=4`、`m'=128` 是当前官方文档默认值，不应外推为所有 V4 checkpoint 或后续版本的固定配置。
- 具体 Indexer、compressor、RoPE 与 cache tensor shape 仍应绑定 checkpoint config 和实现版本。
