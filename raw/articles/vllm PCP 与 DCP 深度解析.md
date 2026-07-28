---
title: "vllm PCP 与 DCP 深度解析"
source: "https://zhuanlan.zhihu.com/p/2032220302676063866"
author:
  - "[[小力龙虾​]]"
published:
created: 2026-07-26
description: "1. 背景：为什么需要上下文并行？1.1 LLM 推理的两阶段LLM 推理分为两个阶段，分别为Prefill 阶段和Decode 阶段。 这两个阶段的计算特性截然不同： ┌─────────────────────────────…"
tags:
  - "clippings"
---
[收录于 · vllm代码详细走读](https://www.zhihu.com/column/c_1991807585247064109)

38 人赞同了该文章

目录

收起

1.1 LLM 推理的两阶段

1.2 长上下文的挑战

1.3 传统并行策略的局限

1.4 Chunked Prefill的局限性

2\. 核心概念：上下文并行（Context Parallel, CP）

3\. DCP — Decode Context Parallel

3.1 要解决的问题

3.2 KV Cache 的二维布局（TP × DCP）

3.3 为什么 DCP 能省显存？

3.4 DCP 的通信模式

3.5 DCP 的适用场景

3.6 DCP 算法总结

3.7 DCP 的演进路线

4\. PCP — Prefill Context Parallel

4.1 要解决的问题

4.2 PCP 的核心思想

4.3 PCP 的两大核心算法

算法一：Ring Attention（环状注意力）

算法二：DeepSpeed Ulysses（All-to-All 方式）

4.4 Ring Attention vs Ulysses 对比

4.5 PCP 的负载均衡

5\. PCP 与 DCP 的对比总结

5.1 一张表看懂所有区别

5.2 两者如何协同

5.3 实际使用建议（vLLM 官方）

6\. vLLM 中的实现

6.1 核心代码位置

6.2 配置参数

6.3 DCP 的实现流程（Decode 阶段）

6.4 PCP 的实现流程（Prefill 阶段，RFC 阶段）

6.5 MLA 模型的 DCP 特殊处理

7\. 完整使用示例

7.1 启动 DCP（Decode Context Parallel）

7.2 未来 PCP + DCP 联合使用

8\. 总结

8.1 一张图看懂 CP 全貌

8.2 核心结论

参考资源

**1\. 背景：为什么需要上下文并行？**

### 1.1 LLM 推理的两阶段

LLM 推理分为两个阶段，分别为Prefill 阶段和 [Decode 阶段](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=Decode+%E9%98%B6%E6%AE%B5&zhida_source=entity) 。

这两个阶段的计算特性截然不同：

```
┌─────────────────────────────────────────────────────────────────┐
│  Prefill 阶段（计算密集型 / Compute-bound）                      │
│  • 输入整个 prompt（可能是 32K~128K tokens）                      │
│  • 计算量 = O(L²)，L 越大计算量越大（Attention 全量计算）         │
│  • 目标：降低 TTFT（Time To First Token，首 token 延迟）         │
│  • 单次计算密集，内存访问相对较少                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Decode 阶段（内存密集型 / Memory-bound）                        │
│  • 每次生成 1 个新 token，需要访问全部历史 KV Cache              │
│  • 计算量 = O(L)，但内存访问量极大（完整读取 KV Cache）           │
│  • 目标：提升 throughput（吞吐量），降低每 token 延迟            │
│  • 生成阶段通常占 80%+ 的总推理时间                              │
└─────────────────────────────────────────────────────────────────┘
```
- Prefill 阶段一次性处理所有输入token，自注意力计算的复杂度为O(n^2)，属于计算密集型（compute-bound）任务——在H100 SXM上GPU利用率可达90%-95%，Tensor Core持续满载运行。此阶段的 [性能瓶颈](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E6%80%A7%E8%83%BD%E7%93%B6%E9%A2%88&zhida_source=entity) 在于GPU的 [浮点运算](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E6%B5%AE%E7%82%B9%E8%BF%90%E7%AE%97&zhida_source=entity) 能力（FLOPS），核心优化目标是控制TTFT。
- Decode 阶段则逐个生成token，每次迭代仅需O(n)的注意力计算和O(1)的 [线性层](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E7%BA%BF%E6%80%A7%E5%B1%82&zhida_source=entity) 计算，但需要从HBM反复读取完整的KV Cache，算术强度仅为60-80 ops/ [byte](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=byte&zhida_source=entity) ，GPU利用率降至20%-40%。此阶段属于 [内存带宽](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E5%86%85%E5%AD%98%E5%B8%A6%E5%AE%BD&zhida_source=entity) 密集型（memory-bandwidth-bound）任务，核心优化目标是降低TPOT（Time Per Output Token）和提升 [吞吐量](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=2&q=%E5%90%9E%E5%90%90%E9%87%8F&zhida_source=entity) 。

### 1.2 长上下文的挑战

| 维度 | Prefill | Decode |
| --- | --- | --- |
| 计算复杂度 | O(n^2) 注意力 + O(n) 线性层 | O(n) 注意力 + O(1) 线性层 |
| 核心操作 | 矩阵- [矩阵乘法](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95&zhida_source=entity) （GEMM） | 矩阵- [向量乘法](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E5%90%91%E9%87%8F%E4%B9%98%E6%B3%95&zhida_source=entity) （GEMV） |
| 瓶颈类型 | 计算密集型（Compute-bound） | 内存带宽密集型（Memory-bandwidth-bound） |
| KV Cache 显存 | 每个 Layer 存储完整 K/V | 随序列长度线性增长 |
| 单卡极限 | 约 32K~64K tokens（受显存限制） | 约 128K tokens（受 KV Cache 限制） |
| GPU利用率 | 90-95% | 20-40% |
| [算术强度](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=2&q=%E7%AE%97%E6%9C%AF%E5%BC%BA%E5%BA%A6&zhida_source=entity) | 200-400 ops/byte | 60-80 ops/byte |
| 关键SLO | TTFT（首token延迟） | TPOT/吞吐量 |
| 批处理需求 | 单序列即可饱和GPU | 需数十至数百并发请求才能饱和带宽 |
| 内存访问模式 | 高数据复用率，权重加载一次 | 每步重复读取全部KV Cache |
| [性能优化](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96&zhida_source=entity) 方向 | 分散FLOPS需求（PCP） | 分散KV Cache存储与带宽（DCP） |

随着 [大型语言模型](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B&zhida_source=entity) （LLM）向超长上下文方向发展——从 128K 到 1M 甚至更长——推理系统面临着两个截然不同但同样严峻的瓶颈。

第一个瓶颈出现在 Prefill（预填充）阶段：处理超长输入序列时，自 [注意力机制](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&zhida_source=entity) 的计算复杂度为 O(n2)， [序列长度](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=2&q=%E5%BA%8F%E5%88%97%E9%95%BF%E5%BA%A6&zhida_source=entity) 翻倍意味着计算量翻四倍。 对于 1M token 的输入，单次 Prefill 的计算量可能需要数十秒甚至数分钟，导致首 token延迟（TTFT）严重超标，用户体验极差。 第二个瓶颈出现在 Decode（解码）阶段：每生成一个新 token，都需要访问所有历史token 的 KV Cache。对于 1M token 的上下文，KV Cache 的显存占用可能达到数十甚至上百GB，远超单张 GPU 的 [显存容量](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E6%98%BE%E5%AD%98%E5%AE%B9%E9%87%8F&zhida_source=entity) 。

预填充受限于GPU的计算峰值（FLOPS），解码受限于HBM的读写带宽。这一差异是PD分离（Prefill-Decode Disaggregation）架构的理论基础，也是PCP和DCP分别独立设计的根本原因。

### 1.3 传统并行策略的局限

| 并行策略 | 切分维度 | 上下文并行的问题 |
| --- | --- | --- |
| [Tensor Parallel](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=Tensor+Parallel&zhida_source=entity) （TP） | 模型权重按 Head 维度切分 | KV Cache 的 K/V Head 数量少，切分深度有限；且 TP 切分后 KV Cache 重复存储在所有 TP rank 上 |
| [Pipeline Parallel](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=Pipeline+Parallel&zhida_source=entity) （PP） | 按 Layer 切分 | 切的是 [模型层数](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E6%A8%A1%E5%9E%8B%E5%B1%82%E6%95%B0&zhida_source=entity) ，不是 [序列维度](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E5%BA%8F%E5%88%97%E7%BB%B4%E5%BA%A6&zhida_source=entity) |
| [Data Parallel](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=Data+Parallel&zhida_source=entity) （DP） | 相同 [权重](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=3&q=%E6%9D%83%E9%87%8D&zhida_source=entity) 复制多份 | 每个实例都要存完整 KV Cache，内存浪费严重 |

> **核心矛盾** ：TP、PP、DP 都无法解决 **序列维度（Sequence Dimension）上的 KV Cache 爆炸问题** ，这正是 Context Parallel 要解决的核心问题。

在 PCP 和 DCP 出现之前，LLM 推理系统主要依赖三种并行策略： [张量](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E5%BC%A0%E9%87%8F&zhida_source=entity) 并行（TP）、流水线并行（PP）和数据并行（DP）。然而，这三种策略都无法有效解决长上下文推理的双重瓶颈。

- 张量并行（TP）将模型权重按维度切分到多张 GPU 上，每张 GPU 只存储 1/tp\_size 的权重。然而，在 [标准注意力机制](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E6%A0%87%E5%87%86%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&zhida_source=entity) 中，每个 token 的注意力计算都需要访问所有历史 token 的KV Cache，因此每张 GPU 仍然需要存储完整的 KV Cache。TP 只减少了权重 [显存](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=5&q=%E6%98%BE%E5%AD%98&zhida_source=entity) ，没有减少KV Cache 显存。对于 GQA（Grouped Query Attention）模型，虽然 KV head 数量少于 Qhead，但每张 GPU 上的 KV Cache 仍然是完整序列的副本。
- [流水线并行](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=2&q=%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%B9%B6%E8%A1%8C&zhida_source=entity) （PP）将模型的不同层分配到不同 GPU 上，形成流水线执行。PP 可以减少单张 GPU 的显存占用（包括权重和 KV Cache），但引入了流水线气泡（pipeline bubble），导致 GPU 利用率下降。更重要的是，PP 的延迟是所有阶段延迟的累加，对于延迟敏感的在线服务场景并不理想。
- [数据并行](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=2&q=%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C&zhida_source=entity) （DP）在多个 GPU 上复制完整模型，每个 GPU 处理不同的请求。DP 可以提高吞吐量，但对单个请求的延迟没有任何改善——每个请求仍然由单个 GPU 独立处理，无法利用多GPU 的计算或存储能力来加速单个长上下文请求。

### 1.4 Chunked Prefill的局限性

Chunked Prefill是长序列推理中广泛使用的 [内存优化](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E5%86%85%E5%AD%98%E4%BC%98%E5%8C%96&zhida_source=entity) 技术，将长提示分成固定大小的块（chunk），将内存复杂度从O(L^2)降低到O(LK)（其中K为 [chunk大小](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=chunk%E5%A4%A7%E5%B0%8F&zhida_source=entity) ）。但该技术存在一个根本性缺陷： **每个块是串行计算的** ，单个请求无法充分利用多GPU环境的并发优势。

如vLLM RFC #22693所指出的，“Each chunk is computed serially, so a single request fails to fully leverage the advantages of concurrency in a multi-GPU environment”。

换言之，Chunked Prefill解决了 [内存溢出](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E5%86%85%E5%AD%98%E6%BA%A2%E5%87%BA&zhida_source=entity) 问题，但并未从根本上加速单个长请求的预填充——它是在 **[时间维度](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E6%97%B6%E9%97%B4%E7%BB%B4%E5%BA%A6&zhida_source=entity)** 上分块，而非在 **空间维度** 上并行。

## 2\. 核心概念：上下文并行（Context Parallel, CP）

**上下文并行** 的本质是： **沿序列维度（Sequence Dimension）切分，将 KV Cache 分布到多个 GPU 上，实现真正的跨设备长序列支持。**

```
传统方式（单卡）：
Sequence: [token_1, token_2, ..., token_L]
KV Cache: ┌─────────────────────────────────┐
          │    所有 Layer 的全部 K/V 存于 1 卡 │
          └─────────────────────────────────┘
​
上下文并行（多卡）：
Sequence: [tok_1..tok_n] [tok_n+1..tok_2n] [tok_2n+1..tok_3n] ...
           ↓                 ↓                 ↓
KV Cache: ┌──────┐  ┌──────┐  ┌──────┐
          │ GPU 0 │  │ GPU 1 │  │ GPU 2 │  ...
          └──────┘  └──────┘  └──────┘
          存储 1/3    存储 1/3    存储 1/3
```

上下文并行（Context Parallelism, CP）的核心思想是将序列维度（sequence dimension）切分到多个 GPU 上，使得每个 GPU 只需处理序列的一部分，从而同时缓解计算和显存瓶颈。

## 3\. DCP — Decode Context Parallel

### 3.1 要解决的问题

DCP 解决的是 **Decode 阶段的 KV Cache 显存爆炸问题** 。

在 Decode 阶段，每个新 token 都要读取 **整个历史序列的 KV Cache** 。对于超长上下文（比如 128K tokens），KV Cache 的显存占用会迅速打满 GPU 显存，导致无法服务。

**DCP 的核心思路** ：把 KV Cache 按序列维度 **interleave 切分** 到多个 GPU 上，每个 GPU 只存储一部分 KV Cache 大幅降低 [单卡显存](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E5%8D%95%E5%8D%A1%E6%98%BE%E5%AD%98&zhida_source=entity) 压力。

**什么是interleave 切分？**

例如上下文长度 T = 16，DCP size = 4。

```
DCP rank0 存 token 0, 4, 8, 12 的 KV
DCP rank1 存 token 1, 5, 9, 13 的 KV
DCP rank2 存 token 2, 6, 10, 14 的 KV
DCP rank3 存 token 3, 7, 11, 15 的 KV
```

连续切：

```
rank0: token 0~255
rank1: token 256~511
rank2: token 512~767
rank3: token 768~1023
```

DCP 更适合 interleaving，原因是 decode 的 KV cache 会持续增长。

每生成一个新 token，就要把它的 KV 写入某个 rank。

如果用 interleaving：

```
token i -> rank i % dcp_world_size
```

那么未来新 token 会自然轮转写到不同 rank：

```
t=0 -> rank0 t=1 -> rank1 t=2 -> rank2 t=3 -> rank3 t=4 -> rank0
```

这样每个 rank 的 KV cache 增长速度天然均衡。

如果用连续切分，就需要不断判断当前 token 属于哪个区间，区间还会随上下文增长动态变化，维护成本更高。

所以 interleaving 的本质是：

```
让动态增长的 decode KV cache 按时间自然均衡分布到 DCP ranks
```

### 3.2 KV Cache 的二维布局（TP × DCP）

DCP 巧妙利用了 **KV Head 数量** 来设计 KV Cache 的二维切分：

```
Key/Value Tensor 维度: [num_tokens, num_kv_heads, head_dim]
​
在 TP 切分时：
  TP 对 Q/K/V 按 num_heads 维度切分
  → 切分后，每个 TP rank 有完整的 num_tokens
  → 但 KV Cache 在 Head 维度被切分了
​
在 DCP 叠加时：
  DCP 进一步对 num_tokens 维度切分
  → 每个 DCP rank 只存储部分序列位置的 KV
  → 形成二维布局：[token_chunk, kv_head] → [tp_dim, dcp_dim]
```

Twitter/X 上的 vLLM 官方说明精炼地总结了这个设计：

> *"1. TP shards KV cache along kv-heads (H dimension)"* *"2. When kv-heads < tp\_size, TP parallelism on KV is limited"* *"3. DCP shards KV cache along token dimension → forming 2D layout: TP/DCP shards heads, DCP shards tokens"*

### 3.3 为什么 DCP 能省显存？

```
假设：num_kv_heads = 8, head_dim = 128, max_seq_len = 128K, dtype = fp8
​
单卡 KV Cache 显存：
  = 128K × 8 × 128 × 2(K+V) × 1 byte ≈ 256 MB / layer
  × 80 layers = 约 20 GB / 实例
​
使用 DCP=2（2卡）：
  每卡只存 64K tokens → KV Cache 减半 = 10 GB / 实例
​
使用 DCP=4（4卡）：
  每卡只存 32K tokens → KV Cache 再次减半 = 5 GB / 实例
```

### 3.4 DCP 的通信模式

```
Decode 阶段单步 Attention 计算（每层）：
  
  Query（来自本 GPU）: [1 token, num_q_heads, head_dim]
  Key（来自所有 DCP rank）: [L tokens, num_kv_heads, head_dim]
  Value（来自所有 DCP rank）: [L tokens, num_kv_heads, head_dim]
  
  每个 DCP rank 需要：
  1. 本地读取本地 KV chunk → 计算局部 attention score
  2. All-Gather：把其他 rank 的 KV 拿到手（因为 Query 要attend所有历史 K）
     或者 Reduce-Scatter（取决于通信后端）
  3. 合并得到完整的 attention 结果
```

vLLM 支持两种通信后端：

| 通信后端 | 全称 | 特点 |
| --- | --- | --- |
| ag\_rs（默认） | AllGather + ReduceScatter | 先 AllGather 收集全部 KV，再计算，再 ReduceScatter 分发结果 |
| a2a | All-to-All | 单次集合通信，通信量更少，但需要硬件支持（InfiniBand等） |

### 3.5 DCP 的适用场景

DCP 最适合的场景： **[长序列](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=3&q=%E9%95%BF%E5%BA%8F%E5%88%97&zhida_source=entity) Decode + KV Head 数量有限** 。

- ✅ GQA（Grouped Query Attention）模型（如 Llama、Qwen）：KV Head 少，DCP 效果显著
- ✅ MLA（Multi-head Latent Attention）模型（如 [DeepSeek](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=DeepSeek&zhida_source=entity) ）：低秩压缩后的 KV Head 更少
- ❌ MHA（Multi-head Attention）模型：KV Head 数量与 Q Head 相同，通常 TP 就够了

**为什么 DCP 主要适合 GQA/MQA/MLA**

DCP 对什么模型最有价值？

答案是：KV heads 很少的模型。

TP沿attention head维度切分计算，每个GPU负责一部分heads。当tp\_size <= num\_kv\_heads时，KV Cache可以自然地随head维度切分，不产生冗余。然而，一旦tp\_size > num\_kv\_heads，TP无法切分比GPU数量更少的KV heads，导致每个GPU不得不存储一份完整的KV Cache副本。此时KV Cache的重复倍数恰好等于tp\_size / num\_kv\_heads。

| 模型 | Attention\*\*类型\*\* | KV Heads | TP=8\*\*重复\*\* | TP=16\*\*重复\*\* |
| --- | --- | --- | --- | --- |
| DeepSeek-V2/R1 | MLA | 1 (effective) | 8x | 16x |
| Llama-3 70B/405B | GQA | 8 | 1x | 2x |
| Qwen3-235B-A22B | GQA | 4 | 2x | 4x |

MLA（Multi-head Latent Attention）架构是其中的重灾区。MLA通过将K/V压缩到低秩latent空间来减少KV Cache footprint，其有效KV head数仅为1。当DeepSeek-R1以单节点TP=8部署时，每个GPU都复制了完整的KV Cache，产生高达8倍的冗余。这种重复不是浪费一点内存的问题——它直接导致可服务的batch size缩减为原来的1/8，吞吐量随之断崖式下降。

vLLM 文档里的例子也类似：

- DeepSeek-R1 MLA 场景，1 个 KV head，TP=8 会导致 8x KV duplication，可以考虑 DCP=8。
- Qwen3-235B-A22B 有 4 个 KV heads，TP=8 时 duplication 是 2x，可以用 DCP=2 去掉重复。

### 3.6 DCP 算法总结

```
DCP 算法（Decode 阶段）：

输入：Query[q_token], KV Cache 分布在 DCP 个 rank 上
输出：Attention output

1. 各 DCP rank 持有本地 KV chunk: K_local, V_local（各自负责部分序列）

2. 本地计算（并行）：
   Score_local = Q × K_local^T  →  形状 [1, 1] × [seq_chunk, head_dim]
   （每个 rank 只计算对自己 KV chunk 的注意力分数）

3. 通信同步：
   - AG_RS 后端：AllGather 所有 rank 的 K/V → 各 rank 得到完整 K/V
     → 计算完整 attention output → ReduceScatter 分发结果
   - A2A 后端：All-to-All 直接交换 K/V

4. 最终：各 rank 得到相同结果的均匀分片（或完整结果）
```

**本地 attention 计算** 每个 DCP rank 先独立计算局部 attention：

```
scores_i = Q @ K_i^T
local_lse_i = logsumexp(scores_i)
local_out_i = softmax(scores_i) @ V_i
```

这里最关键的是： **local\_out\_i 不能直接相加** 。必须通信 LSE。

局 softmax 需要：

```
global_lse = logsumexp(local_lse_0, local_lse_1, local_lse_2, local_lse_3)
```

然后每个 rank 的局部输出要重新缩放：

```
corrected_out_i = exp(local_lse_i - global_lse) * local_out_i
```

最后：

```
global_out = sum(corrected_out_i)
```

这就是 DCP 通信的数学本质。

DCP的通信过程：

1. AllGather + ReduceScatter()
```
tep 1: 每个 rank 用本地 KV shard 计算 local_out 和 local_lse

Step 2: DCP group 内 AllGather local_lse
        每个 rank 拿到所有 rank 的 lse:
        [lse_0, lse_1, lse_2, lse_3]

Step 3: 每个 rank 计算 global_lse
        global_lse = logsumexp(lse_0, lse_1, lse_2, lse_3)

Step 4: 每个 rank 修正自己的 local_out
        corrected_out_i = exp(local_lse_i - global_lse) * local_out_i

Step 5: DCP group 内 ReduceScatter / Reduce
        把 corrected_out_i 汇总成最终 attention output
```

All-to-All:

```
Step 1: 本地 KV shard attention
        得到 local_out_i 和 local_lse_i
​
Step 2: All-to-All 交换 partial outputs 和 LSE
        让需要合并的 rank 收到对应 shard 的 out/lse
​
Step 3: 本地 Triton kernel 做 LSE 修正和输出合并
```

官方配置说明里提到，a2a 可以把 MLA 模型每层 NCCL calls 从 3 次减少到 2 次。

所以可以这样理解：

```
ag_rs:  更传统，AllGather LSE + output reduction 
a2a:  把 partial output 和 LSE 一起交换，再本地融合合并  通信次数更少，但实现更依赖特定 backend/kernel
```

完整通信时序 以 GQA + ag\_rs 为例，可以写成完整时序：

```
输入:
  rank0 has KV_0
  rank1 has KV_1
  rank2 has KV_2
  rank3 has KV_3
  each rank has partial Q heads
​
Step A: Q all-gather
  rank0..3 exchange Q heads
  each rank gets Q needed for local attention
​
Step B: local attention
  rank0 computes Q x KV_0 -> out_0, lse_0
  rank1 computes Q x KV_1 -> out_1, lse_1
  rank2 computes Q x KV_2 -> out_2, lse_2
  rank3 computes Q x KV_3 -> out_3, lse_3
​
Step C: LSE all-gather
  all ranks exchange lse_i
​
Step D: global LSE correction
  each rank computes:
  global_lse = logsumexp(lse_0, lse_1, lse_2, lse_3)
​
  then rescales:
  out_i = exp(lse_i - global_lse) * out_i
​
Step E: output reduce-scatter
  corrected out_i are summed and scattered
  each rank gets the final output shard it should own
​
Step F: continue transformer layer
  projection / residual / MLP / next layer
​
```

如果是 a2a，Step C 到 Step E 会改成：

```
All-to-All exchange local_out + local_lse local fused kernel combines outputs with LSE correction
```

DCP 的 [第一性](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E7%AC%AC%E4%B8%80%E6%80%A7&zhida_source=entity) 原理链条是：

```
decode 每步 query 很少，但需要读取全部历史 KV
长上下文和高并发让 KV cache 成为显存瓶颈
TP 可以按 KV head 切 KV，但 GQA/MQA/MLA 的 KV heads 很少
TP size 大于 KV heads 时，KV cache 会在多个 TP ranks 上重复
DCP 在 TP 域内沿 sequence/context 维度继续切 KV cache
interleaving 让 decode 新增 KV 自然均衡写入各 rank
attention 每个 rank 只看 local KV shard
为了得到全局正确结果，需要 LSE allgather + output correction + allreduce
最终用通信开销换 KV cache 容量，从而支持更长上下文或更大 batch
```

所以，vLLM 中 DCP 的本质是：

```
用 sequence 维度的 KV cache 分片，消除 GQA/MLA/MQA 场景下 TP 带来的 KV 重复； 用分布式 softmax 合并保证 attention 数学等价； 用更多通信换更多 KV cache 容量和更高 decode batch throughput。
```

参考：

- [vLLM Context Parallel Deployment](https://link.zhihu.com/?target=https%3A//docs.vllm.ai/en/latest/serving/context_parallel_deployment/)
- [vLLM RFC: Decode Context Parallel for GQA](https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/issues/24685)
- [vLLM Ascend Context Parallel Guide](https://link.zhihu.com/?target=https%3A//docs.vllm.ai/projects/ascend/en/v0.13.0/developer_guide/feature_guide/context_parallel.html)

### 3.7 DCP 的演进路线

DCP 在 vLLM 中的实现经历了三个阶段的演进：

- 阶段一（PR #23734）：MLA 模型支持。DCP 首先在 FlashMLA 后端中实现，支持DeepSeek-V2/V3 等 MLA 模型。MLA 模型的 KV Cache 已经被压缩到低维隐空间，DCP 的交错分片实现相对简单。
- 阶段二（PR #24864, #25438）：GQA 模型支持。DCP 扩展到 GQA 模型，分别在FlashAttention 和 FlashInfer 后端中实现。GQA 模型的 KV Cache 维度更高，通信量更大，需要更精细的通信优化。
- 阶段三（PR #25132）：Triton 后端支持。DCP 进一步扩展到 Triton 后端，提供更灵活的定制能力。

## 4\. PCP — Prefill Context Parallel

### 4.1 要解决的问题

PCP 解决的是 **Prefill 阶段的全量 Attention 计算瓶颈** 。

Prefill 阶段的核心是：长度为 L 的 prompt 要计算一次完整的 O(L²) Attention 矩阵——这是一次 **计算密集型** 的操作，不像 Decode 那样是内存瓶颈。

对于超长输入序列（如 1M token），Prefill 阶段需要计算序列中每个 token 对所有前序 token 的注意力，计算复杂度为O(n2)。这意味着序列长度翻倍，Prefill 计算量翻四倍。在单 GPU 上，1M token 的Prefill 可能需要数十秒甚至数分钟，导致首 token 延迟（TTFT）严重超标。

即使 GPU 有足够的显存存储完整的 KV Cache，Prefill 的计算量仍然可能超出单 GPU 的 [计算能力](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E8%AE%A1%E7%AE%97%E8%83%BD%E5%8A%9B&zhida_source=entity) 。传统的 TP 可以将计算分散到多个 GPU 上，但 TP 切分的是 head 维度，每个 GPU 仍然需要处理完整的序列长度。对于超长序列，单 GPU 处理完整序列的 Prefill 计算量仍然过大。

当 L 达到 32K~128K 时, 问题总结：

- 单个 GPU 的算力不足以在合理时间内完成计算
- 单个 GPU 的显存也无法容纳这么大的 KV Cache
- 必须沿序列维度切分，让多个 GPU 并行完成 Prefill

### 4.2 PCP 的核心思想

PCP 在 Prefill 阶段 **沿序列维度切分 Q/K/V** ，让多个 GPU [并行计算](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97&zhida_source=entity) Attention。与 DCP 不同，PCP 需要跨 GPU 交换中间结果， **因为 Attention 的本质是"每个 Query 都要看所有 Keys"** 。

### 4.3 PCP 的两大核心算法

Ulysses: 是“换切分维度，让 attention 本地化”； Ring Attention: 是“保持 sequence 切分不变，让 KV 沿环流动，逐块完成精确 attention”。

### 算法一：Ring Attention（环状注意力）

由 **Liu et al.（Berkeley AI Research, 2023）** 提出，是 PCP 最主流的实现方式。

**适用场景：** 序列长度超长（无法在每个GPU上容纳完整KV张量，如百万token级别）。

![](https://pic1.zhimg.com/v2-f047b63fa39fc8247ec0e5f2ba102e8c_1440w.jpg)

**核心思想** ：将 Q 和 KV 都按序列维度切分成 P 块（P = GPU 数量），然后让每块 KV 在 **环形拓扑** 上流动，每一步：

1. 每个GPU计算自己chunk Q 块和 K/V 块
2. 计算局部 Attention
3. 将 KV 块传给下一个 rank，同时接收下一个 K/V 块
4. 重复直到 Q 块遍历完所有 K/V 块
5. 最终累加所有局部结果得到完整 Attention

**Ring Attention 的本质** ： 把序列切到多张卡上，但不去一次性收集全序列 KV，而是让 KV 分块沿环形拓扑逐站流动；每张卡只保留本地 Q，边接收一块 KV 边做一块 attention，并用在线 softmax 把这些分块结果精确累积成最终输出。

| 特性 | 说明 |
| --- | --- |
| 通信模式 | 点对点（P2P），每步只传 KV 块 |
| 通信量 | O(L/P × head\_dim × P) = O(L × head\_dim) 总量不变，但分 P 次 |
| 优点 | 通信可与计算重叠，带宽利用率高 |
| 缺点 | P 步延迟累积，P 越大延迟越高 |

从第一性原理出发讲解：

**1\. “直接切序列”会遇到什么问题？**

Transformer 训练里，计算可以粗略按四个维度看：

- batch 维：B
- hidden 维：H
- layer 深度：L
- sequence 长度：S

传统并行主要覆盖前三个：

- Data Parallel 解决 batch
- Tensor Parallel 解决 hidden
- Pipeline Parallel 解决 layer

但 **超长上下文** 的根本压力来自 S。 尤其 attention，复杂度和显存都强依赖序列长度：

- attention score 大小约是 O(S^2)
- KV / activation 也随 S 线性增长

所以当 S 很大时，单卡先爆的通常不是参数，而是 **激活与 attention 中间张量** 。

为什么不能直接“把序列切开，各卡各算各的”？

- MLP、LayerNorm、残差这些基本是 **逐 token 局部** 操作
- 但 self-attention 是 **跨整个序列全局耦合** 操作

所以如果你把序列切成 P 份，每张卡只保留 S/P 个 token，那么：

- MLP 仍然能本地算
- 但 attention 不行，因为每张卡缺了其余 P-1 份 token 的 K/V
![](https://pic4.zhimg.com/v2-da277d6411c25d3acc50a57dca87995f_1440w.jpg)

对于attention算子， *O* =softmax(QK⊤) *V* 。每个 query token 都依赖所有 key/value token，attention score 天然是 O(S^2)，直接切序列后：

1. 假设有 P 张卡，把序列均匀切开，第 r 张卡只保存 S/P 个 token。
2. 因为某张卡上的本地 query 块，要和 **全序列** 的 KV 交互。
3. attention 这里立刻遇到一个选择：
1. 把所有 KV 一次性 all-gather 到每张卡
	2. 不要一次性 gather，而是让 KV 分块流过来

Ring Attention 选择的是第 2 条路。

对于Ring Attention，有如下两个洞察：

**洞察 A：attention 可以按 KV block 分块计算：**

对某个 query 块 Q\_i 来说，最终输出是对所有 key/value 块贡献的组合：

![](https://pic1.zhimg.com/v2-2edaf8508a42abd89a3f25f4ca49eca6_1440w.jpg)

也就是说，虽然最终 softmax 是“全局的”， 但 logits 可以按块生成：

- 先算 Q\_i 对 K\_1
- 再算 Q\_i 对 K\_2
- ...
- 最后再合成

**洞察 B：softmax 可以在线精确合并**

给定一行 logits 的多个块，假设我们维护：

- 运行中的最大值 m
- 运行中的归一化分母 l
- 运行中的加权和 a

当新块到来时，可以用 **在线 softmax** 精确更新，而不是近似。

**Ring Attention 的基本执行流程**

假设我们有 P 张卡，组成一个 ring。 整段序列被切成 P 个 sequence chunk：

X=\[X1,X2,...,XP\]

每张卡 r 初始保存：

- 本地 query 块 Q\_r
- 本地 key 块 K\_r
- 本地 value 块 V\_r

形状大致都是：(S/P)×d

**前向过程**

对第 r 张卡来说，目标是求出本地 query 块 Q\_r 的最终输出 O\_r。但 Q\_r 需要看完整序列的所有 KV，于是采用 ring 方式：

第 0 轮

- 卡 r 用本地 Q\_r 对本地 K\_r, V\_r 做 block attention
- 得到第一块部分结果
- 同时把 K\_r, V\_r 发给下一张卡，接收上一张卡传来的 KV

第 1 轮

- 卡 r 用同一个 Q\_r
- 对收到的 K *{r-1}, V* {r-1} 再做一次 block attention
- 把部分结果与上一轮结果做在线 softmax 合并
- 再继续传递 KV

...

第 P-1 轮

- 卡 r 最终已经让 Q\_r 看过所有 P 份 KV
- 得到精确的 O\_r

于是每张卡最终持有：

- 自己那段 query 对应的 attention 输出
- 不需要持有整段序列的完整 KV

这就是“ring”的含义：

```
Q 不动 
KV 沿着设备环流动 
每一站处理当前 KV 块对本地 Q 的贡献
```

Ring Attention的优缺点：

**优点：**

1 省显存

果用最朴素的方法做 context parallel：

- 每张卡先保存本地 sequence chunk
- attention 前 all-gather 全部 KV
- 然后本地算完整 attention

那么 attention 时每张卡都要临时持有 **完整序列 KV**

Ring Attention 不这么干。 它只要求每张卡持有：

- 本地 Q
- 当前轮收到的一块 KV
- 少量在线 softmax 的中间状态

2 通信模式适合长序列

如果 sequence 很长、P 很大，一次性 all-gather 的缺点是：

- 峰值通信突发大
- 需要临时持有完整 KV
- 难以与计算重叠

Ring Attention 把它改成：

```
P 轮小块 point-to-point 传输 
每一轮通信都能和本轮 block attention 计算重叠
```

所以它的系统收益不仅是“能跑更长”，还在于：

- 峰值内存更低
- 通信更平滑
- 更容易 overlap

**缺点：**

它不是一次 all-gather，而是 P 轮左右的 ring 交换。如果网络很差，或者 sequence 不够长，通信轮次开销可能不划算。

**一句话总结：**

**Ring Attention 的第一性原理是：attention 的全局依赖不要求“所有数据同时在场”，只要求“每个 query 最终看过所有 KV”；因此可以把完整 attention 重写成“固定本地 Q、让 KV 分块沿 ring 流动、用在线 softmax 精确累积”的流式计算过程，从而在不改变结果的前提下，把超长序列 attention 变成一种可并行、低峰值显存、可通信计算重叠的序列并行策略。**

参考资料：

- 原始论文： [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2310.01889)
- 前置基础： [Blockwise Parallel Transformer for Long Context Large Models](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2305.19370)

### 算法二：DeepSpeed Ulysses（All-to-All 方式）

由 **DeepSpeed 团队** 提出，采用与 Ring Attention 互补的策略。

**核心思想** ：在 Attention 计算 **之前** 通过 All-to-All 重新分配 Q 和 KV 的分布，在 **局部** 完成完整 Attention。

```
Ulysses 的流程：
​
阶段1：All-to-All（Q 重新分片）
  各 GPU 上的 Q: [L/P, num_heads, head_dim]
  ↓ All-to-All
  各 GPU 上的 Q: [L/P, num_heads_per_gpu, head_dim] × P
  （但实际上 Ulysses 将 Q 按 head 维度重新分配）
​
阶段2：局部 Attention
  每个 GPU 上的 Q_local 和 K_local/V_local 来自不同的原始 token
  → 在本地执行 Flash Attention
  → 每个 GPU 计算了部分 attention output
​
阶段3：All-to-All（结果汇聚）
  将各 GPU 的局部结果汇聚得到完整 output
​
通信模式：2次 All-to-All（Q 重分配 + 结果汇聚）
总通信量：O(L/P × num_heads × head_dim) × 2
```

Ulysses 的本质：

- Transformer 的激活按序列维切分，以降低长上下文训练的显存压力；
- 但在 attention 这一层，临时把“按序列切分”重排成“按头切分”，这样每张卡都能在本地看到完整序列、只负责一部分 heads，做完 attention 后再切回按序列切分。

**核心观察：attention 对“序列”耦合，但对“头”天然可并行**

多头注意力有一个极其重要的结构性质：

- **不同 attention heads 彼此独立**
- 只有在最后 concat / output projection 时才合并

也就是说，attention 的依赖结构是：

- 对单个 head：必须看到完整序列
- 对不同 heads：可以分开算

到了 attention 之前，把分片方式从：按 sequence 切

临时变成：按 heads 切

这样每张卡就能拿到：

- **完整序列**
- 但只拿到 **一部分 heads**

Ulysses 的张量流：一步一步看

设：

- sequence parallel 度为 P
- 总 hidden 为 H
- attention heads 数为 N\_h
- 每头维度为 d\_h = H / N\_h

输入张量形状：x\_shape=\[B, S, H\]

**阶段 A：先按序列切分:**

把 sequence 均匀切到 P 张卡上，每张卡持有：\[B, S/p, H\]

这一步的意义很直接：

- 每张卡只保存 1/P 的 token 激活
- 非 attention 模块都能直接本地算

对长序列来说，这已经把很多激活内存降到 1/P。

本地线性投影得到 Q/K/V:\[B, S, Nh,dh\]

此时每张卡：

- 有完整 heads
- 但只有局部 sequence chunk

问题仍然是：attention 要完整序列。

**阶段 B：第一次 all-to-all，把“按序列切分”变成“按头切分”**

经过 all-to-all 之后，每张卡拿到：Qr,Kr,Vr=\[B, S, Nh/p, dh\]

这意味着：

- sequence 维从局部 S/P 变成完整 S
- heads 维从完整 N\_h 变成局部 N\_h/P

换句话说：

```
重排前：我有一小段序列的所有 heads 
重排后：我有完整序列的一小部分 heads
```

这一步不是 gather 全量张量，而是 **结构化 all-to-all 重排** ，因此通信效率高于很多“先 gather 全序列再算”的做法。

**阶段 C：每张卡本地计算 attention**

现在每张卡已经拿到了：

- 完整序列
- 自己负责的 N\_h/P 个 heads

这是 Ulysses 的核心收益点：

- **attention 不再需要跨卡协同算**
- attention kernel 本地完成
- 可以直接结合 FlashAttention 等高效本地 attention kernel

**阶段 D：第二次 all-to-all，切回 sequence-sharded 布局**

attention 算完之后，还要回到后续模块喜欢的 sequence-sharded 布局。

于是第二次 all-to-all 后，每张卡得到：Or=\[B, S/p, Nh, dh\]

也就是：

- 又变回局部序列 chunk
- 重新拥有完整 heads

这样后续的：

- output projection
- residual
- MLP
- LayerNorm

都能继续在 sequence-sharded 布局下本地执行。

**Ulysses 的优势**

显存到底省在哪里

很多人第一次看 Ulysses 会有个误解：

> 既然每张卡在 attention 前又拿到了完整序列，那显存不是又回去了？

关键在于： **回来的只是“完整序列的一部分 heads”** ，而不是“完整序列的全部 heads”。

输入/激活层面： *O* (*B* ⋅ *S* ⋅ *H*) 降到 *O* (*B* ⋅ *S* ⋅ *H* /P)

attention 计算层面: 变成 *O* (*B* ⋅Nh/p⋅\*S2)

### 4.4 Ring Attention vs Ulysses 对比

| 维度 | Ring Attention | DeepSpeed Ulysses |
| --- | --- | --- |
| 通信模式 | P2P（点对点，环状） | All-to-All（全收集） |
| 通信次数 | P 步（延迟累积） | 2 次（延迟固定） |
| 通信量（总量） | O(L × head\_dim × P) | O(L × head\_dim × 2) |
| 通信与计算重叠 | ✅ 容易重叠（P2P 可流水线） | ⚠️ 难以重叠（All-to-All 必须完整执行） |
| 适用场景 | P 较大（>4）， [网络带宽](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E7%BD%91%E7%BB%9C%E5%B8%A6%E5%AE%BD&zhida_source=entity) 高 | P 较小（≤4），硬件 All-to-All 强 |
| 硬件要求 | 普通网络（NVLink/TCP） | 需要强 All-to-All（InfiniBand 等） |

> 实践结论： **当 P 较大时 Ring Attention 更优（通信可流水线）；当 P 较小时 Ulysses 更优（通信量小）。** 也有人提出 Hybrid 方案——结合两者优点。

### 4.5 PCP 的负载均衡

Prefill 阶段还需要处理 **变长序列的负载均衡问题** 。vLLM 的 RFC 文档中指出：

- 将序列 **均匀切分** 到每个 CP group
- 每个 CP group 内对应计算 **一致** （保证正确性）
- 结合 Chunked Prefill 机制处理请求间的动态长度变化

Chunked Prefill与PCP是互补而非竞争的关系：

- Chunked Prefill解决的是单个GPU的内存限制——将长序列在时间维度上分块，降低峰值内存从O(L2)到O(c2)（c为chunk size）
- PCP解决的是单个GPU的计算限制——将计算在空间维度上分布到多个GPU

两者的协同工作流程如下：

1. 调度层：vLLM V1 scheduler在token budget内决定哪些prefill chunk参与当前batch
2. PCP层：每个prefill chunk被进一步沿序列维度分片到N个GPU
3. 策略选择：若chunk size可以容纳完整KV，使用策略1（AllGather）；若chunk size太大，使用策略2（Ring Attention） 这种协同使得超长序列（1M+ token）场景可以同时受益于内存优化（Chunked Prefill）和并行加速（PCP + Ring Attention）

## 5\. PCP 与 DCP 的对比总结

### 5.1 一张表看懂所有区别

| 维度 | PCP（Prefill Context Parallel） | DCP（Decode Context Parallel） |
| --- | --- | --- |
| 作用阶段 | Prefill（计算密集型） | Decode（内存密集型） |
| 核心目标 | 降低 TTFT（首 token 延迟） | 降低 KV Cache 显存占用 + 提升吞吐 |
| 切分对象 | Q、K、V 全部沿序列维度切分 | 仅 KV Cache 沿序列维度切分 |
| Q 的位置 | 各 rank 只有部分 Q（按序列切分） | 各 rank 有当前 decode token 的完整 Q |
| Attention 计算 | 需要 P 步 Ring AllReduce 或 All-to-All 汇聚 | 各 rank 本地计算局部 attention，然后聚合 |
| 主要算法 | Ring Attention / DeepSpeed Ulysses | Interleaved KV Cache + AllGather/A2A |
| 通信模式 | 点对点环状（P2P）或 All-to-All | AllGather + ReduceScatter 或 All-to-All |
| 上线状态 | vLLM v0.18+（RFC 阶段，2025年9月提出） | vLLM 已稳定支持（2025年8月已有） |
| 与 TP 关系 | 可与 TP 叠加，扩大并行度 | 将 TP 切分限制在 KV Head 维度内，余下 GPU 给 DCP |
| 典型配置 | \--cp 2 --tp 4（4卡中 2 卡 PCP，余下 2 卡） | \--dcp 2 --tp 4 |

### 5.2 两者如何协同

在 vLLM 中，PCP 和 DCP 可以协同工作，形成完整的 CP 策略：

```
总 GPU 数 = TP × CP（CP = PCP × DCP）
​
假设 8 卡配置：TP=4, DCP=2
  → TP group 内：4 卡做 Tensor Parallel（切 Q/K/V head）
  → TP group 间：DCP=2 意味着用 2 个 TP group 的 GPU 共享 KV Cache
  → 总共支持 2× 的上下文长度
​
假设 8 卡配置：TP=4, PCP=2  
  → Prefill 时：2 组 TP group 并行处理不同序列段
  → Decode 时：DCP 进一步减小 KV Cache 重复
```

### 5.3 实际使用建议（vLLM 官方）

来自 vLLM 官方文档（2026年4月）：

> **For Decode Context Parallel**: try to increase `-tp` size until you get satisfactory performance, and then add `-dcp` to reduce the KV cache duplication.

```
推荐使用顺序：
1. 先增大 TP size → 直到性能满意
2. 如果显存仍不够 → 添加 --dcp 减少 KV Cache 重复
3. 如果 Prefill TTFT 仍太长 → 未来使用 PCP（PCP 功能仍在 RFC 阶段）
```

## 6\. vLLM 中的实现

### 6.1 核心代码位置

```
vllm/
├── vllm/config/parallel.py              ← 并行配置（CP/DCP/PCP 参数定义）
│   └── Config 中定义 context_parallel_size, tp_size 等
│
├── vllm/attention/backends/             ← 各后端的 Attention 实现
│   ├── flashattn/                        ← Flash Attention 后端
│   ├── torch/                            ← PyTorch 原生后端
│   └── roce_attn/                        ← ROCm 后端（AMD GPU）
│   （DCP 在各后端的 attention forward 中实现）
│
├── vllm/v1/attention/backend.py         ← V1 版本 Attention 后端抽象
│
├── vllm/worker/model_runner.py           ← 单卡执行器（处理 CP 切分逻辑）
│
└── vllm/distributed/
    ├── kv_transfer/                      ← KV 跨设备传输
    └── communication/                    ← CP 集合通信原语
```

### 6.2 配置参数

| 参数 | 说明 | 示例值 |
| --- | --- | --- |
| \--tensor-parallel-size / -tp | Tensor Parallel 大小 | 4 |
| \--context-parallel-size / -cp | 上下文并行总大小 | 4 |
| \--decode-context-parallel-size / -dcp | Decode 阶段 CP | 2 |
| \--prefill-context-parallel-size / -pcp | Prefill 阶段 CP（RFC） | 2 |
| \--cp-interleave-size | KV Cache 存储的 interleaved [块大小](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E5%9D%97%E5%A4%A7%E5%B0%8F&zhida_source=entity) | 1 |
| \--dcp-comm-backend | DCP 通信后端 | ag\_rs 或 a2a |

### 6.3 DCP 的实现流程（Decode 阶段）

```
vLLM Decode 阶段（使用 DCP）：
​
model_runner.py（执行器）
    │
    ├── 1. 准备 Query（当前 token 的 Q）
    │   Q: [1, num_q_heads, head_dim]  ← 单 token，完整 Q
    │
    ├── 2. 加载 KV Cache（来自各 DCP rank）
    │   各 rank 持有 K_local: [seq_chunk, num_kv_heads, head_dim]
    │
    ├── 3. 调用 Attention Backend
    │   attention_ops.flash_attn_with_殷（或对应后端）
    │   │
    │   ├── 本地计算：Q × K_local^T → Score_local
    │   │
    │   ├── 通信同步：
    │   │   AllGather(K_all) + AllGather(V_all)
    │   │   计算完整 attention → ReduceScatter
    │   │   
    │   │   或者：A2A 通信直接交换 K/V
    │   │
    │   └── 输出：完整的 attention output
    │
    └── 4. 更新 KV Cache（新生成的 token 的 K/V 存入对应 rank）
```

### 6.4 PCP 的实现流程（Prefill 阶段，RFC 阶段）

```
vLLM Prefill 阶段（使用 PCP，RFC #25749）：
​
1. 序列切分（按 PCP 大小切分 prompt）
   prompt = [tok_1, ..., tok_L]
   → Q_chunks = [Q_0, Q_1, ..., Q_P-1]  （每块 L/P 个 tokens）
   → K_chunks = [K_0, K_1, ..., K_P-1]
   → V_chunks = [V_0, V_1, ..., V_P-1]
​
2. 各 PCP rank 本地计算局部 Attention
   每个 rank 执行本地 Flash Attention：
   → 得到 partial_output[i] = Attention(Q_i, K_all_chunks)
​
3. Ring Attention / All-to-All 汇聚结果
   各 rank 持有部分 Q，计算对应的完整 attention output
   通过集合通信汇聚得到全局结果
​
4. KV Cache 存储
   PCP 将 KV Cache 沿序列切分存入各 rank
   后续 Decode 阶段的 DCP 可以直接复用这些已分片的 KV Cache
​
5. PCP 特别处理（RFC #25749）
   "In the attention module, we first execute the original DCP computation logic
   within the respective PCP group. Then, before updating the KV Cache..."
   → PCP 先在各自组内执行 DCP 逻辑，再处理跨组更新
```

### 6.5 MLA 模型的 DCP 特殊处理

对于 DeepSeek 的 MLA（Multi-head Latent Attention）模型，DCP 有特殊实现（RFC #24685）：

```
MLA 的 KV Cache 特点：
- 使用低秩投影将 KV 压缩到 latent_dim = 512（远小于原始 kv_head × head_dim）
- 压缩后：K/V head 数量变成 1（latent vector）
- 显存大幅节省，但 TP 可切分的 kv_head 维度变得更窄
​
MLA + DCP 的 KV Cache 二维布局：
  TP 切 kv_head（latent_dim 维度）→ 切分深度有限
  DCP 切 seq_len → 主要节省手段
  
这是 DCP 对 MLA 最重要的价值！
```

## 7\. 完整使用示例

### 7.1 启动 DCP（Decode Context Parallel）

```bash
# 8 卡机器，使用 TP=4, DCP=2
# 即 4 个卡做 TP，剩余 4 个卡做 DCP
vllm serve \
  meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --decode-context-parallel-size 2 \
  --max-model-len 131072
通信拓扑：
[GPU0, GPU1, GPU2, GPU3] = TP=4 group（切 Q/K/V head 维度）
     ↕ DCP=2
[GPU4, GPU5, GPU6, GPU7] = 共享 KV Cache（各持有 1/2 序列）
​
KV Cache 分布：
  GPU0/4: token[0..32K] 的 K/V
  GPU1/5: token[32K..64K] 的 K/V
  GPU2/6: token[64K..96K] 的 K/V
  GPU3/7: token[96K..128K] 的 K/V
```

### 7.2 未来 PCP + DCP 联合使用

```bash
# PCP 稳定后，可以这样用（RFC 阶段，仅供参考）
vllm serve \
  meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --prefill-context-parallel-size 2 \
  --decode-context-parallel-size 2 \
  --max-model-len 26214
```

## 8\. 总结

### 8.1 一张图看懂 CP 全貌

```
vLLM 上下文并行（Context Parallel）
​
                    ┌─────────────────────────────────────┐
                    │         上下文并行（CP）             │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ↓                                         ↓
     ┌─────────────────┐                     ┌─────────────────┐
     │  PCP            │                     │  DCP            │
     │  Prefill CP     │                     │  Decode CP      │
     ├─────────────────┤                     ├─────────────────┤
     │ 目标:↓TTFT      │                     │ 目标:↓KV显存占用│
     │ 切分: Q,K,V     │                     │ 切分: 仅 KV Cache│
     │ 算法:Ring/Ulysses│                    │ 算法:AG_RS / A2A │
     │ 状态:RFC #25749 │                     │ 状态:已上线稳定  │
     └─────────────────┘                     └─────────────────┘
```

### 8.2 核心结论

| 观点 | 内容 |
| --- | --- |
| 背景 | LLM 推理 Prefill（计算密集）和 Decode（内存密集）特性截然不同，需要不同的并行策略 |
| DCP 解决的问题 | Decode 阶段 KV Cache 显存爆炸，通过沿序列维度 interleave 切分减少单卡显存占用 |
| PCP 解决的问题 | Prefill 阶段 O(L²) 计算瓶颈，通过 Ring Attention 等算法将长序列 [并行化](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E5%B9%B6%E8%A1%8C%E5%8C%96&zhida_source=entity) |
| DCP 算法 | Interleaved KV Cache 布局 + AllGather/A2A 通信 + TP 对 kv\_head 的二维切分 |
| PCP 算法 | Ring Attention（P2P 环状流动）或 DeepSpeed Ulysses（All-to-All） |
| vLLM DCP | 已上线，支持 MLA 和 GQA 模型，配置 --dcp 参数即可 |
| vLLM PCP | RFC 阶段（#25749），预计 v0.18+ 逐步支持 |
| [最佳实践](https://zhida.zhihu.com/search?content_id=273886568&content_type=Article&match_order=1&q=%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5&zhida_source=entity) | 先增大 TP → 再加 DCP → PCP 补充 Prefill 优化 |

## 参考资源

| 资源 | 链接 |
| --- | --- |
| vLLM Context Parallel 部署文档 | [docs.vllm.ai/en/latest/](https://link.zhihu.com/?target=https%3A//docs.vllm.ai/en/latest/serving/context_parallel_deployment/) |
| vLLM PCP RFC #25749 | [github.com/vllm-project](https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/issues/25749) |
| vLLM DCP RFC #24685 | [github.com/vllm-project](https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/issues/24685) |
| vLLM CP RFC #22693 | [github.com/vllm-project](https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/issues/22693) |
| vLLM CP 设计文档 | [docs.vllm.ai/projects/a](https://link.zhihu.com/?target=https%3A//docs.vllm.ai/projects/ascend/en/main/developer_guide/Design_Documents/context_parallel.html) |
| Ring Attention 论文 | Liu et al., "Ring Attention", 2023 |
| DeepSpeed Ulysses | DeepSpeed Team, "Ulysses Sequence Parallelism", 2024 |
| vLLM 知乎 CP 详解 | [zhuanlan.zhihu.com/p/20](https://zhuanlan.zhihu.com/p/2019809858040378281) |
| vLLM 知乎 DCP 详解 | [zhuanlan.zhihu.com/p/20](https://zhuanlan.zhihu.com/p/2020086868914499979) |

编辑于 2026-04-27 22:17・浙江

赞同 38