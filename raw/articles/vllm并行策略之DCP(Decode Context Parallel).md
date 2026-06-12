---
title: "vllm并行策略之DCP(Decode Context Parallel)"
source: "https://zhuanlan.zhihu.com/p/2020086868914499979"
author:
  - "[[梦初AI Infra]]"
published:
created: 2026-06-09
description: "接上篇对 vllm CP并行的总体概述，这篇来具体说说DCP。首先，CP并行顾名思义是为了长上下文并行做的优化。由于推理过程中prefill阶段和decode阶段的bottleneck不同，vllm为这俩阶段分别做了优化（PCP和DCP），其中…"
tags:
  - "clippings"
---
67 人赞同了该文章

目录

收起

概述

DCP具体使用

DCP具体实现

KVCache分片

Decode阶段

Prefill阶段

总结

接上篇对 [vllm CP并行的总体概述](https://zhuanlan.zhihu.com/p/2019809858040378281) ，这篇来具体说说DCP。首先，CP并行顾名思义是为了长上下文并行做的优化。由于推理过程中prefill阶段和decode阶段的bottleneck不同，vllm为这俩阶段分别做了优化（PCP和DCP），其中PCP还在开发中，DCP在 [CUDA](https://zhida.zhihu.com/search?content_id=271974950&content_type=Article&match_order=1&q=CUDA&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODExNzIyNDgsInEiOiJDVURBIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjcxOTc0OTUwLCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.hdRe2yCfoLbpTMpTTZRtFulCGpyfHXA6W0amWnLNMG0&zhida_source=entity) 上已经实现，但其他backend可能还没有跟进。在CUDA上直接添加参数--decode-context-parallel-size x就可以使用DCP并行了。

DCP并行的流程其实可以理解为 [Flash Decoding](https://link.zhihu.com/?target=https%3A//crfm.stanford.edu/2023/10/12/flashdecoding.html) 的分布式并行，虽然细节上还是有区别的。

![动图封面](https://pic3.zhimg.com/v2-13fcb10493400523013dcfe55cc9b846_b.jpg)

图0. Flash Decoding流程

## 概述

解码上下文并行 DCP (Decode Context Parallel) 通过 **对KVCache在seq\_len维度切分** 来降低KVCache显存开销。由于decode阶段是memory-bound，因此使用DCP后，单卡KVCache显存会降为原来的1/DCP，读取数据量也会降为原来的1/DCP。在具体计算时，相比于 [TP通信](https://zhida.zhihu.com/search?content_id=271974950&content_type=Article&match_order=1&q=TP%E9%80%9A%E4%BF%A1&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODExNzIyNDgsInEiOiJUUOmAmuS_oSIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI3MTk3NDk1MCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.EjKoGTx_I98bVjONCa0uhMErlTpMPJgCmlNfXoc-dzU&zhida_source=entity) ，DCP会多通信开销。

目前DCP已经支持MLA/MQA/GQA（理论上MHA不会用到DCP，MHA直接TP切分就可以），也兼容Chunked Prefill和 [Prefix Cache](https://zhida.zhihu.com/search?content_id=271974950&content_type=Article&match_order=1&q=Prefix+Cache&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3ODExNzIyNDgsInEiOiJQcmVmaXggQ2FjaGUiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzE5NzQ5NTAsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.zzP50kM4TvtCkB74PN2Eby7MQI7s5UevVEyCacHO6ks&zhida_source=entity) 。

我们知道长序列并行情况下，KVCache显存开销与context\_len成正比，当MLA/MQA/GQA情况下，num\_kv\_heads较小，TP设置一般<=num\_kv\_heads, 导致TP设置不大，特别是MLA Decode/MQA的情况下(num\_kv\_heads=1)。此时当然也可以将TP设大，但这时就需要KVCache复制了，显存开销很大。

具体可见图1，原图来自 [Helix Parallelism论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2507.07120) ，但由于vllm的实现与Helix论文还是有点区别，因此本文在原图基础上做了修改并添加了注释。（主要区别在于名称不一样，以及DCP是复用TP group的，而Helix论文中是独立于TP group的，后续通信方式也有区别）

![](https://pic2.zhimg.com/v2-af7971c733aaf65b73e1b5ef2e6c2b2d_1440w.jpg)

图1. TP和DCP示意图

在图1中，我们以 GQA 为例，Query heads Q=4，KV heads K=2。每个块处理一个上下文长度为 S 的 token。

（a）无 TP：所有Q和KV heads 都位于同一 GPU 上；KVCache无重复。

（b）TP=2：Q heads 分布在 2 个 GPU 上；由于TP ≤ K，KV heads 仍然保持清晰的分区。

（c）TP=4：分片数量多于KV heads；GPU 必须复制KV cache，才能为其指定的Q heads提供服务，这再次导致了 DRAM 容量和带宽效率的低下。

（d） **TP+DCP** （TP=4，DCP=2）：Helix 使用 DCP 将KV cache按序列长度 (S) 分片，因此每个 DCP rank仅包含一个切片 (S/2)。通过将 Attention计算时的TP并行度 限制为 K (KV heads大小)，并将剩余的 GPU 分配给DCP，避免了KVCache重复，并形成了 **KVCache二维布局：TP/DCP 拆分heads，DCP 拆分sequence**.

![](https://picx.zhimg.com/v2-5b03f47050d9aebf1d7828e85c555157_1440w.jpg)

图2. TP+DCP时KVCache布局

## DCP具体使用

\--decode-context-parallel-size x来enable DCP。TP需要被DCP整除。TP world\_size不会因 dcp 而改变，它只是复用 TP 组的 GPU，并将一个 TP 组拆分为 tp\_size//dcp\_size 个 DCP 组。例如：

```
with -tp 8 -dcp 8 , we use 8 GPUs
with -tp 8 -dcp 4 , we use 8 GPUs
with -tp 4 -dcp 4 -pp 2 , we use 8 GPUs
and kvcache token budget always increased by \`dcp\` times.
```

DCP一般用在num\_kv\_cache比较小的情况下：

- MHA一般只用TP，因为此时num\_q\_heads=num\_kv\_heads, 不会出现KVCache复制的情况
- GQA一般设置DCP=TP/num\_kv\_heads, 减少KVCache复制
- MLA/GQA一般设置DCP=TP/num\_kv\_heads=TP/1=TP，减少KVCache复制

## DCP具体实现

参照MLA的DCP实现： [Interleave CP (context parallel) 介绍](https://link.zhihu.com/?target=https%3A//docs.google.com/document/d/1C93PuBSJmq8eUM16kD7CftzYlbNJ9j6Ut2W2kVuXgo4/edit%3Ftab%3Dt.0%23heading%3Dh.bc5zpep7g7ln) ，最终实现形式是链接中的MLA-CP。

### KVCache分片

DCP (Decode Context Parallel) 通过对KVCache在seq\_len维度切分来降低KVCache显存开销。具体是怎么实现的呢？其采用的是交错方式(Interleaving)存储KVCache. 这时相对于全局来说，KVCache blocksize增大为原来的DCP倍。

```
e.g. DCP2, req with prompt_len=5, generation_len=4:
kvcache store in dcp_rank0: 0, 2, 4, 6, 8 
kvcache store in dcp_rank1: 1, 3, 5, 7,
```

以MLA为例，MLA 是单 head kv cache，纯 TP 会出现 duplicated kvcache 显存浪费。Interleave CP 复用了 TP的 process\_group，而未引入独立的 CP process\_group, 核心改动是实现kvcache的interleave存储。 **Interleave 存储的逻辑: per-request的视角下，token\_idx为n的token，其kvcache严格存储到cp\_rank等于n%cp\_world\_size的GPU上。**

我们先来理解下KVcache Interleave存储后的Attention计算。

回顾下 图3左图的 **Flash Attention计算** ，在Flash Attention2中，将Q, K, V在seq\_len维度切分，Outer\_loop遍历Q\_chunk, inner\_loop遍历K\_chunk/V\_chunk。由于Attention需要计算token和token的相关性，所以在inner\_loop遍历获得的结果需要online\_softmax更新output。

而在 **Decode阶段使用DCP后** ，由于使用KVCache, Q变成q, query\_len=1(不考虑投机采样时)，如图3右图所示，假设此时KVCache已经存储了3个token，DCP=2，那么DCP rank0存储了K0和K2, DCP rank1存储了K1 (V同理)。此时计算Attention时：

- DCP rank0上会计算q3和K0, K2, V0, V2的结果得到O0
- DCP rank0上会计算q3和K1, K3, V1, V3的结果得到O1，其中K3,V3是本次step时加入到KVCache中的
- 由于得到完整的Attention Score需要K0-K2, V0-V2的所有结果，因此O0和O1也需要online softmax来更新，获得最终的O。
![](https://picx.zhimg.com/v2-cfd9d3392a5973eba709730f34d13355_1440w.jpg)

图3. Attention计算示意图

**通信**

接下来说说计算出O0,O1之后的通信部分。前面提到DCP是需要多余通信的，这是由于上文提到的不同DCP rank得到的output是partial output，需要online\_saftmax调整。如果是两个partial output, 我们可以直接用merge\_attn\_states kernel，通过ouput和lse来更新，其介绍可见 [\[vLLM实践\]\[算子\]📚vLLM算子开发流程: "保姆级"详细记录](https://zhuanlan.zhihu.com/p/1892966682634473987) 。

DCP并行度不确定，且涉及到多卡通信，其具体是怎么做的呢？请见图4。这是DCP的默认通信方式，目前也已经实现了dcp\_all2all通信（参见 [Helix Parallelism论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2507.07120) ， [PR](https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/pull/34883) ）。

DCP 各个rank上通过自身的KVCache计算出partial output后，会先all-gather DCP各rank上的lse，这样每个DCP rank上就能计算出最终的lse，利用最新的lse更新自己rank上的partial output, 最后通过all-reduce获得最终的output（单卡上的话是相加）。

欸等等，那为什么图4中使用的是reduce-scatter呢？这是由于我们把TP也考虑上去了，下文会说明。

![](https://pic1.zhimg.com/v2-88dd14a2cb1f8d37e30d9d38e98f9ebe_1440w.jpg)

图4. DCP通信

### Decode阶段

好的，那我们现在先来总结下DCP的思路，DCP通过将KVCache按seq\_len维度拆分，来缓解显存受限或者访存带宽的问题，使得推理时可以放置更长seq\_len的request；或者同理在相同seq\_len的情况下可以增大bs。但是DCP rank间计算完的output是partial output，需要按照lse更新，这里会引入多卡通信。

现在我们把TP也考虑上，看看情况变成了什么样。先来看看只使用TP时的Attention计算：在sdpa前后的Q/K/V project和O project，分别采用列并行和行并行。由于Attention计算本身的特点，每个head可以独立计算，最后将结果concat起来，因此TP维度一般<num\_kv\_heads, 并且num\_kv\_heads能被TP整除。

![](https://pic3.zhimg.com/v2-a0ab26575905ab44bfd1eeaecc31f064_1440w.jpg)

图5. 使用TP的Attention计算

前面我们也说了，使用DCP就是因为num\_kv\_heads过小，导致要不TP受限，要不TP>num\_kv\_heads但是KVCache复制。我们同样以MLA为例，MLA的decode阶段计算采用矩阵吸收形式，是MQA形式，num\_kv\_heads=1，所以使用TP就必须KVCache复制，导致显存浪费。采用DCP后，假设单机8卡，TP=DCP=8，在计算sdpa时KVCache在seq\_len上拆分8份，在各自DCP Rank上计算。

在TP基础上使用DCP后，vllm实现如图6所示，图6来自 \[RFC\]: Support Prefill Context Parallel (PCP) [#25749](https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/issues/25749) 。具体计算如下：（注意第1,3两点和TP时是没有区别的）

1. Q/K/V project时，采用TP并行(列并行)
2. 在sdpa计算时，不再是采用TP作为num\_heads维度上的并行度，而是 对KVCache使用二维切分形式：KVCache在num\_heads维度并行度是TP/DCP，在seq\_len维度的并行度是DCP。
3. O project时，采用TP并行(行并行)
![](https://pic4.zhimg.com/v2-f90b77efde541a66fc6bac7de3ba6395_1440w.jpg)

图6. Decode阶段使用TP+DCP

所以在计算sdpa之前，会有DCP all-gather，num\_heads维度上的并行度从TP变成TP/DCP。而在计算完sdpa后num\_heads维度上的并行度从TP/DCP变成TP，这也就是为什么上文说到DCP通信时使用reduce-scatter替换all-reduce。

### Prefill阶段

DCP是针对Decode阶段的优化，那Prefill阶段我们需要做什么呢？第一，我们知道采用DCP后KVCache是interleave存储的，所以在prefill阶段我们需要将KVCache按照这个格式存储。第二，前面说到DCP是支持Chunked Prefill和Prefix Cache的，这也需要Prefill阶段的Attention计算做相应的调整。

**MLA Prefill**

Prefill阶段MLA采用非矩阵吸收形式，具体计算时是MHA形式。因此Prefill阶段MLA阶段直接用TP就OK，无需使用DCP （num\_kv\_heads够给TP拆分）。

以flashattn\_mla.py为例，在不使用DCP之前，其会分别计算query\_len部分的output和context\_len部分的output，然后使用merge\_attn\_states合并结果。

那么Prefill阶段的MLA在enable DCP情况下需要做哪些调整呢？具体可见图7（图7图8也均来自 [链接](https://link.zhihu.com/?target=https%3A//github.com/vllm-project/vllm/issues/25749) ）

- KVcache存储需要interleave存储
- 在计算contex\_len部分的attention时，需要对interleave的KVCache进行allgather和reorg
![](https://pic4.zhimg.com/v2-8f17f5bd40a997aa028aef8396b59973_1440w.jpg)

图7. Prefill阶段MLA的DCP实现

**GQA Prefill**

对于GQA而言，prefill阶段和decode阶段的计算流程并没有什么区别（MLA prefill阶段是MHA形式，decode阶段是MQA形式）。其同样也是 计算query\_len部分的output和context\_len部分的output，然后使用merge\_attn\_states合并结果。

这时候context\_len部分计算也可以用上DCP，也即在计算sdpa时KVCache二维拆分，计算完sdpa后DCP group之间进行通信获得最终结果。

而query\_len计算，由于query\_len一般小于context\_len, 这部分还是采用纯TP计算。

![](https://pic4.zhimg.com/v2-b144d97b2fd12d38317507e65b5d0b45_1440w.jpg)

图8. Prefill阶段GQA的DCP实现

## 总结

本文梳理了vllm在25年Q4引入的并行策略DCP，阐述了其基本原理，使用方式，具体实现。如果在CUDA backend上，大家也遇到了decode阶段KVCache复制的问题，可以尝试加上DCP试试~ （没想到写博客这么花时间，哪怕内容早就熟悉了，组织整理还是比较慢），那就希望能够给大家一个参考吧~

发布于 2026-03-25 16:03・美国