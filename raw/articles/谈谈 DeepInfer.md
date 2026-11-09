---
title: "谈谈 DeepInfer"
source: "https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247499862&idx=1&sn=73c43dc231dc890f66b0003e9d64c63d&chksm=f83ec3cd225e561b9b807bbbba2fb0dd37a4f8908d4cb928c01de1030e544018e912b1fd4fc1&mpshare=1&scene=1&srcid=1023tPpK7szOqxYoORugJTCs&sharer_shareinfo=e03e90a98177cb761472f72086eff653&sharer_shareinfo_first=e03e90a98177cb761472f72086eff653#rd"
author:
  - "[[渣B]]"
published:
created: 2026-11-09
description:
tags:
  - "clippings"
---
渣B zartbot *2026年10月23日 18:44*

`本文仅代表个人观点, 与作者任职的机构无关. `

## TL;DR

DeepSeek的同学前几天在开源 DeepInfer 的时候, 转发朋友圈的时候都在说“印钞机”开源了. 80%的毛利和17个月回本确实能算印钞机了. 被好多朋友催着写点东西, 其实具体的技术细节的解读, 以前的文章都差不多写过, 这一篇我想把整个事情背后的逻辑串起来详细谈谈为什么要这么做, 以及其它几家为什么做不了.

实质性的问题就是 DeepSeek 从第一天开始就很清楚一个最基本的原理: *计算是很容易 Scale 的, 而 Memory 是很难 Scale 的*. 然后整个模型和 Infra 的迭代就充分的围绕着这个逻辑来做.

而在商业逻辑上, 一直延续着 **普惠** 这条路, 少了很多功利化的商业变现需求, 因此变得更加纯粹. “印钞机”开源这种事情本身就是这个逻辑, 让更多的普通人用得起, 我个人也深深认同这个逻辑, 也一直反复的拷问自己:“世界上如果只剩一个人了, 要钱还是要名?”

过去几年我自己的工作重心一直放在普惠的硬件算力和 infra 的构建上, 正好在某厂工作满 4 年了, 四年和大家一起做了两颗芯片, 2.0 发布比 BF3 晚了一些, 但是性能全方面超越. 而 3.0 直接全面超越 BF4/CX8/CX9, 把 DPU / AI-NIC 做到了世界第一的位置. 另一方面在ScaleUP/ScaleOut互联领域也基本上达到了全球领先的位置了, 接下来很长一段时间的工作重心会换到算力芯片上, 很简单的一个逻辑就是下一代北美的模型参数规模 Scale 到 50T/100T, 国内要追赶买 Rubin / Feymann 的钱预计需要 2000 亿美金, 还不一定买得到那么多卡. 那么如何用 1000 ~ 2000 亿人民币解决这个问题?

## 从 Memory 如何 Scale 的视角谈谈

DeepSeek的第一代模型在2024年中规中矩搞了一个标准的 Dense Transformer 架构, 然后第二代就逐渐开启了整个 MoE 的迭代演进, 到 DeepSeek V3 基本上定型. 这一部分是解决了Dense FFN 的算力消耗和 Decode阶段的Memory Scale的问题. 当然也转移到了互联上, 使得国内外都开启了大规模的 ScaleUP 的演进. 具体可以参考 [《详细谈谈DeepSeek MoE相关的技术发展》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247493157&idx=1&sn=51c0e27a347dd3fe1ed868d87f667897&scene=21#wechat_redirect)

然后 DeepSeek 解决了 FFN block 后, 最近几年的重心逐渐转移到 Attention block 上. 首先是通过 MLA 来压缩 KVCache, 然后是 NSA 和 V3.2 开启 Sparse Attn 的路径, 进一步通过 Sparse 在 Attention block上降低访问内存的需求以及外部存储的需求.

![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/llFnu58nOPY3bEQ8sco6UVWhXkGfcS7o6lcJkmv7KnMtFsRfE50qbabiaSkw88wvcunUQW1nib8CjvxSdv6jvBicBiaWJm2DDyHbenGcaiaUeulg/640?wx_fmt=webp&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

大致的路径就是, 在 V4 的时候引入 CSA 和 HCA 进一步在 token block level 进行压缩, 其实 HCA 的实验发现过度的压缩并不是很有效, 所以 DeepSeek 在 V4.1 Flash的时候换了一个维度进行跨层的 KVCache 共享, 然后再引入了YOCO.具体的分析可以参考 [《详细分析 DeepSeek-V4.1 Flash 技术报告(1): 模型结构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247499768&idx=1&sn=d00b5f17a59ec29634538da5ab9acabd&scene=21#wechat_redirect).

## 谈谈推理 Infra

通常的推理场景下, KVCache的存储会有一个多层的结构:

![图片](https://mmbiz.qpic.cn/mmbiz_jpg/llFnu58nOPYDMCo2hFIgicwAyUFRDtMqRIlKjLYqE23XRlAWiaicb426OdmS6AaF8kA0cmX9XzMEWweGQ734Z3MN8y3aXPbZ461oBUrz3zeuYQ/640?wx_fmt=webp&from=appmsg#imgIndex=1)

- G1: GPU HBM
- G2: Host CPU Memory
- G2.5: Host CPU Memory 结合 RDMA 实现的池化版本
- G3: Local SSD
- G3.5: 分布式存储集群
- G4: 对象存储集群

虽然看上去能够更好的充分的利用资源, 但是也涉及到更复杂的通信和网络压力:

- PD 分离的 RDMA 流量
- G1 - G2: CPU 和 GPU 通过PCIe H2D以及H2D copy的流量
- G2.5: 不同节点之间的 RDMA 流量
- G3: 节点内 CPU 到 NVMe 的 PCIe 流量
- G3.5: GPU服务器机头网卡和存储集群之间的 RDMA 流量
- G4: TCP/HTTP 或 RDMA 流量

整个方案存在多种流量争抢, 网络混跑拥塞等情况. 并且 PCIe 上也没有很好的 QoS 控制能力. 各种流量互相干扰较大.

更重要的是在请求调度算法上的问题, 由于 G1~G3 存在, 因此请求通常会做 Cache Aware 的调度, 避免大量 KVCache 的搬运. 但是这样的调度策略其实非常容易导致 Prefill 节点的负载均衡问题, 并引起大量的请求排队带来的长尾.

同时由于 G1 缓存了一些 KVCache 实际 serving 的时候也需要一些换入换出, 并且会导致一些请求batch size 的约束. 最根本的原因其实还是 KVCache 太大了, 数据搬运本身受到了各种带宽瓶颈的约束, 导致难以动弹.

那么我们再来看一下 DeepSeek 是怎么构建的, 原理很简单, 降低 Cache 的层次化结构, 仅保留 G3.5层, 整个数据路径非常干净, 调度策略来看, 无须考虑复杂的KVCache 本地命中等调度策略. **整个计算节点是无状态的, 完善的存算分离实现更加便于弹性缩扩容和调度.** 当 Prefill 节点算完并传输到 Decode 节点后, 就可以释放 G1 内的显存. KVCache 都由 Decode 节点写入存储.

![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/llFnu58nOPYSaRyP9pqahRKulBCusTOuXA3xMBDrpcYdInB7BuaVoMpMxYyUhBFrMlhticicbdzXmx0SCfzC5xc3LiaWGXB3VibMqn0ic2UXoGlA/640?wx_fmt=webp&from=appmsg#imgIndex=2)

具体的实现其实可以参考 [《谈谈DeepSeek DualPath以及ScaleOut直接接入存储》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247497708&idx=1&sn=bd123a32fc5c13197281ed490abbd86c&token=134441754&lang=zh_CN&scene=21#wechat_redirect). 这种架构看上去更干净, 但是GPU服务器和存储之间的网络带宽将变得非常大, 其实我们从DualPath的论文也能看到, 论文应该是在 DeepSeek V3.2 的阶段, 当时有一个问题就是 Prefill 节点受到机头的 400Gbps 带宽约束, 无法充分利用 Prefill 服务器的算力

![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/llFnu58nOPamicZdPGbrC7IfrQK2rTn72Gb2CmplNa3X2BeibOUfoZGiagyVtRbJGg9nsraVN3VqpUH3OnUE1aROrZTRKdsCl97D1gvBs0qic04/640?wx_fmt=webp&from=appmsg#imgIndex=3)

因此通过 DualPath 的处理, 利用 Decode 节点 Relay 的方式来解决这个问题. 当然这只是 Infra 团队的救急策略, 这里再补充一个背景, 按照 DeepSeek V3.2 的KVCache 和在线服务需要 Cache 24小时来算存储的规模, 我去年底大概估计过整个存储服务器的开销估计都会赶上 GPU集群的价格了.

因此 DeepSeek 团队并没有满足于此, 从模型架构上在 V4 和 V4.1 上进一步对 KVCache 进行了优化, 估计现在不需要 DualPath, 直接从 400Gbps的网卡读就行了?

*所以这80%的毛利, 并不是某几个技巧就能搞定的. 而是他们在长期的演进中一点点的细节逐渐扣出来的*, 你可以看到他们开源的推理引擎的每个算子, 基本上都做到了极致, 例如 [《DeepSeek-V4详细分析(2): MegaMoE》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247498412&idx=1&sn=103dd28729b39dd6313eb664bfca3a80&scene=21#wechat_redirect) 就是一件非常棒的艺术品, 把 GPU 用到了极致.

## 为什么其它模型厂做不到?

前段时间写了一篇文章 [《置身 AI 内: 读懂 AI 竞赛, 必须读懂 AI 组织》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247499854&idx=1&sn=478c9757b5a10098af19b13d9023e527&scene=21#wechat_redirect), 那么借着 DeepInfer 来讲讲如果在其它模型厂会出现什么情况.

推理团队遇到 KVCache 大的时候, 会找网络团队优化通信, 会找存储团队一起搞多层的KVCache存储, 然后会在链路上搞复杂的调度提高Cache Locality 的命中率. 每一步在大型的组织结构内似乎都完全正确, 大概率每个团队都会证明自己已经 Try best 都做好了.

或许有几个“大胆”的Infra同学会给高贵的算法爹提一提: 能压一下 KVCache 么? 但大多数基模的算法和训练团队的 KPI 还是模型的质量, 大概率这事最终就会谈成推理团队如何去做 KVCache 的量化加速, 并且模型掉点了还会被算法爹们挑战...

另一方面其实是在一些算法细节上, 例如一些矩阵的 shape 上, 大量的算法哥并不是非常了解 GPU 的微架构, 这也导致推理时某些 shape 的计算很难充分的利用算力, 甚至有些激活参数更小的模型推理的吞吐都赶不上 DeepSeek Flash 这类激活更大的模型. 如何围绕着自己的算力资源池去设计模型, 并渡让一些 KPI 是一件非常难的事情.

还有一个值得探讨的问题是, 组织结构上能否容忍探索过程中的损失? 即便是算法爹们愿意去尝试新的 Attention 架构压缩 KVCache 也会面临巨大的失败的风险. 例如 DeepSeek-V4 中的 HCA, 在 block 维度压缩过了, 整体上 DSv4 pro 刚发布时的模型能力比同期的 GLM-5.3 / Kimi-K3 也是有些差距的. 但它是一个很好的尝试为 DSv4.1 flash 演进到 CSA2 做出了很大的贡献.

而这些常识只有在一个团队有长期主义的组织文化背景下, 整体上算法和Infra充分协同的情况下才能得以实施. 这也是梁文锋在 2024 年的一次访谈 <sup>[1]</sup> 中谈到:

"我们创新缺的肯定不是资本, 而是缺乏信心以及不知道怎么组织高密度的人才实现有效的创新."

认识了很多DeepSeek 的朋友们, 在和他们交流的过程中能够深刻的感受到不同. 大家一起讨论的时候能够看到一个更全面的视角, 从算法到算子, 再到Infra, 最后深入到 GPU 的微架构. 而在其它的团队交流过程中, 很大程度上只能和他们谈论某一个小领域的问题, 并且最终通常陷入到局部领域的屎上雕花的优化中.

最后还涉及到一些领导者的taste 如何分辨这些局部最优的场景, 如何推动整个变革, 如何充分的融合各个领域的团队? 具体可以参考 [《谈谈 AI 时代的组织结构》](https://mp.weixin.qq.com/s?__biz=MzUxNzQ5MTExNw==&mid=2247499800&idx=1&sn=cb4f48cf147291b7ec513e79366371d5&scene=21#wechat_redirect)...

好了, 到此为止吧...

参考资料

\[1\]

梁文锋访谈: *https://finance.sina.com.cn/roll/2024-07-22/doc-inceypaf4744158.shtml*

随笔 · 目录