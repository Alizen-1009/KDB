---
title: "DeepSeekV4中RoPE设计解析"
source: "https://mp.weixin.qq.com/s/lCTvzq8FiY4q6r4D6QLh9Q"
author:
  - "[[kaiyuan]]"
published:
created: 2026-05-11
description: "在CSA/HCA中存在压缩操作，多个token会被压缩为一个token，位置信息应在压缩前注入还是压缩后注入？Attention采用MQA模式，K与V共享表示。若直接对KV旋转，会将位置信息引入V，该如何处理？本文围绕这两个问题，梳理DSV4的位置编码设计。"
tags:
  - "clippings"
---
kaiyuan *2026年5月8日 07:30*

![图片](https://mmbiz.qpic.cn/mmbiz_gif/uIP3tuXZx8BXuw5wK7j3avmicib6aZVBqicxsN6ricR1AIiam4Jh59fsLhzU4cREAxrfxPqILpAuUfrVROyIugfOofqucCL4XR5XlfxQg6IPNibmw/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

点击 **蓝字** ，关注我们

DeepSeek V4采用RoPE进行位置编码，但由于注意力结构升级，会带来两个核心问题：

1. 在CSA/HCA中存在压缩操作，多个token会被压缩为一个token，位置信息应在压缩前注入还是压缩后注入？
2. Attention采用MQA模式，K与V共享表示。若直接对KV旋转，会将位置信息引入V，该如何处理？

下面围绕这两个问题，梳理DSV4的位置编码设计。

**1 MLA中的RoPE处理回顾**

在分析V4之前，先回顾V2/V3的MLA（ **M** ulti-head **L** atent **A** ttention）方案，因为MLA同样涉及MQA与KV cache压缩问题。

在标准RoPE中，计算query（𝑞 <sub>𝑚</sub> ）与key（𝑘 <sub>𝑛</sub> ）内积时，可写为：

(𝑅 <sub>𝑚</sub> 𝑞 <sub>𝑚</sub>) <sup>⊤</sup> (𝑅 <sub>𝑛</sub> 𝑘 <sub>𝑛</sub>)=𝑞 <sub>𝑚</sub> <sup>⊤</sup> 𝑅 <sub>𝑚</sub> <sup>⊤</sup> 𝑅 <sub>𝑛</sub> 𝑘 <sub>𝑛</sub> =𝑞 <sub>𝑚</sub> <sup>⊤</sup> 𝑅 <sub>𝑛</sub> <sub>−𝑚</sub> 𝑘 <sub>𝑛</sub>

其中𝑅 <sub>𝑚</sub> 、𝑅 <sub>𝑛</sub> 是位置𝑚、𝑛对应的旋转矩阵。由于旋转矩阵正交，满足：

𝑅(𝜃) <sup>⊤</sup> =𝑅(𝜃) <sup>−1</sup> =𝑅(−𝜃)

因此内积只依赖相对位置𝑛−𝑚。具体介绍参考 [彻底搞懂RoPE计算原理：从1D到3D](https://mp.weixin.qq.com/s?__biz=MzYyMjA5NzMwOQ==&mid=2247489976&idx=1&sn=6f1dc71772c032365fd0b5357f7b4b73&scene=21#wechat_redirect) <sup>[1]</sup>

在MLA中，KV下采样后K与V共享同一份cache值，这样可以节省显存；但也带来问题：如果给K注入RoPE，V会被一并旋转，导致V值“掺杂”了位置信息。

一种直观做法是 **将K、V拆开，仅对K旋转** 。但这样需要分别存储K cache和V cache，开销会回到接近GQA。

MLA采用了一个折中的做法：在Q、K隐藏维度中设置一部分专门用于RoPE计算。

图片参考：https://github.com/CalvinXKY/InfraTech/tree/main/models/deepseek\_v3

这样既能让K携带位置信息，又能避免污染V；同时只需额外存储较小的RoPE相关K cache（如图中的k\_pe），远小于完整拆分K/V cache。公式推导参考：part2、3 <sup>[2]</sup>

**2 CSA/HCA中的RoPE处理**

在DSV4的CSA/HCA中，同样存在KV cache压缩与MQA下KV共享的问题。CSA与HCA在RoPE处理上的思路一致，下面以HCA为例说明。

HCA中涉及RoPE的主要位置包括：

1. 窗口通道（SWA）的KV值；
2. C128A压缩器输出的压缩KV值；
3. 上采样后的Q值；
4. Attention输出的O值。

图片参考：https://github.com/CalvinXKY/InfraTech/blob/main/models/deepseek\_v4

看到这个设计，思考问题如下：

**2.1 为什么要对输出O做一次旋转？**

前面在MLA中提到，KV共享时直接旋转KV会让V也带上位置信息。HCA中窗口通道与压缩通道都在KV的最后rope\_head\_dim维度上施加RoPE。对应维度上的计算可写为：

𝑜 <sub>𝑛</sub> =𝜙((𝑅 <sub>𝑚</sub> 𝑞 <sub>𝑚</sub>) <sup>⊤</sup> (𝑅 <sub>𝑛</sub> 𝑘 <sub>𝑛</sub>))𝑅 <sub>𝑛</sub> 𝑣 <sub>𝑛</sub> =𝑝⋅𝑅 <sub>𝑛</sub> 𝑣 <sub>𝑛</sub>

这里多出的𝑅 <sub>𝑛</sub> 等价于给输出 **引入绝对位置信息** 。

绝对位置信息并非一定不可训练，但在 **可扩展性** （尤其是长上下文外推）上通常不如相对位置形式稳定。

因此HCA会对输出再做一次逆旋转：

𝑂′=𝑅 <sub>−𝑖</sub> ⋅𝑂=𝑅 <sub>−𝑖</sub> 𝜙((𝑅 <sub>𝑚</sub> 𝑞 <sub>𝑚</sub>) <sup>⊤</sup> (𝑅 <sub>𝑛</sub> 𝑘 <sub>𝑛</sub>))𝑅 <sub>𝑛</sub> 𝑣 <sub>𝑛</sub> =𝑝⋅𝑅 <sub>𝑛−𝑖</sub> 𝑣 <sub>𝑛</sub>

这样位置项从绝对位置转为相对位置。

这里有个小问题：是否能采用正向旋转−𝑖？答案是否定的，因为从公式看到，结果将仍偏向绝对位置表达。

**2.2 能否直接给P旋转？**

不行。设𝑉末两维为\[seq,head\_dim\]，而𝑃维度为\[seq,seq\]。RoPE旋转作用在head\_dim维，𝑃与旋转维度不匹配。 从计算上看，𝑃𝑉可理解为“标量权重乘向量”，𝑃本身是标量权重集合，不具备可旋转的向量维度。

**2.3 旋转应在压缩前还是压缩后？**

RoPE角度与绝对位置相关：

𝜃(𝑚,𝑖)=𝑚⋅𝜃 <sub>𝑖</sub> =𝑚⋅10000 <sup>−2𝑖/𝑑</sup>

其中𝑚是token位置索引，d为注意力头维度，即hidden\_size/num\_heads，i的取值范围是0,1,…,𝑑/2−1。而C128A会将128个KV状态压缩为1个KV状态，QK计算使用的是压缩后的K。

核心问题是：K旋转角度的系数位置m怎么选？

- 若在压缩前旋转：每个token先带位置再压缩。这样看似直观，但位置信息会在序列维累加混合，容易破坏RoPE所需的相对位置结构。
- 若在压缩后旋转：给每个压缩K值指定一个标定位置即可。该位置可选起始、结束或中点，只要映射规则全程一致。

HCA采用的是“每128段取起始位置”，旋转角度公式：

𝜃(𝑚′,𝑖)=(128⋅𝑡)⋅10000 <sup>−2𝑖/𝑑</sup>

其中是当前压缩K值在压缩序列中的索引。

C128A压缩计算中RoPE的位置

建议的前置阅读:

- [彻底搞懂RoPE计算原理：从1D到3D](https://mp.weixin.qq.com/s?__biz=MzYyMjA5NzMwOQ==&mid=2247489976&idx=1&sn=6f1dc71772c032365fd0b5357f7b4b73&scene=21#wechat_redirect)
- [图解DeepSeek V4：详细计算流程解析](https://mp.weixin.qq.com/s?__biz=MzYyMjA5NzMwOQ==&mid=2247490109&idx=1&sn=8ab153160d29bd514c45c0d2ddf9b345&scene=21#wechat_redirect)

---

**参考：**

1. ^https://zhuanlan.zhihu.com/p/2023493768003724514
2. ^https://spaces.ac.cn/archives/10091

想深耕AI Infra领域？欢迎访问InfraTech库！内容涵盖大模型基础、PyTorch/vLLM/SGLang框架入门、性能加速等核心方向，配套50+知识干货及适合初学者的notebook练习： **https://github.com/CalvinXKY/InfraTech**

扫码关注我们，了解更多AI Infra基础知识。

大模型基础知识 · 目录

继续滑动看下一个

InfraTech

向上滑动看下一个