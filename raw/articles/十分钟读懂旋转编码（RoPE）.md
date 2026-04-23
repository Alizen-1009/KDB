---
title: "十分钟读懂旋转编码（RoPE）"
source: "https://zhuanlan.zhihu.com/p/647109286"
author:
  - "[[绝密伏击​​​电子科技大学 软件工程硕士]]"
published:
created: 2026-04-22
description: "打个小广告 ☻，知乎专栏 《大模型前沿应用》的内容已经收录在新书《揭秘大模型：从原理到实战》中。感兴趣的朋友可以购买，多谢支持！♥♥旋转位置编码（Rotary Position Embedding，RoPE）是论文 Roformer: Enha…"
tags:
  - "clippings"
---
目录

收起

1\. 旋转编码 RoPE

1.1 基本概念

1.2 绝对位置编码

1.3 2维旋转位置编码

1.4 扩展到多维

1.5 RoPE 的高效计算

1.6 远程衰减

2\. RoPE实验

3\. RoPE代码实现

3.1 在LLAMA中的实现

3.2 在ChatGLM中的实现

4\. RoPE的外推性

总结

附录

参考

打个小广告 ☻，知乎专栏 **《大模型前沿应用》** 的内容已经收录在新书 **《揭秘大模型：从原理到实战》** 中。感兴趣的朋友可以购买，多谢支持！♥♥

---

旋转位置编码（Rotary Position Embedding，RoPE）是论文 [Roformer: Enhanced Transformer With Rotray Position Embedding](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2104.09864.pdf) 提出的一种能够将相对位置信息依赖集成到 self-attention 中并提升 [transformer](https://zhida.zhihu.com/search?content_id=231932826&content_type=Article&match_order=1&q=transformer&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzcwMDYxMjgsInEiOiJ0cmFuc2Zvcm1lciIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjIzMTkzMjgyNiwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.pIYkzn6814xR75SBigCDUJC-s54tel3MJqjSU_7zKiE&zhida_source=entity) 架构性能的位置编码方式。而目前很火的 LLaMA、GLM 模型也是采用该位置编码方式。

和相对位置编码相比，RoPE 具有更好的 **外推性** ，目前是大模型相对位置编码中应用最广的方式之一。

**备注：什么是大模型外推性？**

外推性是指大模型在训练时和预测时的输入长度不一致，导致模型的泛化能力下降的问题。例如，如果一个模型在训练时只使用了512个 token 的文本，那么在预测时如果输入超过512个 token，模型可能无法正确处理。这就限制了大模型在处理长文本或多轮对话等任务时的效果。

**备注：下一篇文章将详细介绍如何基于RoPE提升大模型的外推能力（2k推8k，甚至更长）**

[![](https://picx.zhimg.com/v2-82001f4fd7a4432d5952c7144c49cdb0.jpg?source=7e7ef6e2&needBackground=1)](https://zhuanlan.zhihu.com/p/675243992)

## 1\. 旋转编码 RoPE

### 1.1 基本概念

在介绍 RoPE 之前，先给出一些符号定义，以及基本背景。

首先定义一个长度为 $N$ 的输入序列为：

$(\text{1}) \mathbb{S}_{N} = \left{\right. w_{i} \left.\right}_{i = 1}^{N}$

其中 $w_{i}$ 表示输入序列中第 $i$ 个 token，而输入序列 $\mathbb{S}_{N}$ 对应的 embedding 表示为：

$(\text{2}) \mathbb{E}_{N} = \left{\right. \mathbf{\mathit{x}}_{i} \left.\right}_{i = 1}^{N}$

其中 $\mathbf{\mathit{x}}_{i}$ 表示第 $i$ 个 token $w_{i}$ 对应的 $d$ 维词嵌入向量。

接着在做 self-attention 之前，会用词嵌入向量计算 $\mathbf{\mathit{q}} , \mathbf{\mathit{k}} , \mathbf{\mathit{v}}$ 向量同时加入位置信息，函数公式表达如下：

$(\text{3}) \mathbf{\mathit{q}}_{m} = f_{q} \left(\right. \mathbf{\mathit{x}}_{m} , m \left.\right) \mathbf{\mathit{k}}_{n} = f_{k} \left(\right. \mathbf{\mathit{x}}_{n} , n \left.\right) \mathbf{\mathit{v}}_{n} = f_{v} \left(\right. \mathbf{\mathit{x}}_{n} , n \left.\right)$

其中 $\mathbf{\mathit{q}}_{m}$ 表示第 $m$ 个 token 对应的词向量 $\mathbf{\mathit{x}}_{m}$ 集成位置信息 $m$ 之后的 query 向量。而 $\mathbf{\mathit{k}}_{n}$ 和 $\mathbf{\mathit{v}}_{n}$ 则表示第 $n$ 个 token 对应的词向量 $\mathbf{\mathit{x}}_{n}$ 集成位置信息 $n$ 之后的 key 和 value 向量。

而基于 transformer 的位置编码方法都是着重于构造一个合适的 $f \left(\right. \mathbf{\mathit{q}} , \mathbf{\mathit{k}} , \mathbf{\mathit{v}} \left.\right)$ 函数形式。

而计算第 $m$ 个词嵌入向量 $\mathbf{\mathit{x}}_{m}$ 对应的 self-attention 输出结果，就是 $\mathbf{\mathit{q}}_{m}$ 和其他 $\mathbf{\mathit{k}}_{n}$ 都计算一个 attention score ，然后再将 attention score 乘以对应的 $\mathbf{\mathit{v}}_{n}$ 再求和得到输出向量 $\mathbf{\mathit{o}}_{m}$ ：

$(\text{4}) a_{m , n} = \frac{\text{exp} \left(\right. \frac{\mathbf{\mathit{q}}_{m}^{\textbf{T}} \mathbf{\mathit{k}}_{n}}{\sqrt{d}} \left.\right)}{\sum_{j = 1}^{N} \text{exp} \left(\right. \frac{\mathbf{\mathit{q}}_{m}^{\textbf{T}} \mathbf{\mathit{k}}_{j}}{\sqrt{d}} \left.\right)} \mathbf{\mathit{o}}_{m} = \sum_{n = 1}^{N} a_{m , n} \mathbf{\mathit{v}}_{n}$

### 1.2 绝对位置编码

对于位置编码，常规的做法是在计算 query, key 和 value 向量之前，会计算一个位置编码向量 $\mathbf{\mathit{p}}_{i}$ 加到词嵌入 $\mathbf{\mathit{x}}_{i}$ 上，位置编码向量 $\mathbf{\mathit{p}}_{i}$ 同样也是 $d$ 维向量，然后再乘以对应的变换矩阵 $\mathbf{\mathit{W}}$ ：

$(\text{5}) f_{t : t \in \left{\right. q , k , v \left.\right}} \left(\right. \mathbf{\mathit{x}}_{i} , i \left.\right) := \mathbf{\mathit{W}}_{t : t \in \left{\right. q , k , v \left.\right}} \left(\right. \mathbf{\mathit{x}}_{i} + \mathbf{\mathit{p}}_{i} \left.\right)$

而经典的位置编码向量 $\mathbf{\mathit{p}}_{i}$ 的计算方式是使用 Sinusoidal 函数：

$(\text{6}) \mathbf{\mathit{p}}_{i , 2 t} = \text{sin} \left(\right. k / 10000^{2 t / d} \left.\right) \mathbf{\mathit{p}}_{i , 2 t + 1} = \text{cos} \left(\right. k / 10000^{2 t / d} \left.\right)$

其中 $\mathbf{\mathit{p}}_{i , 2 t}$ 表示位置 $d$ 维度向量 $\mathbf{\mathit{p}}_{i}$ 中的第 $2 t$ 位置分量也就是偶数索引位置的计算公式，而 $\mathbf{\mathit{p}}_{i , 2 t + 1}$ 就对应第 $2 t + 1$ 位置分量也就是奇数索引位置的计算公式。

### 1.3 2维旋转位置编码

论文中提出为了能利用上 token 之间的相对位置信息，假定 query 向量 $\mathbf{\mathit{q}}_{m}$ 和 key 向量 $\mathbf{\mathit{k}}_{n}$ 之间的内积操作可以被一个函数 $g$ 表示，该函数 $g$ 的输入是词嵌入向量 $\mathbf{\mathit{x}}_{m}$ ， $\mathbf{\mathit{x}}_{n}$ 和它们之间的相对位置 $m - n$ ：

$(\text{7}) \langle \mathbf{\mathit{f}}_{q} \left(\right. \mathbf{\mathit{x}}_{m} , m \left.\right) , f_{k} \left(\right. \mathbf{\mathit{x}}_{n} , n \left.\right) \rangle = g \left(\right. \mathbf{\mathit{x}}_{m} , \mathbf{\mathit{x}}_{n} , m - n \left.\right)$

接下来的目标就是找到一个等价的位置编码方式，从而使得上述关系成立。

假定现在词嵌入向量的维度是两维 $d = 2$ ，这样就可以利用上2维度平面上的向量的几何性质，然后论文中提出了一个满足上述关系的 $f$ 和 $g$ 的形式如下：

$(\text{8}) f_{q} \left(\right. \mathbf{\mathit{x}}_{m} , m \left.\right) = \left(\right. \mathbf{\mathit{W}}_{q} \mathbf{\mathit{x}}_{m} \left.\right) e^{i m \theta} f_{k} \left(\right. \mathbf{\mathit{x}}_{n} , n \left.\right) = \left(\right. \mathbf{\mathit{W}}_{k} \mathbf{\mathit{x}}_{n} \left.\right) e^{i n \theta} g \left(\right. \mathbf{\mathit{x}}_{m} , \mathbf{\mathit{x}}_{n} , m - n \left.\right) = \text{Re} \left[\right. \left(\right. \mathbf{\mathit{W}}_{q} \mathbf{\mathit{x}}_{m} \left.\right) \left(\right. \mathbf{\mathit{W}}_{k} \mathbf{\mathit{x}}_{n} \left.\right)^{*} e^{i \left(\right. m - n \left.\right) \theta} \left]\right.$

这里面 Re 表示复数的实部。

进一步地， $f_{q}$ 可以表示成下面的式子：

$(\text{9}) \begin{matrix}f_{q} \left(\right. \mathbf{\mathit{x}}_{m} , m \left.\right) & = \left(\right. \begin{matrix}cos ⁡ m \theta & - sin ⁡ m \theta \left.\right) \\ sin ⁡ m \theta & cos ⁡ m \theta\end{matrix} \left.\right) \left(\right. \begin{matrix}W_{q}^{\left(\right. 1 , 1 \left.\right)} & W_{q}^{\left(\right. 1 , 2 \left.\right)} \\ W_{q}^{\left(\right. 2 , 1 \left.\right)} & W_{q}^{\left(\right. 2 , 2 \left.\right)}\end{matrix} \left.\right) \left(\right. \begin{matrix}x_{m}^{\left(\right. 1 \left.\right)} \\ x_{m}^{\left(\right. 2 \left.\right)}\end{matrix} \left.\right) \\ = \left(\right. \begin{matrix}cos ⁡ m \theta & - sin ⁡ m \theta \left.\right) \\ sin ⁡ m \theta & cos ⁡ m \theta\end{matrix} \left.\right) \left(\right. \begin{matrix}q_{m}^{\left(\right. 1 \left.\right)} \\ q_{m}^{\left(\right. 2 \left.\right)}\end{matrix} \left.\right)\end{matrix}$

看到这里会发现，这不就是 query 向量乘以了一个旋转矩阵吗？这就是为什么叫做旋转位置编码的原因。

同理， $f_{k}$ 可以表示成下面的式子：

$(\text{10}) \begin{matrix}f_{k} \left(\right. \mathbf{\mathit{x}}_{m} , m \left.\right) & = \left(\right. \begin{matrix}cos ⁡ m \theta & - sin ⁡ m \theta \left.\right) \\ sin ⁡ m \theta & cos ⁡ m \theta\end{matrix} \left.\right) \left(\right. \begin{matrix}W_{k}^{\left(\right. 1 , 1 \left.\right)} & W_{k}^{\left(\right. 1 , 2 \left.\right)} \\ W_{k}^{\left(\right. 2 , 1 \left.\right)} & W_{k}^{\left(\right. 2 , 2 \left.\right)}\end{matrix} \left.\right) \left(\right. \begin{matrix}x_{m}^{\left(\right. 1 \left.\right)} \\ x_{m}^{\left(\right. 2 \left.\right)}\end{matrix} \left.\right) \\ = \left(\right. \begin{matrix}cos ⁡ m \theta & - sin ⁡ m \theta \left.\right) \\ sin ⁡ m \theta & cos ⁡ m \theta\end{matrix} \left.\right) \left(\right. \begin{matrix}k_{m}^{\left(\right. 1 \left.\right)} \\ k_{m}^{\left(\right. 2 \left.\right)}\end{matrix} \left.\right)\end{matrix}$

最终 $g \left(\right. \mathbf{\mathit{x}}_{m} , \mathbf{\mathit{x}}_{n} , m - n \left.\right)$ 可以表示如下：

$(\text{11}) g \left(\right. \mathbf{\mathit{x}}_{m} , \mathbf{\mathit{x}}_{n} , m - n \left.\right) = \left(\right. \begin{matrix}\mathbf{\mathit{q}}_{m}^{\left(\right. 1 \left.\right)} & \mathbf{\mathit{q}}_{m}^{\left(\right. 2 \left.\right)}\end{matrix} \left.\right) \left(\right. \begin{matrix}cos ⁡ \left(\right. \left(\right. m - n \left.\right) \theta \left.\right) & - sin ⁡ \left(\right. \left(\right. m - n \left.\right) \theta \left.\right) \\ sin ⁡ \left(\right. \left(\right. m - n \left.\right) \theta \left.\right) & cos ⁡ \left(\right. \left(\right. m - n \left.\right) \theta \left.\right)\end{matrix} \left.\right) \left(\right. \begin{matrix}k_{n}^{\left(\right. 1 \left.\right)} \\ k_{n}^{\left(\right. 2 \left.\right)}\end{matrix} \left.\right)$

关于上面公式（8）~（11）的具体推导，可以参见文章最后的 **附录** ，或者参考文章： [一文看懂 LLaMA 中的旋转式位置编码（Rotary Position Embedding）](https://zhuanlan.zhihu.com/p/642884818) 。

### 1.4 扩展到多维

将2维推广到任意维度，可以表示如下：

$(\text{12}) f_{\left{\right. q , k \left.\right}} \left(\right. \mathbf{\mathit{x}}_{m} , m \left.\right) = \mathbf{\mathit{R}}_{\Theta , m}^{d} \mathbf{\mathit{W}}_{\left{\right. q , k \left.\right}} \mathbf{\mathit{x}}_{m}$

内积满足线性叠加性，因此任意偶数维的RoPE，我们都可以表示为二维情形的拼接，即

$(\text{13}) \mathbf{\mathit{R}}_{\Theta , m}^{d} = \underset{\mathbf{\mathit{W}}_{m}}{\underbrace{\left(\right. \begin{matrix}cos ⁡ m \theta_{0} & - sin ⁡ m \theta_{0} & 0 & 0 & \hdots & 0 & 0 \\ sin ⁡ m \theta_{0} & cos ⁡ m \theta_{0} & 0 & 0 & \hdots & 0 & 0 \\ 0 & 0 & cos ⁡ m \theta_{1} & - sin ⁡ m \theta_{1} & \hdots & 0 & 0 \\ 0 & 0 & sin ⁡ m \theta_{1} & cos ⁡ m \theta_{1} & \hdots & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \hdots & cos ⁡ m \theta_{d / 2 - 1} & - sin ⁡ m \theta_{d / 2 - 1} \\ 0 & 0 & 0 & 0 & \hdots & sin ⁡ m \theta_{d / 2 - 1} & cos ⁡ m \theta_{d / 2 - 1}\end{matrix} \left.\right)}}$

$\Theta = \left{\right. \theta_{i} = 10000^{- 2 \left(\right. i - 1 \left.\right) / d} , i \in \left[\right. 1 , 2 , . . . , d / 2 \left]\right. \left.\right}$

将 RoPE 应用到前面公式（4）的 Self-Attention 计算，可以得到 **包含相对位置信息的Self-Attetion** ：

$(\text{14}) \mathbf{\mathit{q}}_{m}^{\textbf{T}} \mathbf{\mathit{k}}_{n} = \left(\left(\right. \mathbf{\mathit{R}}_{\Theta , m}^{d} \mathbf{\mathit{W}}_{q} \mathbf{\mathit{x}}_{m} \left.\right)\right)^{\textbf{T}} \left(\right. \mathbf{\mathit{R}}_{\Theta , n}^{d} \mathbf{\mathit{W}}_{k} \mathbf{\mathit{x}}_{n} \left.\right) = \mathbf{\mathit{x}}_{m}^{\textbf{T}} \mathbf{\mathit{W}}_{q} \mathbf{\mathit{R}}_{\Theta , n - m}^{d} \mathbf{\mathit{W}}_{k} \mathbf{\mathit{x}}_{n}$ 其中， $\mathbf{\mathit{R}}_{\Theta , n - m}^{d} = \left(\left(\right. \mathbf{\mathit{R}}_{\Theta , m}^{d} \left.\right)\right)^{\textbf{T}} \mathbf{\mathit{R}}_{\Theta , n}^{d}$ 。

值得指出的是，由于 $\mathbf{\mathit{R}}_{\Theta}^{d}$ 是一个正交矩阵，它不会改变向量的模长，因此通常来说它不会改变原模型的稳定性。

### 1.5 RoPE 的高效计算

由于 $\mathbf{\mathit{R}}_{\Theta , m}^{d}$ 的稀疏性，所以直接用矩阵乘法来实现会很浪费算力， **推荐通过下述方式来实现 RoPE** ：

$(\text{15}) \mathbf{\mathit{R}}_{\Theta , m}^{d} \mathbf{\mathit{x}} = \left(\right. \begin{matrix}x_{0} \\ x_{1} \\ x_{2} \\ x_{3} \\ \vdots \\ x_{d - 2} \\ x_{d - 1}\end{matrix} \left.\right) \bigotimes \left(\right. \begin{matrix}cos ⁡ m \theta_{0} \\ cos ⁡ m \theta_{0} \\ cos ⁡ m \theta_{1} \\ cos ⁡ m \theta_{1} \\ \vdots \\ cos ⁡ m \theta_{d / 2 - 1} \\ cos ⁡ m \theta_{d / 2 - 1}\end{matrix} \left.\right) + \left(\right. \begin{matrix}- x_{1} \\ x_{0} \\ - x_{3} \\ x_{2} \\ \vdots \\ - x_{d - 1} \\ x_{d - 2}\end{matrix} \left.\right) \bigotimes \left(\right. \begin{matrix}sin ⁡ m \theta_{0} \\ sin ⁡ m \theta_{0} \\ sin ⁡ m \theta_{1} \\ sin ⁡ m \theta_{1} \\ \vdots \\ sin ⁡ m \theta_{d / 2 - 1} \\ sin ⁡ m \theta_{d / 2 - 1}\end{matrix} \left.\right)$

其中 $\bigotimes$ 是逐位对应相乘，即计算框架中的 $*$ 运算。从这个实现也可以看到，RoPE 可以视为是乘性位置编码的变体。

总结来说，RoPE 的 self-attention 操作的流程是：对于 token 序列中的每个词嵌入向量，首先计算其对应的 query 和 key 向量，然后对每个 token 位置都计算对应的旋转位置编码，接着对每个 token 位置的 query 和 key 向量的元素按照 **两两一组** 应用旋转变换，最后再计算 query 和 key 之间的内积得到 self-attention 的计算结果。

论文中有个很直观的图片展示了旋转变换的过程：

![](https://pic3.zhimg.com/v2-6db340ee9d34709e9c25921f5a0c2a0e_1440w.jpg)

### 1.6 远程衰减

可以看到，RoPE 形式上和前面公式（6） [Sinusoidal 位置编码](https://zhida.zhihu.com/search?content_id=231932826&content_type=Article&match_order=1&q=Sinusoidal+%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzcwMDYxMjgsInEiOiJTaW51c29pZGFsIOS9jee9rue8lueggSIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjIzMTkzMjgyNiwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.iFR5C3adD4BHX0OiLTj9THyi8E9NU5ByoavEOrL9MMY&zhida_source=entity) 有点相似，只不过 Sinusoidal 位置编码是加性的，而 RoPE 可以视为乘性的。在 $\theta_{i}$ 的选择上，RoPE 同样沿用了 Sinusoidal 位置编码的方案，即 $\theta_{i} = 10000^{- 2 i / d}$ ，它可以带来一定的远程衰减性。

具体证明如下：将 $\mathbf{\mathit{q}} , \mathbf{\mathit{k}}$ 两两分组后，它们加上 RoPE 后的内积可以用复数乘法表示为：

$(\text{16}) \left(\left(\right. \mathbf{\mathit{R}}_{\Theta , m}^{d} \mathbf{\mathit{W}}_{q} \mathbf{\mathit{x}}_{m} \left.\right)\right)^{\textbf{T}} \left(\right. \mathbf{\mathit{R}}_{\Theta , n}^{d} \mathbf{\mathit{W}}_{k} \mathbf{\mathit{x}}_{n} \left.\right) = \text{Re} \left[\right. \sum_{i = 0}^{d / 2 - 1} \mathbf{\mathit{q}}_{\left[\right. 2 i : 2 i + 1 \left]\right.} \mathbf{\mathit{k}}_{\left[\right. 2 i : 2 i + 1 \left]\right.}^{*} e^{\text{i} \left(\right. m - n \left.\right) \theta_{i}} \left]\right.$

记 $h_{i} = \mathbf{\mathit{q}}_{\left[\right. 2 i : 2 i + 1 \left]\right.} \mathbf{\mathit{k}}_{\left[\right. 2 i : 2 i + 1 \left]\right.}^{*} , S_{j} = \sum_{i = 0}^{j - 1} e^{\text{i} \left(\right. m - n \left.\right) \theta_{i}}$ ，并约定 $h_{d / 2} = 0 , S_{0} = 0$ ，那么由 [Abel变换（分部求和法）](https://link.zhihu.com/?target=https%3A//zh.wikipedia.org/wiki/%25E5%2588%2586%25E9%2583%25A8%25E6%25B1%2582%25E5%2592%258C%25E6%25B3%2595) 可以得到：

$(\text{17}) \sum_{i = 0}^{d / 2 - 1} \mathbf{\mathit{q}}_{\left[\right. 2 i : 2 i + 1 \left]\right.} \mathbf{\mathit{k}}_{\left[\right. 2 i : 2 i + 1 \left]\right.}^{*} e^{\text{i} \left(\right. m - n \left.\right) \theta_{i}} = \sum_{i = 0}^{d / 2 - 1} h_{i} \left(\right. S_{i + 1} - S_{i} \left.\right) = - \sum_{i = 0}^{d / 2 - 1} S_{i + 1} \left(\right. h_{i + 1} - h_{i} \left.\right)$

所以

$(\text{18}) \begin{matrix}\left|\right. \sum_{i = 0}^{d / 2 - 1} \mathbf{\mathit{q}}_{\left[\right. 2 i : 2 i + 1 \left]\right.} \mathbf{\mathit{k}}_{\left[\right. 2 i : 2 i + 1 \left]\right.}^{*} e^{\text{i} \left(\right. m - n \left.\right) \theta_{i}} \left|\right. = & \left|\right. \sum_{i = 0}^{d / 2 - 1} S_{i + 1} \left(\right. h_{i + 1} - h_{i} \left.\right) \left|\right. \\ \leq & \sum_{i = 0}^{d / 2 - 1} \left|\right. S_{i + 1} \left|\right. \left|\right. h_{i + 1} - h_{i} \left|\right. \\ \leq & \left(\right. \underset{i}{max} \left|\right. h_{i + 1} - h_{i} \left|\right. \left.\right) \sum_{i = 0}^{d / 2 - 1} \left|\right. S_{i + 1} \left|\right.\end{matrix}$

因此我们可以考察 $\frac{1}{d / 2} \sum_{i = 1}^{d / 2} \left|\right. S_{i} \left|\right.$ 随着相对距离的变化情况来作为衰减性的体现：

![](https://pic3.zhimg.com/v2-6376b397b8ea3e8f05d74d433e98a3a4_1440w.jpg)

从图中我们可以看到 **随着相对距离的变大，内积结果有衰减趋势** 的出现。因此，选择 $\theta_{i} = 10000^{- 2 i / d}$ ，确实能带来一定的远程衰减性。论文中还试过以 $\theta_{i} = 10000^{- 2 i / d}$ 为初始化，将 $\theta_{i}$ 视为可训练参数，然后训练一段时间后发现 $\theta_{i}$ 并没有显著更新，因此干脆就直接固定 $\theta_{i} = 10000^{- 2 i / d}$ 了。

## 2\. RoPE实验

我们看一下 RoPE 在预训练阶段的实验效果：

| Stage | Max seq length | Batch size | Training steps | Loss | Accuracy |
| --- | --- | --- | --- | --- | --- |
| 1 | 512 | 256 | 200k | 1.73 | 65.0% |
| 2 | 1536 | 256 | 12.5k | 1.61 | 66.8% |
| 3 | 256 | 256 | 120k | 1.75 | 64.6% |
| 4 | 128 | 512 | 80k | 1.83 | 63.4% |
| 5 | 1536 | 256 | 10k | 1.58 | 67.4% |
| 6 | 512 | 512 | 30k | 1.66 | 66.2% |

从上面可以看出，增大序列长度，预训练的准确率反而有所提升，这体现了 **RoPE 具有良好的外推能力** 。

下面是在下游任务上的实验结果：

| Model | Validation | Test |
| --- | --- | --- |
| BERT-512 | 64.13% | 67.77% |
| WoBERT-512 | 64.07% | 68.10% |
| [RoFormer](https://zhida.zhihu.com/search?content_id=231932826&content_type=Article&match_order=1&q=RoFormer&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzcwMDYxMjgsInEiOiJSb0Zvcm1lciIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjIzMTkzMjgyNiwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.O5SudUavQFBr0TOmnNcVlI81ODBqp2VSPk5L4tV1DbY&zhida_source=entity) -512 | 64.13% | 68.29% |
| RoFormer-1024 | 66.07% | 69.79% |

其中 RoFormer 是一个绝对位置编码替换为 RoPE 的 [WoBERT](https://link.zhihu.com/?target=https%3A//github.com/ZhuiyiTechnology/WoBERT) 模型，后面的参数（512）是微调时截断的maxlen，可以看到 RoPE 确实能较好地处理长文本语义。

## 3\. RoPE代码实现

[Meta](https://zhida.zhihu.com/search?content_id=231932826&content_type=Article&match_order=1&q=Meta&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzcwMDYxMjgsInEiOiJNZXRhIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjMxOTMyODI2LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.Jqt8RyZjFVL8GPLxrYAGvX758l3hHjlbSnWmLSxUuyc&zhida_source=entity) 的 LLAMA 和 清华的 ChatGLM 都使用了 RoPE 编码，下面看一下具体实现。

### 3.1 在LLAMA中的实现

```
# 生成旋转矩阵
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度\theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2] 
    freqs = torch.outer(t, freqs).float()  # 计算m * \theta

    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis

# 旋转位置编码计算
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    
    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)
    
    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.wq = Linear(...)
        self.wk = Linear(...)
        self.wv = Linear(...)
        
        self.freqs_cis = precompute_freqs_cis(dim, max_seq_len * 2)

    def forward(self, x: torch.Tensor):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(batch_size, seq_len, dim)
        xk = xk.view(batch_size, seq_len, dim)
        xv = xv.view(batch_size, seq_len, dim)

        # attention 操作之前，应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # scores.shape = (bs, seqlen, seqlen)
        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(dim)
        scores = F.softmax(scores.float(), dim=-1)
        output = torch.matmul(scores, xv)  # (batch_size, seq_len, dim)
  # ......
```

这里举一个例子，假设batch\_size=10, seq\_len=3, d=8，则调用函数precompute\_freqs\_cis(d, seq\_len)后，生成结果为：

```
In [239]: freqs_cis
Out[239]: 
tensor([[ 1.0000+0.0000j,  1.0000+0.0000j,  1.0000+0.0000j,  1.0000+0.0000j],
        [ 0.5403+0.8415j,  0.9950+0.0998j,  0.9999+0.0100j,  1.0000+0.0010j],
        [-0.4161+0.9093j,  0.9801+0.1987j,  0.9998+0.0200j,  1.0000+0.0020j]])
```

以结果中的第二行为例（对应的 m = 1），也就是：

$(\text{19}) \begin{matrix}c o s \left(\right. 1 * \theta_{0} \left.\right) & = c o s \left(\right. 1 \left.\right) = 0.5403 , & s i n \left(\right. 1 * \theta_{0} \left.\right) = s i n \left(\right. 1 \left.\right) = 0.8415 \\ c o s \left(\right. 1 * \theta_{1} \left.\right) & = c o s \left(\right. 0.1 \left.\right) = 0.9950 , & s i n \left(\right. 1 * \theta_{1} \left.\right) = s i n \left(\right. 0.1 \left.\right) = 0.0998 \\ c o s \left(\right. 1 * \theta_{2} \left.\right) & = c o s \left(\right. 0.01 \left.\right) = 0.9999 , & s i n \left(\right. 1 * \theta_{2} \left.\right) = s i n \left(\right. 0.01 \left.\right) = 0.0100 \\ c o s \left(\right. 1 * \theta_{3} \left.\right) & = c o s \left(\right. 0.001 \left.\right) = 1.0000 , & s i n \left(\right. 1 * \theta_{3} \left.\right) = s i n \left(\right. 0.001 \left.\right) = 0.0010\end{matrix}$

最终按照公式（12）可以得到编码之后的 $\mathbf{\mathit{q}} , \mathbf{\mathit{k}}$ 。

**注意：** 在代码中是直接用freqs\_cis\[0\] \* xq\_\[0\]的结果表示第一个 token 对应的旋转编码（和公式12计算方式有所区别）。其中将原始的 query 向量 $\mathbf{\mathit{q}}$ 转换为了复数形式。

```
In [351]: q_ = q.float().reshape(*q.shape[:-1], -1, 2)

In [352]: q_[0]
Out[352]: 
tensor([[[ 1.0247,  0.4782],
         [ 1.5593,  0.2119],
         [ 0.4175,  0.5309],
         [ 0.4858,  0.1850]],

        [[-1.7456,  0.6849],
         [ 0.3844,  1.1492],
         [ 0.1700,  0.2106],
         [ 0.5433,  0.2261]],

        [[-1.1206,  0.6969],
         [ 0.8371, -0.7765],
         [-0.3076,  0.1704],
         [-0.5999, -1.7029]]])

In [353]: xq = torch.view_as_complex(q_)

In [354]: xq[0]
Out[354]: 
tensor([[ 1.0247+0.4782j,  1.5593+0.2119j,  0.4175+0.5309j,  0.4858+0.1850j],
        [-1.7456+0.6849j,  0.3844+1.1492j,  0.1700+0.2106j,  0.5433+0.2261j],
        [-1.1206+0.6969j,  0.8371-0.7765j, -0.3076+0.1704j, -0.5999-1.7029j]])
```

**这里为什么可以这样计算？**

主要是利用了复数的乘法性质。

我们首先来复习一下复数乘法的性质：

$\left(\right. a + i b \left.\right) \cdot \left(\right. c + i d \left.\right) = a c + i b c + i a d + i^{2} b d = \left(\right. a c - b d \left.\right) + i \left(\right. b c + a d \left.\right)$

因此要计算：

$f_{q} \left(\right. \mathbf{\mathit{x}}_{m} , m \left.\right) & = \left(\right. \begin{matrix}cos ⁡ m \theta & - sin ⁡ m \theta \\ sin ⁡ m \theta & cos ⁡ m \theta\end{matrix} \left.\right) \left(\right. \begin{matrix}q_{m}^{\left(\right. 1 \left.\right)} \\ q_{m}^{\left(\right. 2 \left.\right)}\end{matrix} \left.\right) \\ = \left(\right. cos ⁡ m \theta * q_{m}^{\left(\right. 1 \left.\right)} - sin ⁡ m \theta * q_{m}^{\left(\right. 2 \left.\right)} , sin ⁡ m \theta * q_{m}^{\left(\right. 1 \left.\right)} - cos ⁡ m \theta * q_{m}^{\left(\right. 2 \left.\right)} \left.\right)$

可以转化为计算：

$\left(\right. cos ⁡ m \theta + i sin ⁡ m \theta \left.\right) \cdot \left(\right. q_{m}^{\left(\right. 1 \left.\right)} + i q_{m}^{\left(\right. 2 \left.\right)} \left.\right)$

所以可以将公式（12）转化为两个复数的乘法运算。

### 3.2 在ChatGLM中的实现

和 LLAMA 的实现方式相差不大。代码如下：

```
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
         # 计算 \theta_i
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        inv_freq = inv_freq.half()
        
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            # 对应m * \theta
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # 将 m * \theta 拼接两次，对应复数的实部和虚部
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]  # 计算得到cos(m*\theta)
            sin_cached = emb.sin()[:, None, :]  # 计算得到cos(m*\theta)
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn)

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)
```

## 4\. RoPE的外推性

我们都知道 RoPE 具有很好的外推性，前面的实验结果也证明了这一点。这里解释下具体原因。

RoPE 可以通过旋转矩阵来实现位置编码的外推，即可以通过旋转矩阵来生成超过预期训练长度的位置编码。这样可以提高模型的泛化能力和鲁棒性。

我们回顾一下 RoPE 的工作原理：假设我们有一个 $d$ 维的绝对位置编码 $P_{i}$ ，其中 $i$ 是位置索引。我们可以将 $P_{i}$ 看成一个 $d$ 维空间中的一个点。我们可以定义一个 $d$ 维空间中的一个旋转矩阵 $\mathbf{\mathit{R}}$ ，它可以将任意一个点沿着某个轴旋转一定的角度。我们可以用 $\mathbf{\mathit{R}}$ 来变换 $P_{i}$ ，得到一个新的点 $Q_{i} = \mathbf{\mathit{R}} * P_{i}$ 。我们可以发现， $Q_{i}$ 和 $P_{i}$ 的距离是相等的，即 $\left|\right. \left|\right. Q_{i} - P_{i} \left|\right. \left|\right. = 0$ 。这意味着 $Q_{i}$ 和 $P_{i}$ 的相对关系没有改变。但是， $Q_{i}$ 和 $P_{i}$ 的距离可能发生改变，即 $\left|\right. \left|\right. Q_{i} - P_{j} \left|\right. \left|\right. \neq \left|\right. \left|\right. P_{i} - P_{j} \left|\right. \left|\right.$ 。这意味着 $Q_{i}$ 和 $P_{j}$ 的相对关系有所改变。因此，我们可以用 $\mathbf{\mathit{R}}$ 来调整不同位置之间的相对关系。

如果我们想要生成超过预训练长度的位置编码，我们只需要用 $\mathbf{\mathit{R}}$ 来重复变换最后一个预训练位置编码 $P_{n}$ ，得到新的位置编码 $Q_{n + 1} = \mathbf{\mathit{R}} * P_{n} ， Q_{n + 2} = \mathbf{\mathit{R}} * Q_{n + 1} ， Q_{n + 3} = \mathbf{\mathit{R}} * Q_{n + 2}$ ，依此类推。这样就可以得到任意长度的位置编码序列 $Q_{1} , Q_{2} , \ldots , Q_{m}$ ，其中 $m$ 可以大于 $n$ 。由于 $\mathbf{\mathit{R}}$ 是一个正交矩阵，它保证了 $Q_{i}$ 和 $Q_{j}$ 的距离不会无限增大或缩小，而是在一个有限范围内波动。这样就可以避免数值溢出或下溢的问题。同时，由于 $\mathbf{\mathit{R}}$ 是一个可逆矩阵，它保证了 $Q_{i}$ 和 $Q_{j}$ 的距离可以通过 $\mathbf{\mathit{R}}$ 的逆矩阵 $\mathbf{\mathit{R}}^{- 1}$ 还原到 $P_{i}$ 和 $P_{j}$ 的距离，即 $\left|\right. \left|\right. \mathbf{\mathit{R}}^{- 1} * Q_{i} - \mathbf{\mathit{R}}^{- 1} * Q_{j} \left|\right. \left|\right. = \left|\right. \left|\right. P_{i} - P_{j} \left|\right. \left|\right.$ 。这样就可以保证位置编码的可逆性和可解释性。

总结而言：

**旋转编码 RoPE 可以有效地保持位置信息的相对关系** ，即相邻位置的编码之间有一定的相似性，而远离位置的编码之间有一定的差异性。这样可以增强模型对位置信息的感知和利用。这一点是其他绝对位置编码方式（如正弦位置编码、学习的位置编码等）所不具备的，因为它们只能表示绝对位置，而不能表示相对位置。

**旋转编码 RoPE 可以通过旋转矩阵来实现位置编码的外推** ，即可以通过旋转矩阵来生成超过预训练长度的位置编码。这样可以提高模型的泛化能力和鲁棒性。这一点是其他固定位置编码方式（如正弦位置编码、固定相对位置编码等）所不具备的，因为它们只能表示预训练长度内的位置，而不能表示超过预训练长度的位置。

**旋转编码 RoPE 可以与线性注意力机制兼容** ，即不需要额外的计算或参数来实现相对位置编码。这样可以降低模型的计算复杂度和内存消耗。这一点是其他混合位置编码方式（如Transformer-XL、XLNet等）所不具备的，因为它们需要额外的计算或参数来实现相对位置编码。

## 总结

最近一直听到旋转编码这个词，但是一直没有仔细看具体原理。今天花时间仔细看了一遍，确实理论写的比较完备，而且实验效果也不错。目前很多的大模型，都选择了使用了这种编码方式（LLAMA、GLM等）。

## 附录

这里补充一下前面公式1.3.2节中，公式（8）~（11）是怎么推导出来的。

回到之前的公式（8），编码之后的 $\mathbf{\mathit{q}} , \mathbf{\mathit{v}}$ 以及内积 $\langle \mathbf{\mathit{q}} , \mathbf{\mathit{v}} \rangle$ 的形式如下：

$f_{q} \left(\right. \mathbf{\mathit{x}}_{m} , m \left.\right) = \left(\right. \mathbf{\mathit{W}}_{q} \mathbf{\mathit{x}}_{m} \left.\right) e^{i m \theta} f_{k} \left(\right. \mathbf{\mathit{x}}_{n} , n \left.\right) = \left(\right. \mathbf{\mathit{W}}_{k} x_{n} \left.\right) e^{i n \theta} g \left(\right. \mathbf{\mathit{x}}_{m} , x_{n} , m - n \left.\right) = R e \left[\right. \left(\right. \mathbf{\mathit{W}}_{\mathbf{\mathit{q}}} x_{m} \left.\right) \left(\right. \mathbf{\mathit{W}}_{k} \mathbf{\mathit{x}}_{n} \left.\right)^{*} e^{i \left(\right. m - n \left.\right) \theta} \left]\right.$

上面的公式为什么满足： $\langle \mathbf{\mathit{f}}_{q} \left(\right. \mathbf{\mathit{x}}_{m} , m \left.\right) , f_{k} \left(\right. \mathbf{\mathit{x}}_{n} , n \left.\right) \rangle = g \left(\right. \mathbf{\mathit{x}}_{m} , \mathbf{\mathit{x}}_{n} , m - n \left.\right)$ 。

首先我们得先了解一下基本的复数相关知识。

首先看到上述 $f$ 和 $g$ 公式中有个指数函数： $e^{i x}$

这个其实是欧拉公式，其中 $x$ 表示任意实数， $e$ 是自然对数的底数， $i$ 是复数中的虚数单位，则根据欧拉公式有：

$e^{i x} = cos ⁡ x + i sin ⁡ x$

则是上述指数函数可以表示为实部为 $cos ⁡ x$ ，虚部为 $sin ⁡ x$ 的一个复数，欧拉公式建立了指数函数、三角函数和复数之间的桥梁。

则上述 $f$ 和 $g$ 公式的

$e^{i m \theta} = cos ⁡ \left(\right. m \theta \left.\right) + i sin ⁡ \left(\right. m \theta \left.\right) e^{i n \theta} = cos ⁡ \left(\right. n \theta \left.\right) + i sin ⁡ \left(\right. n \theta \left.\right) e^{i \left(\right. m - n \left.\right) \theta} = cos ⁡ \left(\right. \left(\right. m - n \left.\right) \theta \left.\right) + i sin ⁡ \left(\right. \left(\right. m - n \left.\right) \theta \left.\right)$

然后我们看回公式：

$f_{q} \left(\right. \mathbf{\mathit{x}}_{m} , m \left.\right) = \left(\right. \mathbf{\mathit{W}}_{q} \mathbf{\mathit{x}}_{m} \left.\right) e^{i m \theta}$

其中 是个二维矩阵， 是个二维向量，相乘的结果也是一个二维向量，这里用 表示：

然后首先将 表示成复数形式：

接着

其实就是两个复数相乘：

然后就有：

将结果重新表达成实数向量形式就是：

**这里不难发现就是 query 向量乘以了一个旋转矩阵** 。

**这就是为什么叫做旋转式位置编码的原因。**

同理可得 key 向量 ：

最后还有个函数 ：

其中 表示一个复数 的实部部分，而 则表示复数 的共轭。

复习一下共轭复数的定义：

所以可得：

继续可得：

接下来我们就要证明函数 的计算公式是成立的。

首先回顾一下 attention 操作， 位置 的 query 和位置 的 key 会做一个内积操作：

接着进行推导，我们整理一下：

这就证明上述关系是成立的，位置 的 query 和位置 的 key 的内积就是函数 。

把上面的式子用矩阵向量乘的形式来表达就是：

## 参考

[ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2104.09864.pdf)

[梁德澎：一文看懂 LLaMA 中的旋转式位置编码（Rotary Position Embedding）](https://zhuanlan.zhihu.com/p/642884818)

[马梦之：一步一步，推导旋转位置编码 (Rotary Position Embedding, RoPE)](https://zhuanlan.zhihu.com/p/644585013)

[苏剑林：Transformer升级之路：2、博采众长的旋转式位置编码](https://zhuanlan.zhihu.com/p/359502624)

编辑于 2026-02-02 17:51・北京[旋转门](https://www.zhihu.com/topic/19576328)[编码](https://www.zhihu.com/topic/19590100)[编码技术](https://www.zhihu.com/topic/19629442)