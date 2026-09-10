---
title: "一文通透TTT：Learning to “Learn at Test Time”，让RNN的隐藏层变成可学习的函数，把T"
source: "https://blog.csdn.net/v_JULY_v/article/details/140610924"
author:
  - "[[v_JULY_v]]"
published: 2024-07-22
created: 2026-08-26
description: "文章浏览阅读1.6w次，点赞56次，收藏69次。TTT出来有一段时间了，让我确定要写TTT解读的，是源于我司LLM论文100篇课程群里的一学员辰子说，“校长 最近的TTT考不考虑讲一下”故当时想着：解读完mamba2之后，则解读open-television、我司7方面review微调gemma2，再接下来是TTT、nature审稿微调、序列并行、Flash Attention3..如今虽然mamba2的解读还没完全修订完，但“open-television、我司7方面review微调gemma2”都解读的差不多了，故今天开写TTT。_ttt layer"
tags:
  - "clippings"
---
## 前言

截止到24年7月份，TTT出来有一段时间了，让我确定要写TTT解读的，是源于

1. 我司LLM论文100篇课程群里的一学员辰子说，“校长 最近的TTT考不考虑讲一下”，以及微博上也有朋友问：“老师，TTT 怎么说？”  
	  
	故当时想着：解读完mamba2之后，则解读open-television、我司7方面review微调gemma2，再接下来是TTT、nature审稿微调、序列并行、Flash Attention3、vLLM(*Efficient memory management for large language model serving with pagedattention*)..
2. 如今虽然mamba2的解读还没完全修订完，但“ *open-television、我司7方面review微调gemma2* ”都解读的差不多了

故今天开写TTT，对应论文为： [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620 "Learning to (Learn at Test Time): RNNs with Expressive Hidden States")

![](https://i-blog.csdnimg.cn/direct/3420442549f244a2becedc3619b36bde.png)

比我科研水平高的人大有人在，但比我写的 大模型 技术博客还要更通俗易懂的则寥寥无几，究其原因，还是在于比他人花更多的时间和心思

- 对于他人，写一篇博客可能只愿意花1-2天时，我愿意花1-2周
- 对于他人，改一篇博客可能只愿意花1-2周时，我愿意花1-2个月

核心动力还是来源于写博十多年下来积累的习惯：尽我最大努力，让最广大的读者可以最大程度的最快理解

## 第一部分 TTT的关键探索与关键方法

### 1.1 TTT提出之前的背景与关键探索

#### 1.1.1 困境：RNN的线性复杂度与面对长下文时的局限

虽说相对于transformer的二次复杂度，RNN有着线性的复杂度

当然，如果序列长度比较短，那么无论是二次还是线性，其实区别不大，只有在序列长度比较长，即 长上下文 时，RNN这个线性的复杂度才有比较高的价值或意义

那到底多长之后，会使得RNN这个线性复杂度具有极高的价值或意义呢，如下图所示，这个 上下文长度 是在8k之后

可问题是一旦上下文足够长后，现有的RNN如Mamba在实际利用这些额外信息时会遇到困难(*On the other hand,once context is long enough, existing RNNs such as Mamba struggle to actually take advantage ofthe extra information being conditioned on*)

为何RNN在面对长下文时处理起来会比较困难呢，原因在于 ***与自注意力不同，RNN 层必须将上下文压缩到一个固定大小的隐藏状态中。作为一种压缩启发式方法，其更新规则需要从成千上万甚至可能数百万个 token 中挖掘出潜在的结构和关系***

> 如下图所示，先根据输入和前一时刻的隐藏状态计算出最新的隐藏状态，在此之后，便可以 **根据最新的隐藏状态预测出** 了
> 
> ![](https://i-blog.csdnimg.cn/direct/2620f031426142a58dfa4c997f965937.png)
> 
> ---
> 
> 至于RNN的详细介绍，详见此文： [如何从RNN起步，一步一步通俗理解LSTM](https://blog.csdn.net/v_JULY_v/article/details/89894058 "如何从RNN起步，一步一步通俗理解LSTM")

#### 1.1.2 对比：RNN与Transformer的各自优劣

其实，所有序列建模层都可以从将历史上下文存储到隐藏状态的角度来看，比如RNN层——如LSTM \[33\]、RWKV \[56\]和Mamba \[26\]层——在时间上将上下文压缩到一个固定大小的状态中

如下图所示(*如你所见，一个通用的序列建模层表示为一个根据更新规则转换的隐藏状态。所有序列建模层都可以看作是该图中三个组件的不同实例：初始状态、更新规则和输出规则*)

![](https://i-blog.csdnimg.cn/direct/eb7ac746c8d449f59d2b23bd1f9f6195.png)

1. 简单的RNN和TTT层都 **将增长的上下文压缩成固定大小的隐藏状态(***Both the naive RNN and TTT layer compress the growing context into a hidden state of fixed size, therefore their cost per token stays constant***)**  
	  
	然这种压缩有好有弊  
	$\rightarrow$ 好的是，其输入token 到输出token 的映射十分高效，即每个token的更新与输出的时间都是常量时间，成本可控  
	*mapping an input token xt to output token zt is efficient, because both the update rule and output rule take constant time per token*  
	$\rightarrow$ 不好的是，RNN层在长上下文中的性能受到其隐藏状态的表达能力的限制——即表达能力不强(*the performance of RNN layers in long context is limited by the expressive power of its hidden states st*)  
	而只有隐藏状态表达的好，才能让最终的整体性能好，否则 则难说了..
2. 自注意力机制的隐藏状态(通常称为键值KV缓存)随着上下文增长，因此每个token的成本也在增长(随着线性增长)
	|  | **我** | **是** | **中** | **国** | ？ |
	| --- | --- | --- | --- | --- | --- |
	| 我 | **q1 k1** | **q2 k1** | **q3 k1** | **q4 k1** |  |
	| 爱 | **q1 k2** | **q2 k2** | **q3 k2** | **q4 k2** |  |
	| 中 | **q1 k3** | **q2 k3** | **q3 k3** | **q4 k3** |  |
	| 国 | **q1 k4** | **q2 k4** | **q3 k4** | **q4 k4** |  |
	总之，这里的隐藏状态明确的存储了所有历史上下文，而不进行压缩  
	好的是，表达能力强；不好的是，成本增加快

所以现在的问题变成了，我们既希望将成千上万甚至数百万的token压缩到一个隐藏状态中，且该隐藏状态还能有效的捕捉它们的底层结构和关系(*we need to compress thousands or potentially millions of tokens into a hidden state that can effectively capture their underlying structures and relationship*)，从而兼顾高效与质量

### 1.2 TTT更新隐藏状态的关键：输出规则和更新规则

过去的一年半，LLM火爆全球

1. 其通过自监督的下一个词预测任务——即 **下一个token预测** 进行训练，它们的权重可以被视为互联网上现有知识的压缩存储形式  
	(*所以对于咱们大模型开发者而言，预训练或微调后得到的模型权重是最关键的产物，如果一个模型的权重没开源，则不算开源*)
2. 通过查询 LLMs，可以从它们的权重中提取知识  
	更重要的是，LLMs 通常表现出对现有知识之间语义连接的深刻理解，以表达新的推理片段

总之，种种迹象表明，可以

1. ***使用自监督学习来将历史上下文 $x_{1}, \ldots, x_{t}$ ，压缩到一个* *隐藏状态中* *，通过将上下文变成一个无标签的数据集*** ，并 *将隐藏状态变成一个模型*  
	(*Our key idea is to use self-supervised learning to compress the historic context x1,..., xt into a hidden state st, by making the context an unlabeled dataset and the state a model*)
2. 具体而言，可以将 *隐藏状态现在等同于 ——模型的参数(权重)，模型可以是线性  
	模型、小型補经网络或其他任何模型*  
	(***the hidden state st is now equivalent to Wt, the weights of a model f, which can be a linear model, a small neural network, or anything else***)
	![](https://i-blog.csdnimg.cn/direct/7eb55e6f535b45d99635ad572bade82d.png)

而TTT通过建模为模型的权重来更新隐藏状态的过程中，涉及到：输出规则、更新规则

1. 对于前者， **输出规则** 很简单，即为公式1  
	$z_{t}=f\left(x_{t} ; W_{t}\right)$  
	  
	直白讲，输出token 只是对 的预测  
	怎么预测呢？由使用更新后权重的给出  
	(*the output token is just the prediction on xt, made by f with the updated weights Wt*)  
	  
	为方便大家更好的理解，我举个例子，如下表格所示：预测“我 是 中 国”之后的下一个token：人
	|  | 我 |  |  |  |  |
	| --- | --- | --- | --- | --- | --- |
	|  | 我 | 是 |  |  |  |
	|  | 我 | 是 | 中 |  |  |
	| \= 对{ ***我 + 是 + 中 + 国*** }的压缩 | 我 | 是 | 中 | 国 |  |
	| $W_{t}=W_{t-1}-\eta \nabla \ell\left(W_{t-1} ; x_{t}\right)$   其中， = { *或是“人”、或是“的”* } | 我 | 是 | 中 | 国 | 基于，可得的预测 = 人 |
2. 对于后者， **更新规则** ，是对某些自监督损失进行梯度下降的一步(记为公式2)：  
	更新规则是在某个自监督损失 上的一次梯度下降步骤：  
	  
	$W_{t}=W_{t-1}-\eta \nabla \ell\left(W_{t-1} ; x_{t}\right)$  
	  
	其中  
	$\rightarrow$ 学习率为 $\eta$ ，从压缩的角度来看，每个启发式方法都需要决定记住或忘记哪个输入  
	而 **此处的记住：产生大梯度的输入** ——直观地说，就是那些让 学到很多的输入  
	$\rightarrow$ 此外，其中一种选择loss 的方法是重构本身，为了重建，可以将处理成一个带噪输入 $\tilde{x}_{t}$ ，然后如下优化(记为公式3)  
	$\ell\left(W ; x_{t}\right)=\left\|f\left(\tilde{x}_{t} ; W\right)-x_{t}\right\|^{2}$  
	  
	这个表达式类似去噪自编码器 \[75\]，需要发现 各个维度之间的相关性，以便从部分信息 $\tilde{x}_{t}$ 中重建它(*f needs to discover the correlations between dimensions of xt in order to reconstruct it from partial information ˜xt*)  
	  
	————  
	如下图所示，梯度下降能够减少损失，但不能将其减少到零  
	![](https://i-blog.csdnimg.cn/direct/94557923ab964a8fba21a03f8bd8d079.png)  
	*如上图所示，自监督TTT损失 ℓ在所有测试序列上的平均值形式为 $x_{1}, \ldots, x_{T}$ ，其中 ，对于具有125M参数的网络中的前三个TTT层  
	一次梯度下降能够将TTT损失从 $\ell\left(W_{t-1} ; x_{t}\right)$ 降低到 $\ell\left(W_{t} ; x_{t}\right)$ ，且随着 沿测试序列进一步移动， $\ell\left(W_{t} ; x_{t}\right)$ 也进一步从 $\ell\left(W_{0} ; x_{t}\right)$ 改善  
	另，为便于可视化，对损失值在长度为 10 个时间步的滑动窗口上进行了平均*

与其他RNN层和 自注意力 机制一样，算法将输入序列 x1,..., xT映射到输出序列 z1,..., zT可以通过使用上述隐藏状态、更新规则和输出规则编程到序列建模层的前向传递中

即使在测试时，新层仍然训练一组不同的权重序列W1,..., WT——对于每个输入序列，因此，称之为测试时训练(TTT)层

且

1. TTT 层的前向传播同样具有对应的反向传播  
	$W_{t}=W_{t-1}-\eta \nabla \ell\left(W_{t-1} ; x_{t}\right)$  
	前向传播仅由标准的可微算子组成，除了梯度算子∇。然而，∇只是映射从一个函数到另一函数，比如 到 $\nabla \ell$ ，并且 ∇ℓ也是由可微算子构成的  
	从概念上讲，对 ∇ℓ 调用反向传播，意味着对梯度进行梯度计算
2. TTT 层与 RNN 层和自注意力层具有相同的接口，因此可以在任意更大的网络架构中进行替换，而这些架构通常包含许多此类序列建模层  
	*使用 TTT 层来训练网络的方式也与训练其他语言模型（例如 Transformer）相同。可以使用相同的数据、训练流程和目标（如下一token的预测）来优化网络其余部分的参数*  
	  
	作者，将 训练更大的网络称为外层循环，而在每个 TTT 层内训练 称为内层循环  
	两个嵌套学习问题之间的一个重要区别是  
	$\rightarrow$ 内层循环梯度 $\nabla \ell$ 是相对于(即的参数)计算的  
	$\rightarrow$ 而外层循环梯度是相对于网络其余部分的参数计算的，将其表示为 $\theta_{\text {rest }}$ (*外循环参数始终用 $\theta$ 表示，并带有各种下标*)

最后，你将在下文依次看到更多的公式(公式4-7还不理解 没事，继续看下文即可)

<table><tbody><tr><td><strong>输出规则</strong> (外循环)</td><td>公式1<br><math>z_{t}=f\left(x_{t}; W_{t}\right)</math></td><td rowspan="2">公式3<br><math>\ell\left(W; x_{t}\right)=\left\|f\left(\tilde{x}_{t}; W\right)-x_{t}\right\|^{2}</math></td><td>公式4<br><math>\ell\left(W; x_{t}\right)=\left\|f\left(\theta_{K} x_{t}; W\right)-\theta_{V} x_{t}\right\|^{2}</math></td><td>公式5<br><math>z_{t}=f\left(\theta_{Q} x_{t}; W_{t}\right)</math></td><td></td><td></td></tr><tr><td><strong>更新规则</strong> (内循环)</td><td>公式2<br><math>W_{t}=W_{t-1}-\eta \nabla \ell\left(W_{t-1}; x_{t}\right)</math></td><td></td><td></td><td>公式6<br><math>W_{t}=W_{t-1}-\eta G_{t}=W_{0}-\eta \sum_{s=1}^{t} G_{s}</math></td><td>公式7</td></tr></tbody></table>

#### 1.2.1 对于输出规则——使用TTT层训练隐藏层网络：重建x中设计损失函数

考虑到TTT的最终目标是使公式1—— $z_{t}=f\left(x_{t} ; W_{t}\right)$ 在语言建模上表现良好，故可以基于人类先验知识的自监督任务，采用一种更端到端的方法——直接优化自监督任务以实现下一个词预测的最终目标

具体来说，在外循环中学习自监督任务

> ![](https://i-blog.csdnimg.cn/direct/7eb55e6f535b45d99635ad572bade82d.png)

1. 从公式3 $\ell\left(W ; x_{t}\right)=\left\|f\left(\tilde{x}_{t} ; W\right)-x_{t}\right\|^{2}$ 中的朴素重建任务出发，加入一些外循环参数，使这一任务变得可学习
2. 为产生 $\tilde{x}_{t}$ 从的损坏  
	$\rightarrow$ 一种设计是使其成为低秩投影 $\tilde{x}_{t}=\theta_{K} x_{t}$ ，其中 $\theta_{K}$ 是一个可学习的矩阵，总之， $\theta_{K} x_{t}$ 被称为训练view  
	$\rightarrow$ 由于并不是中的所有信息都值得记住，因此重建标签可以是另一个低秩投影 $\theta_{V} x_{t}$ 而不是，这里 $\theta_{V} x_{t}$ 被称为标签view，其中 $\theta_{V}$ 也是可学习的  
	  
	因此，新的自监督损失由公式3 $\ell\left(W ; x_{t}\right)=\left\|f\left(\tilde{x}_{t} ; W\right)-x_{t}\right\|^{2}$ 变成如下所示(**公式4**)  
	$\ell\left(W ; x_{t}\right)=\left\|f\left(\theta_{K} x_{t} ; W\right)-\theta_{V} x_{t}\right\|^{2}$  
	  
	注意，因为在内循环中，只有被优化，所以写作为loss 的一个参数，而 *$\theta$* 是这个损失函数的超参数  
	在外循环中， $\theta_{K}, \theta_{V}, \theta_{Q}$ 与 $\theta_{\text {rest }}$ 一起被优化，而只是一个隐藏状态，而非一个参数  
	  
	如下代码所示，其中 $\theta_{K}$ 和 $\theta_{V}$ 作为TTT层的参数实现，类似于自注意力的Key和Value参数(  
	*TTT\_Layer可以像其他序列建模层一样被插入到更大的网络中  
	**$\rightarrow$** **对于下图左边** ，训练网络将优化TTT\_Layer中Task的参数，因为两者都是nn.Modul的子类  
	**$\rightarrow$** **对于下图右边** ，由于Learner不是nn.Module的子类，因此在每次调用 state.train 的内层循环中，需要手动更新 state.model* *。为了简化，有时将model重载为model.parameter* ![](https://i-blog.csdnimg.cn/direct/8bf1f502c4e14f4a82bb1585c9c7e1d2.png)
3. 最后，训练view $\theta_{K} x_{t}$ 的维度比少，因此不能再使用方程 $z_{t}=f\left(x_{t} ; W_{t}\right)$ 中的输出规则  
	那怎么办呢？最简单的解决方案是创建一个测试view $\theta_{Q} x_{t}$ ，并将输出规则更改为(**公式5**)：  
	$z_{t}=f\left(\theta_{Q} x_{t} ; W_{t}\right)$

弄这么多不同view的好处是什么呢？

- 首先，训练view $\theta_{K} x_{t}$ 和标签view $\theta_{V} x_{t}$ 指定了在中被压缩到 并随时间向前传播的信息
- 其次，测试视图 $\theta_{Q} x_{t}$ 指定了可能不同的信息，这些信息被映射到当前输出token 并通过网络层向前传播，因此为自监督任务增加了更多的灵活性

#### 1.2.2 对于更新规则——使用小批量TTT进行并行化：小批量梯度下降

目前的TTT有个比较明显的问题是，其更新规则——公式2 $W_{t}=W_{t-1}-\eta \nabla l\left(W_{t-1} ; x_{t}\right)$ 不能被并行化，因为在两个地方依赖于

1. 一个地方是在减号前
2. 一个地方在 $\nabla l$ 内部

而由于后者 $\nabla l$ 内部包含了大部分计算， *所以我们要重点对后者—— $\nabla l$ 内部，做并行化*

首先，梯度下降GD有许多变体，GD的一般更新规则可以表示为(记为公式6)

$$
W_{t}=W_{t-1}-\eta G_{t}=W_{0}-\eta \sum_{s=1}^{t} G_{s}
$$

其中，是下降方向，这个公式的价值和意义在于，一旦计算出 $G_{t} \text { for } t=1, \ldots, T$ ，便可以基于上述公式的后半部分做累加和来得到所有的 $W_{t} \mathrm{~s}$

而对于我们所想要的更新规则是在线梯度下降，使用 $G_{t}=\nabla l\left(W_{t-1} ; x_{t}\right)$

1. 为了并行化 $G_{t} \text { for } t=1, \ldots, T$ ，可以对所有这些变量相对于进行计算  
	  
	这种令 $G_{t}=\nabla \ell\left(W_{0} ; x_{t}\right)$ 的变体被称为 批量梯度下降  
	因为 $\sum_{s=1}^{t} \nabla \ell\left(W_{0} ; x_{t}\right)$ 与相对于、在 $x_{1}, \ldots, x_{t}$ 这一批数据上的梯度是相同的
2. 不过，在批量梯度下降中，实际上只比多一步梯度步长「 *这与在线梯度下降形成对比，因为在 **在线梯度下降** ，距离有步之遥* 」，因此，批量梯度下降的有效搜索空间会相对比较小
3. 怎么办呢，好在可以在 批量梯度下降 与 在线梯度下降 之中，取个折中： **小批量梯度下降** ，即将批量大小设置为相对较小的b，具体如下图所示
	![](https://i-blog.csdnimg.cn/direct/6ad1230188bf4b29bea964f4baf1aae6.png)
4. 然后使用 $G_{t}=\nabla \ell\left(W_{t^{\prime}} ; x_{t}\right)$ ，其中 $t^{\prime}=t-\bmod (t, b)$ 是前一个小批量的最后一个时间步(对于第一个小批量为0，有点类似 $t^{\prime}=t-1$ 的意思)，因为可以一次并行化个梯度计算

进一步，这个其实控制着速度与质量之间的权衡，如下图所示(*在TTT的所有实验中，作者们选择*)

![](https://i-blog.csdnimg.cn/direct/fc92141b04054ed2802ec7cdbcd9aecc.png)

总之，有两个潜在的渠道可以将信息从 传播到，其中：对梯度算子做累加和(cumsum and the gradient operator)

1. 累加和始终是激活的，但梯度通道只有在来自先前的小批量时才会激活。 不同变体的梯度下降只会影响梯度通道，即，下降方向，具体来说是关于对哪个 求梯度
2. 然而，由于更新规则的自回归特性，下降步骤 $W_{t}=W_{t-1}-\eta G_{t}$ 总是从开始，这与的选择无关

然，上面面介绍的并行化是必要的，但对于wall-clock time的效率而言还不够(说白了，就是速度还不够快)，故接下来，咱们来探讨下 **对偶形式**

现代加速器专门用于矩阵-矩阵乘法，称为 matmuls。例如，NVIDIA A100 GPU 包含高度优化的单元，称为 TensorCores，它们只能执行一种操作——将两个大小为 16 ×16 的矩阵相乘。 如果没有足够的矩阵乘法，TensorCores 就会闲置，A100 的大部分潜力将无法实现

不幸的是，即使使用小批量开发的 TTT 层仍然有很少的矩阵乘法

考虑 ℓ的最简单情况，其中 $\theta_{K}=\theta_{V}=\theta_{Q}=I$ ，仅针对第一个大小为 b的 TTT 小批量

此外，考虑作为线性模型，复制公式3 $\ell\left(W ; x_{t}\right)=\left\|f\left(\tilde{x}_{t} ; W\right)-x_{t}\right\|^{2}$ ，在时间的损失为：

$$
\ell\left(W_{0} ; x_{t}\right)=\left\|f\left(x_{t} ; W_{0}\right)-x_{t}\right\|^{2}=\left\|W_{0} x_{t}-x_{t}\right\|^{2}
$$

> 为帮助大家更好的理解，回顾一下之前的公式1、公式2、公式3，以及公式6
> 
> <table><tbody><tr><td>输出规则(外循环)</td><td><strong>公式1<br><math>z_{t}=f\left(x_{t}; W_{t}\right)</math></strong></td><td rowspan="2"><strong>公式3<br><math>\ell\left(W; x_{t}\right)=\left\|f\left(\tilde{x}_{t}; W\right)-x_{t}\right\|^{2}</math></strong></td><td>公式4<br><math>\ell\left(W; x_{t}\right)=\left\|f\left(\theta_{K} x_{t}; W\right)-\theta_{V} x_{t}\right\|^{2}</math></td><td>公式5<br><math>z_{t}=f\left(\theta_{Q} x_{t}; W_{t}\right)</math></td><td></td></tr><tr><td>更新规则(内循环)</td><td><strong>公式2<br><math>W_{t}=W_{t-1}-\eta \nabla \ell\left(W_{t-1}; x_{t}\right)</math></strong></td><td></td><td></td><td><strong>公式6<br><math>W_{t}=W_{t-1}-\eta G_{t}=W_{0}-\eta \sum_{s=1}^{t} G_{s}</math></strong></td></tr></tbody></table>

如上一节所讨论的，可以并行化以下计算

$$
G_{t}=\nabla \ell\left(W_{0} ; x_{t}\right)=2\left(W_{0} x_{t}-x_{t}\right) x_{t}^{T},
$$

对于 t= 1,..., b，然而不能通过一个单一的matmul计算所有的b个

相反，需要 b个外积来逐个计算它们。更糟糕的是，对于每个 $x_{t} \in \mathbb{R}^{d}$ ，是 d × d，这会比在大 时产生更大的内存占用和I/O成本

为了解决这两个问题，他们做了一个简单的观察：实际上不需要具体化 $G_{1}, \ldots, G_{b}$ ，只要我们能在小批量结束时计算 和 输出token $z_{1}, \ldots, z_{b}$

> ![](https://i-blog.csdnimg.cn/direct/6ad1230188bf4b29bea964f4baf1aae6.png)

现在用上面简化的TTT-Linear案例来演示这些计算

1. 记 X= \[x1,...., xb\], 然后  
	![](https://i-blog.csdnimg.cn/direct/c3d4576d31844805860a677704e1d2a8.png)
2. 所以 Wb可以通过 matmul方便地计算。 为了计算 Z= \[z1,.............., zb\]，我们知道(记为 **公式7**)  
	![](https://i-blog.csdnimg.cn/direct/59fec385a5794896bc66c07d07238d94.png)
3. 记 $\delta_{t}=\sum_{s=1}^{t}\left(W_{0} x_{s}-x_{s}\right) x_{s}^{T} x_{s}$ 和矩阵 $\Delta=\left[\delta_{1}, \ldots, \delta_{b}\right]$ ，可以推导出  
	$\Delta=\operatorname{mask}\left(X^{T} X\right)\left(W_{0} X-X\right)$

其中掩码是具有零值的下三角掩码(类似于注意力掩码，但用零代替无穷大)，并且项可  
以从的计算中重用。 现在 ∆也可以方便地用矩阵乘法计算。将 ∆代入公式7，我们得到

$$
Z=W_{0} X-2 \eta \Delta
$$

以上，称这个过程为对偶形式，与之前的原始形式相对比，其中 G和 W是显式物化的，如前所述，这两种形式在输出上是等价的

## 第二部分 两种TTT层的变体——TTT-Linear和TTT-MLP

### 2.1 证明：具有线性模型和批量GD的TTT层等价于线性注意力

#### 2.1.1 定理1及其证明

回顾一下

1. 在1.2节的开头，提到 f可以是线性模型或神经网络
2. 在1.2.2小节中，还讨论了更新规则的三种变体：在线GD、批量GD和小批量GD  
	这三种2 ×3组合中的每一种都会引发TTT层的不同实例化，如下图所示
	![](https://i-blog.csdnimg.cn/direct/889dad8ef4644e80b3b187c74d05bb9b.png)
	*如上图所示，参数化学习器需要定义两个属性：模型和优化器(左），每个学习器唯一地引发一个TTT层(右)*  
	*本文则提出了两种引发的TTT层：TTT-Linear和TTT-MLP，具有线性模型和批量GD的TTT层等价于线性注意力\[41\]*

其实，在这些引发的实例化中，具有线性模型和批量GD的TTT层其实等价于线性注意力\[*Transformers are rnns: Fast autoregressive transformers with linear attention*\]——一种广为人知的RNN层

> 简而言之，线性注意力 \[41\] 只是没有softmax的自注意力。 回顾自注意力的定义：
> 
> $$
> z_{t}=V_{t} \operatorname{softmax}\left(K_{t}^{T} q_{t}\right)
> $$
> 
> 没有 softmax, 其便变成了
> 
> $$
> z_{t}=V_{t}\left(K_{t}^{T} q_{t}\right)=\sum_{s=1}^{t} v_{s} k_{s}^{T} q_{t}
> $$
> 
> 这是线性注意力的最简单形式
> 
> ---
> 
> 与其他RNN层类似，它可以写成递归形式，其中 $\sum_{s=1}^{t} v_{s} k_{s}^{T}$ 是隐藏状态  
> 而由于 $\sum_{s=1}^{t} v_{s} k_{s}^{T}$ 可以通过 cumsum在每个 $t=1, \ldots, T$ 计算，因此线性注意力相对于 T也具有线性复杂度

**定理1** 考虑TTT层，其中作为内循环模型，批量梯度下降， $\eta=1 / 2$ 作为更新规则，并且

然后，给定相同的输入序列 $x_{1}, \ldots, x_{T}$ ，公式5 $z_{t}=f\left(\theta_{Q} x_{t} ; W_{t}\right)$ 中定义的输出规则产生相同的输出序列 $z_{1}, \ldots, z_{T}$ 作为线性注意力

> 且为方便大家理解，特此再列一下上文介绍过的公式4、公式5、公式6
> 
> <table><tbody><tr><td><strong>输出规则</strong> (外循环)</td><td>公式1<br><math>z_{t}=f\left(x_{t}; W_{t}\right)</math></td><td rowspan="2">公式3<br><math>\ell\left(W; x_{t}\right)=\left\|f\left(\tilde{x}_{t}; W\right)-x_{t}\right\|^{2}</math></td><td><strong>公式4<br><math>\ell\left(W; x_{t}\right)=\left\|f\left(\theta_{K} x_{t}; W\right)-\theta_{V} x_{t}\right\|^{2}</math></strong></td><td><strong>公式5<br><math>z_{t}=f\left(\theta_{Q} x_{t}; W_{t}\right)</math></strong></td><td></td></tr><tr><td><strong>更新规则</strong> (内循环)</td><td>公式2<br><math>W_{t}=W_{t-1}-\eta \nabla \ell\left(W_{t-1}; x_{t}\right)</math></td><td></td><td></td><td><strong>公式6<br><math>W_{t}=W_{t-1}-\eta G_{t}=W_{0}-\eta \sum_{s=1}^{t} G_{s}</math><br>其中</strong><br><u><math>G_{t}=\nabla \ell\left(W_{0}; x_{t}\right)</math></u></td></tr></tbody></table>

该定理的证明如下

根据公式4 $\ell\left(W ; x_{t}\right)=\left\|f\left(\theta_{K} x_{t} ; W\right)-\theta_{V} x_{t}\right\|^{2}$ 中 的定义，有：

$$
\nabla \ell\left(W_{0} ; x_{t}\right)=-2\left(\theta_{V} x_{t}\right)\left(\theta_{K} x_{t}\right)^{T}
$$

且根据公式6 **$W_{t}=W_{t-1}-\eta G_{t}=W_{0}-\eta \sum_{s=1}^{t} G_{s}$** 中的批量GD定义 $G_{t}=\nabla \ell\left(W_{0} ; x_{t}\right)$ ，可知

![](https://i-blog.csdnimg.cn/direct/f6bcf81556134cd49fc74f8abe61a66b.png)

将 代入公式5 $z_{t}=f\left(\theta_{Q} x_{t} ; W_{t}\right)$ 中的输出规则，可得到输出token

$$
z_{t}=f\left(\theta_{Q} x_{t} ; W_{t}\right)=\sum_{s=1}^{t}\left(\theta_{V} x_{s}\right)\left(\theta_{K} x_{s}\right)^{T}\left(\theta_{Q} x_{t}\right)
$$

这是线性注意力的定义

#### 2.1.2 定理2

**定理2** 考虑使用Nadaraya-Watson估计器\[7, 12\]定义的TTT层：

$$
f\left(x ; x_{1}, \ldots, x_{t}\right)=\frac{1}{\sum_{s=1}^{t} \kappa\left(x, x_{s}\right)} \sum_{s=1}^{t} \kappa\left(x, x_{s}\right) y_{s}
$$

其中 $y_{s}=\theta_{V} x_{s}$ 是第1.2.1节讨论的标签view

> 1. 从公式3 $\ell\left(W ; x_{t}\right)=\left\|f\left(\tilde{x}_{t} ; W\right)-x_{t}\right\|^{2}$ 中的简单重建任务开始，添加了一些外循环参数，使这个任务可以学习
> 2. 为产生 $\tilde{x}_{t}$ 从的损坏  
> 	$\rightarrow$ 一种设计是使其成为低秩投影 $\tilde{x}_{t}=\theta_{K} x_{t}$ ，其中 $\theta_{K}$ 是一个可学习的矩阵，总之， $\theta_{K} x_{t}$ 被称为训练view  
> 	$\rightarrow$ 由于并不是中的所有信息都值得记住，因此重建标签可以是另一个低秩投影 $\theta_{V} x_{t}$ 而不是，这里 $\theta_{V} x_{t}$ 被称为标签view，其中 $\theta_{V}$ 也是可学习的

并且

$$
\kappa\left(x, x^{\prime} ; \theta_{K}, \theta_{Q}\right) \propto e^{\left(\theta_{K} x\right)^{T} \theta_{Q} x^{\prime}}
$$

是一个带有带宽超参数的核函数 $\theta_{K}$ 和 $\theta_{Q}$ ，然后给定相同的输入序列 $x_{1}, \ldots, x_{T}$, 公式5 $z_{t}=f\left(\theta_{Q} x_{t} ; W_{t}\right)$ 中定义的输出规则产生与自注意力相同的输出序列 $z_{1}, \ldots, z_{T}$

1. 对于上面的TTT层，隐藏状态是 $x_{1}, \ldots, x_{t}$ 或类似的处理过的训练数据列表，更新规则将添加到列表中，输出规则使用 扫描列表
2. 在前面的章节中，隐藏状态被定义为 ，更新规则是一个梯度步骤，输出规则是调用

为了统一这两种构造，可定义一种新的抽象，称为学习者，它唯一地引发了TTT层

类似于标准 机器学习 包中的定义 \[54\]，所有学习者都需要实现两个方法：训练和预测。 现在将induced的TTT层的隐藏状态重新定义为学习者的内部存储，并将更新和输出规则重新定义为训练和预测方法

在这种新的TTT层定义下，定理1中的参数学习器和定理2中的非参数学习器都可以包括在内，比如下图便总结了在所有序列建模层的更广泛范围内TTT层的这一通用定义

![](https://i-blog.csdnimg.cn/direct/d5ddae9edb0245bb99fbe70f38fceca6.png)

这种通用定义对参数学习器有一个额外的好处：在参数学习器的内部存储中，除了 之外，还可以有更多的对象，例如优化器状态，这也将包含在诱导的TTT层的隐藏状态中。 这种扩展允许TTT层在未来的工作中使用更复杂的优化器，例如Adam \[42\]

### 2.2 实现的更多细节以及骨干架构

#### 2.2.1 的实例化

提出了两种TTT层的变体——TTT-Linear和TTT-MLP，它们仅在的实例化上有所不同

- 对于TTT-Linear， $f_{\operatorname{lin}}(x)=W x$ ，其中W是方阵
- 对于TTT-MLP，有两层，类似于Transformer中的MLP

具体来说，隐藏维度是输入维度的4×，然后是GELU激活\[31\]

且为了在TTT期间获得更好的稳定性，总是包含层归一化(LN)和残差连接。即， $f(x)=x&plus;\operatorname{LN}\left(f_{\mathrm{res}}(x)\right)$ ，其中可以是或

#### 2.2.2 可学习的

TTT初始化 在所有序列之间共享，尽管后续权重 $W_{1}, \ldots, W_{T}$ 对于每个输入序列是不同的。可以将作为外循环的一部分来学习，而不是将其设置为0

由于外循环参数总是用θ而不是W表示，将别名 $\theta_{\text {init }}=W_{0}$ 分配给它。 在实践中， $\theta_{\text {init }}$ 与重建视图 $\theta_{K}, \theta_{Q}, \theta_{V}$ 相比，增加的参数量可以忽略不计，因为它的输入和输出都是低维的。根据经验，观察到学习显著提高了训练的稳定性

#### 2.2.3 可学习的\\eta

先回顾下

> **1.2.2 对于更新规则——使用小批量TTT进行并行化：小批量梯度下降**
> 
> 目前的TTT有个比较明显的问题是，其更新规则——公式2 $W_{t}=W_{t-1}-\eta \nabla l\left(W_{t-1} ; x_{t}\right)$ 不能被并行化，因为在两个地方依赖于
> 
> 1. 一个地方是在减号前
> 2. 一个地方在 $\nabla l$ 内部
> 
> 而由于后者 $\nabla l$ 内部包含了大部分计算，所以我们要重点对后者—— $\nabla l$ 内部，做并行化
> 
> 首先，梯度下降GD有许多变体，GD的一般更新规则可以表示为(记为公式6)
> 
> $$
> W_{t}=W_{t-1}-\eta G_{t}=W_{0}-\eta \sum_{s=1}^{t} G_{s}
> $$
> 
> 其中，是下降方向，这个公式的价值和意义在于，一旦计算出 $G_{t} \text { for } t=1, \ldots, T$ ，便可以基于上述公式的后半部分且通过cumsum获得所有的 $W_{t} \mathrm{~s}$
> 
> 而对于我们所想要的更新规则是在线梯度下降，使用 $G_{t}=\nabla l\left(W_{t-1} ; x_{t}\right)$
> 
> 1. 为了并行化 $G_{t} \text { for } t=1, \ldots, T$ ，可以对所有这些变量相对于进行计算，在这种变体情况下， 其中 $G_{t}=\nabla \ell\left(W_{0} ; x_{t}\right)$ 被称为 **批量梯度下降**

学习率通常是梯度下降中最重要的超参数，因此尝试在外循环中学习内循环学习率 $\eta$ ，如之前介绍过的公式6

$$
W_{t}=W_{t-1}-\eta G_{t}=W_{0}-\eta \sum_{s=1}^{t} G_{s}
$$

为了增加灵活性，可以使 $\eta$ 成为 输入token 的函数(因此在时间上有所不同)

具体来说，设计 $\eta(x)=\eta_{\text {base }} \sigma\left(\theta_{\text {lr }} \cdot x\right)$ ，其中可学习向量 $\theta_{\mathrm{lr}}$ 是一个外循环参数， $\sigma$ 是sigmoid函数，标量 $\eta_{\text {base }}$ 是基础学习率，对于TTT-Linear设为1，对于TTT-MLP设为0.1。当然， $\eta(x)$ 也可以解释为 $\nabla \ell$ 的一个门

#### 2.2.4 骨干架构：默认使用Mamba骨干网络

将任何RNN层集成到更大的架构中的最干净方法是直接替换Transformer中的自注意力机制，在这种情况下称为骨干

然而，现有的RNN如Mamba \[26\] 和Gri n \[18\] 都使用与Transformer不同的骨干。最显著的是，它们的骨干在RNN层之前包含时间卷积，这可能有助于收集跨时间的局部信息

且在实验了Mamba骨干后，发现它也提高了TTT层的困惑度，因此TTT将其纳入TTT的方法中，详见下图

![](https://i-blog.csdnimg.cn/direct/1310fb63bda747d5a3685fd260c0cb15.png)

如上图所示

- 左：一个残差块，Transformer的基本构建块。序列建模块被实例化为两种变体：Transformer骨干和Mamba骨干
- 中：Transformer骨干中的TTT层。 在 O之前的LN来自NormFormer \[*Normformer: Improved transformer pretraining with extra normalization*\]
- 右：受Mamba \[26\]和Griffin\[*Griffin:Mixing gated linear recurrences with local attention for efficient language model*\]启发的骨干中的TTT层。 遵循这两种架构， $\sigma$ 这里是GELU \[31\]  
	为了在不改变嵌入维度的情况下容纳门控的额外参数，可简单地将 $\theta_{K}$ 和 $\theta_{Q}$ 组合成一个投影

实际实现时，由于Transformer和Mamba使用不同的骨干网络，而TTT-Linear和TTT-MLP总是默认使用Mamba骨干网络，除非另有说明

## 第三部分 实验：对TTT效果的评估

接下来，通过与两个基线——Transformer和现代RNN Mamba进行比较来评估TTT-Linear和TTT-MLP

且主要代码库基于EasyLM \[25\]，这是一个用于在JAX中训练和服务LLM的开源项目。另，所有  
实验都可以使用这里《 [https://github.com/test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch "https://github.com/test-time-training/ttt-lm-pytorch") 》提供的公开代码和数据集进行复现

在数据集的选择上，根据Mamba论文 \[26\]，在Pile \[24\] 上进行标准实验，使用2k和8k的上下文长度，这是一个用于训练开源LLM的流行文档数据集 \[9\]

然而，Pile包含的长度超过8k的序列很少 \[19\]。 为了评估长上下文的能力，我们还在1k到 32k 范围  
内以2×递增的上下文长度进行实验，使用一个名为Books3的Pile子集，该子集已被广泛用于训练长上下文的LLM \[49, 3\]

且为了确保评估公平，作者们在可能的情况下严格遵循Mamba论文中的评估协议：

- 对于每个评估设置（例如，数据集、上下文长度和方法），我们实验了四种模型大小：125M、350M、760M和1.3B参数。 对于Mamba，相应的大小是130M、370M、790M和1.4B，因为Mamba不遵循Transformer配置
- 所有模型都使用Mamba论文中描述并在附录C中重现的Chinchilla配方9进行训练。他们的Transformer基线基于Llama架构\[73\]，也遵循Mamba论文中的基线  
	*值得一提的是，与Mamba论文中的唯一不同之处在于分词器。 Mamba论文在各种实验中使用了两种不同的分词器——GPT-2和GPT-NeoX。 为了保持一致性，TTT坚持使用单一分词器，并选择了Llama 2的分词器*
- 没有对混合架构（例如 Grin \[18\]）进行实验，因为基线不是混合架构。 虽然使用自注意力和TTT层的混合架构可能会提高性能，但它们会降低学术评估的清晰度

### 3.1 短上下文：the Pile

#### 3.1.1 Transformer与mamba、TTT-linear、TTT-MLP的相互PK

如下图所示「 *当一个图同时包含Transformer骨干和Mamba骨干时，则分别用(T)和 (M)表示* 」

![](https://i-blog.csdnimg.cn/direct/de4e5537c323491da57cf20ac5a1811c.png)

- 在2k上下文中，TTT-Linear (M)、Mamba和Transformer具有可比的性能，因为这些线条大多重叠。 在大FLOP预算下，TTT-MLP (M)的表现略差。 尽管TTT-MLP在每个模型大小上都比TTT-Linear具有更好的困惑度，但在FLOPs上的额外成本抵消了这一优势
- 在8k上下文中，TTT-Linear (M)和TTT-MLP (M)的表现显著优于Mamba，这与在2k时的观察结果相反。 即使是具有Transformer骨干的TTT-MLP (T)在大约1.3B时表现也略好于Mamba  
	且在本文中观察到的一个稳健现象是，随着上下文长度的增加，TTT层相对于Mamba的优势变得更大
- 在8k上下文中，Transformer在每个模型规模上仍然具有良好的（如果不是最好的）困惑度，但由于FLOPs的成本，其曲线不具竞争力

总之，TTT-Linear在2k上下文中表现与Mamba相当，而在8k上下文中表现更好

#### 3.1.2 骨干网络的影响：具备mamba骨干网络的TTT-Linear可以比肩TTT-MLP

将TTT层从Mamba骨干网络切换到Transformer骨干网络有两个影响

1. 首先，迄今为止，具有Mamba骨干网络的TTT层在评估中表现更好  
	*Switching the TTT layers from Mamba backbone into Transformer backbone  
	has two effects. First, **TTT layers with Mamba backbone** perform better in our evaluations so far.*
2. 其次，使用Mamba骨干网络时，TTT-MLP(M)最多只能与TTT-Linear(M)相当；但使用Transformer骨干网络时，TTT-MLP(T)明显更好  
	*Second, with Mamba backbone, TTT-MLP at best is only comparable to TTT-Linear; but with Transformer backbone, TTT-MLP is clearly better.*

我们假设，当序列建模层具有较不具表现力的隐藏状态时，Mamba骨干中的时间卷积会更有帮  
助(*We hypothesize that the temporal convolutions in the Mamba backbone help more when the sequence modeling layer has a less expressive hidden state.*)

所以虽说线性模型的表现力不如MLP，但线性模型可以从mamba的卷积中受益更多，才使得具备mamba骨干网络的TTT-Linear可以比肩TTT-MLP(*The linear model is less expressive than the MLP, therefore benefits more from the convolutions.We will revisit this hypothesis in the next subsection*)，即 **线性 + mamba ≈ MLP，即mamba的卷积部分可以弥补线性模型的劣势**

### 3.2 长上下文：Books

如下图所示

![](https://i-blog.csdnimg.cn/direct/8756e1fd412e4b2593cdb6e45cda52c8.png)

- 在Books的2k上下文中，Pile 2k的所有观察结果仍然成立，除了Mamba现在表现略优于TTT-Linear(而在Pile 2k中它们的线大致重叠）
- **在32k上下文中，TTT-Linear (M)和TTT-MLP (M)的表现都优于Mamba** ，类似于Pile 8k的观察结果。 即使使用Transformer骨干的TTT-MLP (T)在32k上下文中表现也略好于Mamba
- 在1.3B规模下，TTT-MLP(T)仅略逊于TTT-MLP (M)。 如前所述，由于缺乏清晰的线性拟合，很难推导出经验缩放定律。然而，TTT-MLP (T)的强劲趋势表明，Transformer骨干可能更适合于评估之外的更大模型和更长上下文

## 第四部分 与相关工作的对比总结

### 4.1 TTT-MLP优于TTT-Linear，而TTT-Linear又优于Mamba

Mamba是众多结构化状态空间模型之一 \[27, 21, 57, 18\]。 这些模型中的隐藏状态是一个向量，类似于LSTM

在TTT-Linear或TTT-MLP中，隐藏状态是一个矩阵或两个矩阵，因此更大

如下图所示，我们发现TTT层可以利用其更大的隐藏状态在长上下文中压缩更多信息，其中TTT-MLP优于TTT-Linear，而TTT-Linear又优于Mamba

![](https://i-blog.csdnimg.cn/direct/2da55234d4b442d48d097f7aa6f8d7f0.png)

与TTT-Linear类似，RWKV \[55, 56\]、xLSTM \[5\]和门控线性注意力(GLA)\[79\]也具有矩阵隐藏状  
态，这些状态继承自线性注意力 \[41\]。 现代RNN如GLA使用块状并行来提高硬件效率，因此块内  
的标记可以通过matmul而不是cumsum来处理

然而，块状并行并没有改变模型的表达能力，因为所有时间依赖性仍然等同于 cumsum

相比之下，小批量TTT允许跨小批量的更复杂的时间依赖性。每个隐藏状态 依赖于其小批量内的先前 仍然通过 cumsum，但也通过梯度算子依赖于先前小批量中的

如之前这图所示

> ![](https://i-blog.csdnimg.cn/direct/fc92141b04054ed2802ec7cdbcd9aecc.png)

小批量TTT在表达能力和硬件效率之间实现了权衡，因为较小的批量大小b会以更高的延迟为代价带来更好的困惑度。 这种权衡是TTT的一个独特且重要的特性

如下表所示，中间批量大小b= 16显著优于 b=T完全cumsum

![](https://i-blog.csdnimg.cn/direct/bd5eafc1435c41aeb01ea32c9da52b6b.png)

如上表所示，这里所有模型都有 125M 参数，并按照小节 3.1 中的配方进行训练

1. 最后一行，困惑度为 11.09，是下图中 TTT-Linear 的最终结果
	![](https://i-blog.csdnimg.cn/direct/de4e5537c323491da57cf20ac5a1811c.png)
2. 从小节 2.1 中讨论的等效性开始，可学习的略有损害，但下面的行在没有它的情况下无法稳定训练
3. 最大的改进来自小批量 TTT(从 b= T= 2048 改为 b= 16)  
	第二个来自于实例化内部模型与LN和残差连接  
	如果没有TTT的概念框架，这两种设计都将很难遇到

### 4.2 测试时学习——Learning at Test Time

在机器学习中，测试时学习的概念有着悠久的历史。 这种概念最早的版本之一被称为局部学习（Bottou 和 Vapnik \[10\]）：对于每个测试输入，在做出预测之前，先对其邻居进行训练。 这种程序已被有效地应用于从SVM \[81\]到现代LLM \[29\]的各种模型

测试时学习的另一个早期版本被称为传导学习 \[22\]。 Vladimir Vapnik \[74\]提出的传导原则是“...得到你真正需要的答案，而不是更一般的答案。”

传导学习的实际应用使用测试数据来为SVM的边界添加约束 \[39, 17\]。 然而，传导学习通常需要多个测试实例才能在经验上有效，不像许多测试时训练的实例化，只需要一次一个测试实例(图像、视频或自然语言序列)

在计算机视觉中，测试时学习的理念已经应用于面部检测 \[38\]、物体检测 \[53\]、图像超分辨率 \[65\]和 3D 重建 \[50\] 等应用领域数十年

最近，同样的理念也被应用于自然语言处理领域，在那里它被称为动态评估 \[44, 45\]。 基本方法是直接在测试序列上微调语言模型，这通常以prompt的形式出现

接下来，我们详细讨论两个相关的工作方向：测试时训练和快速权重

#### 4.2.1 测试时训练——Test-Time Training

测试时训练 (TTT) 的核心思想是每个测试实例定义其自身的学习问题，其中该测试实例本身是泛化的目标 \[69\]

具体来说

- 对于每个测试实例x，传统做法是使用预测器预测，该预测器是针对所有训练实例的平均优化的
- TTT首先通过定义一个学习问题，然后在上训练一个模型 (通常以f作为初始化)，并预测

由于测试实例没有标签，学习问题只能通过自监督任务来制定。 先前的工作表明，使用重建的TTT显著提高了性能，尤其是在异常值上\[23\]

当在以流的形式到达的视频帧上进行测试并且TTT是自回归的时，改进变得更加显著\[76\]，因为是在过去的帧 $x_{1}, \ldots, x_{t}$ 上训练的。 自回归连接使得\[76\]与TTT的论文最相关

从概念上讲，TTT与先前工作的最大区别在于TTT的重建任务是在外循环中学习的，而不是通过人为先验手工制作的。 TTT的后续工作探索了诸如机器人操作\[28\]和运动\[68\]等应用，这些应用通常需要对自监督任务进行不同的设计

#### 4.2.2 快速权重

快速权重的一般思想是仅在最相关的数据上更新“快速”模型的参数，而不是像传统做法那样在所有数据上更新“慢速”模型 \[71\]

这个想法自20世纪80年代以来就存在了 \[32\]。 最相关的数据可以是测试实例本身，因此TTT可以被视为快速权重的一种特殊情况

先前关于快速权重的工作通常避免形成一个明确的学习问题来优化数据上的某些目标。 例如，Hebbian学习和Hopfield网络的更新规则 \[35\] 只是简单地将 xxT（或其某些变体） \[4\] 添加到每个输入 x的快速权重中。 相比之下，TTT接受了明确制定学习问题的理念，其中测试实例是泛化的目标。 TTT的更新规则也是一个明确的优化步骤

快速权重程序员(FWPs)的想法是用一个“慢”模型来更新快速权重 \[62\]

1. TTT的内循环权重 W可以被视为“快”的，而外循环权重 θ则被视为“慢”的  
	因此，包含TTT层的网络可以被视为FWPs的一个特例 \[43\]，类似于TTT可以被视为快速权重的一个特例。 上述使用Hebbian更新规则的FWP等价于线性注意力 \[60\]，因此也等价于使用批量梯度下降的朴素TTT-Linear
2. FWPs的定义非常广泛。 事实上，所有具有某种门控机制的网络，例如带有SwiGLU块的Transformer \[63\]，也可以被视为FWPs的一个特例16。 最近的工作一直在尝试将FWPs用于语言建模：Irie等人 \[37\] 设计了权重作为“慢”网络输出的“快”网络。Clark等人 \[16\] 给Transformer添加了一个快速权重的最终层，其初始化被训练为慢权重
3. TTT相对于现有FWP工作的贡献在于，提出了一个明确的更新学习问题，这使得能够借用诸如小批量和LN等学习工具

### 4.3 Learning to Learn

几十年来，研究人员一直在争论，学习如何学习，也称为元学习或双层优化，应该是智能的关键组成部分 \[61, 6, 70, 47\]。 在之前的工作中，例如 \[2\]、\[20\] 和 \[52\]，内循环每次从整个数据集而不是序列中学习，因此外循环需要一组数据集或任务。 简而言之，外循环是“比常规训练高一级”。由于很难收集数百万个数据集，这个外循环很难扩展

相比之下，对于TTT，每个序列本身就是一个数据集，并定义了自己的泛化问题。内循环比常规训练“低一级”，因此TTT的外循环只是监督学习经典问题的另一种解决方案，而不是像跨数据集泛化那样的新问题设置

如下表所示，TTT的外循环与常规训练“处于同一级别”，这使得TTT的外循环更容易扩展

![](https://i-blog.csdnimg.cn/direct/582c4781fac94d4c93c4eca004da7e31.png)

总之，如上表所示，TTT的论文将监督学习重新表述为学习如何学习，具有两个嵌套循环

外循环的高亮行与常规训练中的相同，外循环的参数成为内循环的超参数。 直观地说，内循环，即TTT，是常规训练的“下一级”

最后，如TTT论文所说

1. 为什么我们要研究TTT？先前的工作通常尝试用机器学习来模拟人类学习，其中训练是在一个洗牌后的i.i.d.实例数据集上进行的，推理是在一个单独的测试集上进行的。然而，人类并不是自然地通过i.i.d.实例学习，也没有训练-测试分割
2. 人类学习与TTT(TTT的内部循环)有更有前途的联系，其数据是一个潜在的非常长的序列，具有强烈的时间依赖性，任何数据片段都可以用于训练和测试。 这就是研究TTT的原因