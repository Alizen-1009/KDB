---
title: "【LLM2】Standford TTT模型(Learn at Test Time)"
source: "https://zhuanlan.zhihu.com/p/6827298295"
author:
  - "[[举个栗子​]]"
published:
created: 2026-08-26
description: "论文地址： https://arxiv.org/pdf/2407.04620Github地址： https://github.com/test-time-training/ttt-lm-pytorch摘要self-attention在长文本问题中表现良好，但计算复杂度为二次方。RNN具有线性复杂度，但它们…"
tags:
  - "clippings"
---
10 人赞同了该文章

目录

收起

摘要

动机

方法

TTT 层的定义

包含TTT层的网络训练

TTT的效率优化

实验结果

结论

论文地址： [arxiv.org/pdf/2407.0462](https://arxiv.org/pdf/2407.04620)

Github地址： [github.com/test-time-tr](https://github.com/test-time-training/ttt-lm-pytorch)

## 摘要

[self-attention](https://zhida.zhihu.com/search?content_id=250387830&content_type=Article&match_order=1&q=self-attention&zhida_source=entity) 在长文本问题中表现良好，但计算复杂度为二次方。RNN具有线性复杂度，但它们在长文本环境中的表现受到隐状态表达能力的限制。为了弥补二者各自的问题， **作者提出了一类具有线性复杂度和富有表达力的隐状态的序列建模层。关键思想是使隐状态本身成为一个机器学习模型，更新规则成为自监督学习的一个步骤。由于隐状态即使在测试序列上也会通过训练进行更新，因此所提出的层被称为Test-Time Training Layers（TTT）** 。作者考虑了两种具体实现：TTT-Linear和 [TTT-MLP](https://zhida.zhihu.com/search?content_id=250387830&content_type=Article&match_order=1&q=TTT-MLP&zhida_source=entity) ，它们的隐状态分别是一个线性模型和一个两层的MLP。作者在125M到1.3B参数的规模上进行了评估，并与Transformer和 [Mamba](https://zhida.zhihu.com/search?content_id=250387830&content_type=Article&match_order=1&q=Mamba&zhida_source=entity) 进行了比较。TTT-Linear和TTT-MLP均达到或超过了基线。与Transformer类似，它们可以通过依赖更多的token来持续降低困惑度（Perplexity），而Mamba在大于16k上下文后无法做到。在初步的系统优化后，TTT-Linear在8k上下文时已经比Transformer更快，并且与Mamba在训练时间上相匹配。作者认为TTT-MLP在内存I/O方面虽然仍面临挑战，但在长文本环境中显示出更大的潜力，为未来研究指明了一个有希望的方向。

## 动机

所有的序列建模的目的都可以定义为需要将历史上下文存储到隐藏状态中。而现在两种主流结构，即RNN（LSTM/Mamba/RWKV）和Transformer都有各自的优缺点：

- RNN（LSTM/Mamba/RWKV）：将 context 压缩为随时间变化的固定大小的隐状态。由于该隐状态是固定大小的，因此其计算量/存储空间友好。但比较尴尬的是，这种优势应该在体现在长下文中，但对于长上下文，固定大小的隐状态又表达能力有限。
- Transformer：在自注意力机制中，将context压缩为KV缓存中作为隐状态，其是一个随t线性增长的列表，表达能力更强，但显然其计算时间会线性增长。

因此，为了在长上下文中保持高效和表达力，我们需要更好的压缩知识的方法。具体来说，我们需要将数千或可能数百万个Token压缩成一个隐藏状态，以有效地捕获它们的底层结构和关系。

## 方法

### TTT 层的定义

大语言模型本身就是压缩知识的优秀例子。通过 [next-token prediction](https://zhida.zhihu.com/search?content_id=250387830&content_type=Article&match_order=1&q=next-token+prediction&zhida_source=entity) 的自监督任务进行训练，互联网上海量的知识被压缩存储在LLM的权重中。通过查询 LLM，我们可以从其权重中提取出这些知识。更重要的是，LLM 通常表现出对现有知识之间的语义联系的深刻理解，以实现新的推理。

受此启发，作者提出使用自监督学习，将上下文 $x_{1} , . . . , x_{t}$ 压缩为隐状态 $s_{t}$ ，将上下文视作无标签数据集，隐状态视作模型。具体来说，用一个模型 $f$ 的权重 $W_{t}$ 来表示隐状态 $s_{t}$ ，这个模型 $f$ 可以是线性模型、小型神经网络等等。其 **输出规则** 为

$z_{t} = f \left(x_{t} ; W_{t}\right) (\text{1})$ 直观地说，输出 $z_{t}$ 只是由模型 $f$ 使用更新后的权重 $W_{t}$ 进行的对 $x_{t}$ 的预测。 **更新规则** 是对某种自监督损失 $l$ 进行的梯度下降

$W_{t} = W_{t - 1} - \eta \nabla l \left(W_{t - 1} ; x_{t}\right) (\text{2})$ $W$ 会记住使得损失 $l$ 产生大梯度的输入，此时输入让 $W$ 学到了更多的东西。

文中将 $l$ 选择为一种重建损失，目的是重建 $x_{t}$ 本身。将 $x_{t}$ 破坏成一个损坏的 $\overset{\sim}{x}_{t}$ ，然后优化

$l \left(W ; x_{t}\right) =∥ f \left(\overset{\sim}{x}_{t} ; W\right) - x_{t} \parallel^{2} (\text{3})$ 与去噪自编码器类似， $f$ 需要发现 $x_{t}$ 维度之间的相关性，以便从部分信息 $\overset{\sim}{x}_{t}$ 中重建它。

与其他 RNN 层和self-attention一样，这种将输入序列 $x_{1} , . . . , x_{T}$ 映射到输出序列 $z_{1} , . . . , z_{T}$ 的方法，可以使用上面的隐藏状态、更新规则和输出规则融入到序列建模层的前向传递中。即使在测试时，这种网络层仍然可以为每个输入序列训练不同的权重序列 $W_{1} , . . . W_{T}$ 。因此这种网络层被命名为Test-Time Training (TTT) layer。

### 包含TTT层的网络训练

**TTT层可以在任何更大的网络架构中，替换对应的序列建模层。** 将训练更大的网络称为 **外循环** ，将训练每个TTT层内的 $W$ 称为 **内循环** 。TTT最重要的部分是选择自我监督任务，因为它决定了 $W$ 将从测试序列中学到的特征类型。针对语言建模任务，作者直接将自监督损失 $l$ 选择为next-token prediction任务。

具体来说，将自监督任务作为 **外循环** 的一部分，在式（3）中，添加一些可学习的外循环参数。首先通过设计低秩投影 $\overset{\sim}{x}_{t} = \theta_{K} x_{t}$ ， $\theta_{K} x_{t}$ 被称为训练视图。由于并不是 $x_{t}$ 中的所有信息都值得记住，因此将重建标签也设计为一个低秩投影 $\theta_{V} x_{t}$ ，而不是直接使用 $x_{t}$ ， $\theta_{V} x_{t}$ 被称为标签视图。因此，式（3）被更新为

$$
l \left(W ; x_{t}\right) =∥ f \left(\theta_{K} x_{t} ; W\right) - \theta_{V} x_{t} \parallel^{2} (\text{4})
$$

虽然在上式中， $W$ 和各种 $\theta$ 参数一起出现，但它们本质是不同的，并且会在不同的过程中被更新。在 **内循环** 中，只有 $W$ 被优化，因此它是 $l$ 的参数，而 $\theta$ 是这个内循环中的“超参数”。在 **外循环** 中， $\theta_{K}$ 、 $\theta_{V}$ 、 $\theta_{Q}$ 与 $\theta_{r e s t}$ 一起优化，优化目标可以依任务而定， $W$ 只是一个隐藏状态，而不是一个参数。下图的代码说明了两种循环的区别，其中 $\theta_{K}$ 和 $\theta_{V}$ 作为 TTT 层的参数，类似于自注意力机制中的 Key 和 Value 参数。

![](https://pic2.zhimg.com/v2-1a8c70eb3301387e01984b83447d6b7f_1440w.jpg)

最后，由于训练视图 $\theta_{K} x_{t}$ 的维度比 $x_{t}$ 小，需要重新建立输出规则。最简单的解决方案是创建一个测试视图 $\theta_{Q} x_{t}$ ，并将输出规则更改为：

$z_{t} = f \left(\theta_{Q} x_{t} ; W_{t}\right) (\text{5})$ 这样做的一个好处是，训练视图和标签视图压缩了 $x_{t}$ 中的信息到 $W_{t}$ 并随时间向前传播。测试视图指定了可能不同的信息，这些信息映射到当前输出 $z_{t}$ 并通过网络层向前传播，因此为自监督任务增加了更大的灵活性。 $\theta_{K}$ 、 $\theta_{V}$ 、 $\theta_{Q}$ 的所有可能选项的集合会构成一系列不同的多视图重建任务，外循环则选择其中一个任务。为了简化，作者将所有视图设计为线性投影。

### TTT的效率优化

然而，TTT有着和RNN一样的问题，即梯度的计算有前文依赖性，导致无法并行进行参数的更新，那么怎么解决这个问题呢？

1. 小批量优化

式（2）的更新规则无法实现并行化，主要因为其梯度的计算依赖于 $W_{t - 1}$ 。如果使用在线梯度下降法的话，式（2）可以写为 $W_{t} = W_{t - 1} - \eta G_{t} = W_{0} - \eta \sum_{s = 1}^{t} G_{s} (\text{6})$ 其中， $G_{t} = \nabla l \left(W_{t - 1} ; x_{t}\right)$ 。一旦计算了对于 $t = 1 , . . . , T$ 下的所有 $G_{t}$ ，可以上述公式的第二部分累加和得到所有的 $W_{t}$ 。

那么为了对所有 $G_{t}$ 实现并行化，可以将它们全部对 $W_{0}$ 进行计算，那么 $G_{t} = \nabla l \left(W_{0} x_{t}\right)$ ，这种方法称为批量梯度下降，因为 $\sum_{s = 1}^{t} \nabla l \left(W_{0} ; x_{t}\right)$ 与 $x_{1} , . . . , x_{t}$ 作为一个批次相对于 $W_{0}$ 的梯度是相同的。但是这种方法不适合于语言建模这种序列任务。因此，作者提出小批量梯度下降法，如下图所示，节点代表变量，边代表计算。由于 $G_{1} , . . . , G_{b}$ 没有相互连接，它们之间不存在顺序依赖关系，因此可以实现并行计算。

![](https://pica.zhimg.com/v2-09bbc6d1d7be4a86f350eee726c34924_1440w.jpg)

使用 $b$ 表示小批量大小，那么 $G_{t} = \nabla \left(W_{t^{'}} ; x_{t}\right)$ ，其中 $t^{'} = t - m o d \left(t , b\right)$ 为前一个小批量的最后一个时间步，这样实现近似并行计算 $b$ 个梯度。不过这里 $b$ 的选取过大，模型的困惑度（perplexity）会变大，选取的小，速度变慢，但困惑度会变小，这个平衡就比较玄学了。

2\. 对偶形式

作者提出可以使用一种矩阵乘法的技巧，使用一种对偶形式计算 $W_{b}$ ，进一步节省了内存和I/O开销。

## 实验结果

作者使用Transformer和Mamba作为baseline，评估了TTT-Linear和TTT-MLP的性能。测试集为 [Pile](https://zhida.zhihu.com/search?content_id=250387830&content_type=Article&match_order=1&q=Pile&zhida_source=entity) ，分别进行了2k和8k上下文长度的实验。模型参数为125M，250M，760M和1.3B。TTT-Linear和TTT-MLP始终使用Mamba骨干架构，除非另有说明。当一幅图同时包含Transformer骨干和Mamba骨干时，分别用(T)和(M)表示。

**结论1：** 在2k上下文长度下，TTT-Linear（M）、Mamba和Transformer表现相当，因为它们的曲线大部分重叠。TTT-MLP（M）在大的FLOP预算下表现稍逊。尽管TTT-MLP在每个模型尺寸下的困惑度更佳，但额外的FLOP成本抵消了这一优势。

**结论2：** 在8k上下文长度下，TTT-Linear（M）和TTT-MLP（M）都显著优于Mamba，与2k的观察相比截然不同。甚至使用Transformer骨干的TTT-MLP（T）在大约13亿时表现略优于Mamba。

![](https://pica.zhimg.com/v2-a9477cf5b84d364bf4f034882c9575e0_1440w.jpg)

**结论3：** 在32k上下文中，TTT-Linear（M）和TTT-MLP（M）的表现都优于Mamba，这与Pile 8k的观察相似。即使是使用Transformer骨干的TTT-MLP（T），在32k上下文中也略优于Mamba。

![](https://pic1.zhimg.com/v2-9782cebaba7cc2d1eb1215cf5a10b970_1440w.jpg)

结论4：对于前向过程，随着文本变长，标准Transformer处理每个token的时间变长，TTT的方法几乎不变。对于生成(解码)过程，TTT-Linear和Mamba具有几乎相同的延迟，明显小于Transformer和TTT-MLP。

![](https://picx.zhimg.com/v2-6df7f24ad8f130952788b236a9b7c92b_1440w.jpg)

## 结论

之前的工作经常尝试使用机器学习对人类学习进行建模，其中训练是在带有分布内实例的随机数据集上进行的，而推理是在单独的测试集上进行的。然而，人类不会自然地使用分布内实例进行学习，也不会进行训练和测试集的拆分。作者相信人类学习与TTT的内循环学习过程是有相同之处的，TTT认为数据应该是一个非常长的序列，具有很强的时间依赖性，在这个序列中，任何一个时间点的数据都可以用于训练和测试。因为对于人类而言，在这个序列里，每一个数据点都可以既是学习的材料，也是检验我们学习成果的“考题”。不过作者也提出TTT还有许多需要改进的地方，包括外循环参数的重建任务选择、 $f$ 的实例化选择、更好的系统优化方法。

BTW，我个人认为觉得本文的实验也有些不充足，模型性能方面只有困惑度一个指标肯定是不够的，并且如果想要提出Transformer的替代结构的话，1.3B的模型尺寸也不够，此外文章的方法也利用了很多trick来对各种超参数进初始化和学习，感觉是不是用起来会很难调。不过本文的思路还是非常好的，而且据说也有很多人正在魔改TTT用在其他领域了，还是非常期待TTT的后续工作的！

发布于 2024-11-21 17:54・安徽

赞同 10