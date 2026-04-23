---
title: "Flash Attention 详细解释推演与Pytorch代码实现"
source: "https://zhuanlan.zhihu.com/p/2023872534563619385"
author:
  - "[[柠檬沙棘1996​软件开发行业 工程师]]"
published:
created: 2026-04-14
description: "概要 FlashAttention 算法，作为加速 Transformer 模型中自注意力（Self-Attention）计算并降低显存占用的算法，由斯坦福大学的 Tri Dao 等人在 2022 年提出，目前已经成为大语言模型（如 GPT-4, Llama, Falcon 等…"
tags:
  - "clippings"
---
目录

收起

概要

一. 核心痛点：标准 Attention 的内存瓶颈

二. FlashAttention 的核心思想：IO 感知

三. 一维运算的特例简化

0\. 注意力公式分解

1\. 第一步：传统 Softmax 为什么在硬件上“很慢”？

2\. 第二步：Online Softmax 的核心魔法（分块计算）

1\. 处理 Block 1

2\. 处理 Block 2

3\. 见证奇迹的时刻：如何合并？

3.第三步：在 FlashAttention 中结合 Value (V) 矩阵

1\. 处理 Block 1

2\. 处理 Block 2 （结合 FA 的最终推导）

4\. FlashAttention 到底讲透了什么，省了什么？

5\. 总结与通俗比喻

四. 由一维向量到二维矩阵的推广

1\. 明确目标：Softmax 在矩阵中是怎么算的？

2\. 矩阵分块：FlashAttention 的双重循环

步骤一： 乘以 的第 1 块（记作 ）

步骤二： 乘以 的第 2 块（记作 ）

步骤三：矩阵级别的合并（Online Softmax）

五. 分子分母分开计算

1\. 传统思路的直觉陷阱

2\. 为什么现在的 FA（尤其是 FA-2）不让 d 参与 的局部计算？

3\. 最优解：分子分母“兵分两路”，各算各的

在处理 Block 1 时

在处理 Block 2 时

4\. 那么， 什么时候才登场？

六. 矩阵分块与内外双层循环的对应关系

1\. 现代标准：FlashAttention-2 的双层循环（外 Q 内 KV）

2\. 历史的弯路：FlashAttention-1 为什么慢一点？（外 KV 内 Q）

3\. 还有一个隐藏的红利：Causal Mask（因果掩码）

4\. 示例代码

5\. 结果分析

1\. 最大计算误差：0.00000000e+00

2\. 时延骤降：从 679.73 ms 降到 404.28 ms

3\. 深度思考：为什么提速比“只有” 1.68x？

总结

参考文献

## 概要

**FlashAttention** 算法，作为加速 [Transformer](https://zhida.zhihu.com/search?content_id=272569857&content_type=Article&match_order=1&q=Transformer&zhida_source=entity) 模型中自注意力（ [Self-Attention](https://zhida.zhihu.com/search?content_id=272569857&content_type=Article&match_order=1&q=Self-Attention&zhida_source=entity) ）计算并降低显存占用的算法，由斯坦福大学的 Tri Dao 等人在 2022 年提出，目前已经成为大语言模型（如 [GPT-4](https://zhida.zhihu.com/search?content_id=272569857&content_type=Article&match_order=1&q=GPT-4&zhida_source=entity), [Llama](https://zhida.zhihu.com/search?content_id=272569857&content_type=Article&match_order=1&q=Llama&zhida_source=entity), [Falcon](https://zhida.zhihu.com/search?content_id=272569857&content_type=Article&match_order=1&q=Falcon&zhida_source=entity) 等）训练和推理的标准配置。经历诸多次版本迭代，现在已经成为了大语言模型大厦的主要根基。要理解 FlashAttention，我们需要先了解标准注意力机制的瓶颈，然后看 FlashAttention 是如何通过 **IO 感知（IO-Awareness）** 来解决这个问题的。

## 一. 核心痛点：标准 Attention 的内存瓶颈

在标准的 Transformer 注意力机制中，计算公式为：

$(*) \text{Attention} \left(\right. Q , K , V \left.\right) = \text{Softmax} \left(\right. \frac{Q K^{T}}{\sqrt{d}} \left.\right) V$

假设序列长度为 $N$ ，主要问题在于中间产生的那个 **Attention Matrix（注意力矩阵）** ，其大小为 $N \times N$ 。

- **显存占用爆炸（O(** $N^{2}$ **))：** 当序列变长（比如从 4k 增加到 32k）， $N \times N$ 的矩阵会变得巨大，导致显存溢出（OOM）。
- **内存带宽瓶颈（Memory Wall）：** GPU 的计算速度（FLOPs）非常快，但显存（HBM）的读写速度相对较慢。
- 标准做法需要把庞大的 $N \times N$ 矩阵在 **HBM（高带宽内存/显存）** 和 **SRAM（GPU芯片上的快速缓存）** 之间反复搬运。
	- **读写（IO）耗时远超计算耗时** 。这就好比厨师（计算核心）切菜很快，但每次切完一根葱都要跑到隔壁房间（HBM）去拿下一根，时间都浪费在路上了。

## 二. FlashAttention 的核心思想：IO 感知

FlashAttention 的核心目标是： **减少对 HBM（慢速显存）的读写次数，尽可能在 SRAM（快速缓存）中完成计算。** 它主要利用了两个关键技术： **分块计算（Tiling）** 和 **Sofmax重计算（Online Softmax Recomputation）** 。

![](https://picx.zhimg.com/v2-9c3a9b1c700823fd17b3397d5dda4423_1440w.jpg)

FA with IO-Awareness

由于 FlashAttention 算法非常复杂，下面的内容将逐步拆解，以Online Softmax，也是FA的重大核心点为基础，按照从特例到一般逐步推导，详细分析算法的实现。

## 三. 一维运算的特例简化

### 0\. 注意力公式分解

一般来说，注意力公式可以分解成以下 **三步** （为计算方便，暂时省略除以 $\sqrt{d}$ 的纯数学计算逻辑）：

$(\text{1}) S = Q K^{T} \\ (\text{2}) P = \text{softmax} \left(\right. S \left.\right) \\ (\text{3}) O = P V$

注：上述公式中的中间变量符号一般为通用表示，S：Attention Score，P：Probablity，O：Output.

理解 FlashAttention (FA) 中的 **Online Softmax** ，核心在于明白 **“我们为什么要这么做”** 以及 **“数学上使用了什么巧妙的等价代换”** 。我们分三步来拆解：首先看传统的 Softmax 有什么致命缺陷，然后推导 Online Softmax 的核心魔法（ **缩放因子** ），最后看它在 FA 中是如何和 $V$ 矩阵结合的。

### 1\. 第一步：传统 Softmax 为什么在硬件上“很慢”？

我们先以一维数组作为例子，假设我们有一组注意力分数（即上述公式1中的结果 $S$ ），为简化计划先姑且以一维向量为例：

$(\text{1}.\text{1}) x = \left[\right. x_{1} , x_{2} , \ldots , x_{N} \left]\right.$

标准的 Softmax 公式是：

$(\text{1}.\text{2}) y_{i} = \frac{e^{x_{i}}}{\underset{j}{\sum} e^{x_{j}}}$

**问题 1：数值溢出（指数爆炸）** 如果 $x_{i}$ 很大， $e^{x_{i}}$ 会直接溢出（变成无穷大）。为了解决这个问题，实际工程中一直使用的是 **Safe Softmax** ，即减去最大值：

找出最大值： $(\text{1}.\text{3}) m = max \left(\right. x \left.\right)$

计算指数并求和： $(\text{1}.\text{4}) d = \underset{j}{\sum} e^{x_{j} - m}$

计算最终结果： $(\text{1}.\text{5}) y_{i} = \frac{e^{x_{i} - m}}{d}$

**问题 2：内存读写灾难（Safe Softmax 的代价）** 上面的 3 步，对一个较大的形状 tensor 的计算而言，在 GPU 中意味着需要进行三步 **读取+计算** 的过程。

- **第 1 遍读取+计算** ：把 $x$ 从慢速内存（HBM）搬到快速内存（SRAM），找出最大值 $m$ ，存回 HBM。
- **第 2 遍读取+计算** ：把 $x$ 和 $m$ 从 HBM 搬到 SRAM，计算 $e^{x_{i} - m}$ 和它们的和 $d$ ，把中间结果存回 HBM。
- **第 3 遍读取+计算** ：把中间结果和 $d$ 搬到 SRAM，做除法得到 $y_{i}$ ，存回 HBM。

**GPU 的计算速度极快，但 HBM 的读写极慢。** 依据 [RoofLine 理论](https://zhida.zhihu.com/search?content_id=272569857&content_type=Article&match_order=1&q=RoofLine+%E7%90%86%E8%AE%BA&zhida_source=entity) 模型，这个传统 Softmax 计算方式因为要依赖全局最大值 $m$ ，必须把整个数组遍历三遍，大量时间浪费在搬运数据上，出现 memory-bound。

### 2\. 第二步：Online Softmax 的核心魔法（分块计算）

**Online Softmax 的目标是：只遍历一次数据，边读数据、边算最大值、边更新结果！** 假设内存有限，为了简化运算，我们设定数组元素个数为4个，即

$(\text{2}.\text{1}) x = \left[\right. x_{1} , x_{2} , x_{3} , x_{4} \left]\right.$

并且把数据分成两块处理： **Block 1** 和 **Block 2** 。

### 1\. 处理 Block 1

假设 Block 1 的数据是 $\left[\right. x_{1} , x_{2} \left]\right.$ 。

找到局部的最大值： $(\text{2}.\text{2}) m_{1} = max \left(\right. x_{1} , x_{2} \left.\right)$

计算局部的分母（指数和）： $(\text{2}.\text{3}) d_{1} = e^{x_{1} - m_{1}} + e^{x_{2} - m_{1}}$

### 2\. 处理 Block 2

假设 Block 2 的数据是 $\left[\right. x_{3} , x_{4} \left]\right.$ 。

找到局部的最大值： $(\text{2}.\text{4}) m_{2} = max \left(\right. x_{3} , x_{4} \left.\right)$

计算局部的分母（指数和）： $(\text{2}.\text{5}) d_{2} = e^{x_{3} - m_{2}} + e^{x_{4} - m_{2}}$

### 3\. 见证奇迹的时刻：如何合并？

现在我们要得到这四个数字的全局结果。

**全局最大值** 很简单： $(\text{2}.\text{6}) m_{n e w} = max \left(\right. m_{1} , m_{2} \left.\right)$

- **全局的分母** $d_{n e w}$ 怎么算？我们能直接把 $d_{1} + d_{2}$ 吗？ **绝对不行！** 因为 $d_{1}$ 是以 $m_{1}$ 为基准减出来的， $d_{2}$ 是以 $m_{2}$ 为基准减出来的，它们的“参考系”不同。

**魔法来了：改变参考系（指数补偿）** 我们想把 $d_{1}$ 的参考系从 $m_{1}$ 变成 $m_{n e w}$ 。 数学上：

$(\text{2}.\text{7}) e^{x_{1} - m_{n e w}} = e^{x_{1} - m_{1} + m_{1} - m_{n e w}} = e^{x_{1} - m_{1}} \times e^{m_{1} - m_{n e w}}$

所以，我们只需要给旧的分母乘上一个 **“补偿系数”** $e^{m_{1} - m_{n e w}}$ 即可！

修正后的  
$(\text{2}.\text{8}) d_{1}^{'} = d_{1} \times e^{m_{1} - m_{n e w}} ， d_{2}^{'} = d_{2} \times e^{m_{2} - m_{n e w}}$

- **全局分母** ：  
	$(\text{2}.\text{9}) d_{n e w} = d_{1}^{'} + d_{2}^{'}$  
	  
	这就是 Online Softmax 的核心：你不需要一开始就知道全局最大值。你可以先用局部最大值算着，等后面发现了更大的全局最大值，只需要把之前算好的历史结果乘上一个 $e^{旧 最 大 值 - 新 最 大 值}$ 修正系数 进行打折修正就可以了！
- 对应的 Softmax 函数输出值：  
	$(\text{2}.\text{10}) s o f t m a x \left(\right. x_{i} \left.\right) = p_{i} = \frac{e^{x_{i} - m_{n e w}}}{d_{n e w}}$  
	注意了，这里的 Softmax 计算，在 FA 计算中不需要进行，FA的最终目的不是算 Softmax 概率，而是用 Softmax 的结果 $P$ 去对 $V$ （Value矩阵）做加权求和，也就是算输出 $O$ 。至于接下来的过程继续看第三步。

### 3.第三步：在 FlashAttention 中结合 Value (V) 矩阵

假设 $V$ 也分成两块 $v_{1} , v_{2}$ 和 $v_{3} , v_{4}$ 。我们定义 $O$ 是加权求和的结果。

### 1\. 处理 Block 1

当处理 Block 1 的时候，此时我们只能看到两个元素，局部的输出结果是 $O_{1}$ ：

$(\text{3}.\text{1}) O_{1} = \frac{e^{x_{1} - m_{1}} v_{1} + e^{x_{2} - m_{1}} v_{2}}{d_{1}}$

注意，这里 $O_{1}$ 已经是一个完整向量了。

### 2\. 处理 Block 2 （结合 FA 的最终推导）

Block2 进来之后，我们现在有了新的全局最大值 $m_{n e w}$ ，以及新的全局分母 $d_{n e w}$ 。

先不考虑合并两个 Block，我们回归到 **最朴素的计算方式** ，全局正确的输出 $O_{n e w}$ 应该是把四个元素的加权全部统一在 $m_{n e w}$ 的标准下：

$(\text{3}.\text{2}) O_{n e w} = \frac{\left(\right. e^{x_{1} - m_{n e w}} v_{1} + e^{x_{2} - m_{n e w}} v_{2} \left.\right) + \left(\right. e^{x_{3} - m_{n e w}} v_{3} + e^{x_{4} - m_{n e w}} v_{4} \left.\right)}{d_{n e w}}$

这里的所有的 new 下标的全是全局的含义。那么重点又来了，我们怎么利用旧的 $O_{1}$ 算出 $O_{n e w}$ ，而不重新计算前面的东西？也就是找寻全局的 $O_{n e w}$ 与分段 $O_{1}$ 的关系。

继续推导：

$(\text{3}.\text{3}) O_{n e w} = \frac{\left(\right. e^{x_{1} - m_{n e w}} v_{1} + e^{x_{2} - m_{n e w}} v_{2} \left.\right) + \left(\right. e^{x_{3} - m_{n e w}} v_{3} + e^{x_{4} - m_{n e w}} v_{4} \left.\right)}{d_{n e w}} \\ (\text{3}.\text{4}) = \frac{e^{m_{1} - m_{n e w}} \cdot \left(\right. e^{x_{1} - m_{1}} v_{1} + e^{x_{2} - m_{1}} v_{2} \left.\right) + e^{m_{2} - m_{n e w}} \cdot \left(\right. e^{x_{3} - m_{2}} v_{3} + e^{x_{4} - m_{2}} v_{4} \left.\right)}{d_{n e w}} \\ (\text{3}.\text{5}) = \frac{e^{m_{1} - m_{n e w}} \times d_{1} \times O_{1} + e^{m_{2} - m_{n e w}} \times d_{2} \times O_{2}}{d_{n e w}}$

观察式子 $\left(\right. 3.3 \left.\right)$ 中，可以对于分子中的第一部分（属于 Block 1 的部分）：

$(\text{3}.\text{6}) e^{x_{1} - m_{n e w}} v_{1} + e^{x_{2} - m_{n e w}} v_{2}$

提取出缩放因子 $e^{m_{1} - m_{n e w}}$ ，它等于：

$(\text{3}.\text{7}) e^{m_{1} - m_{n e w}} \times \left(\right. e^{x_{1} - m_{1}} v_{1} + e^{x_{2} - m_{1}} v_{2} \left.\right)$

因此得到 式子 $\left(\right. 3.4 \left.\right)$ ，而括号里的这一长串，不就是旧分子 $d_{1} \times O_{1}$ 吗？所以，Block 1 修正后的分子为：

$(\text{3}.\text{8}) e^{m_{1} - m_{n e w}} \times d_{1} \times O_{1}$

即式子 $\left(\right. 3.5 \left.\right)$ 。同理，对与Block 2 新加进来的分子：

$(\text{3}.\text{9}) e^{m_{2} - m_{n e w}} \times \left(\right. e^{x_{3} - m_{2}} v_{3} + e^{x_{4} - m_{2}} v_{4} \left.\right)$

可以同样应用这一变化，最终， **FlashAttention 最核心的更新公式诞生了：**

$(\text{3}.\text{10}) O_{n e w} = \frac{旧分子修正 + 新分子}{新分母}$

即：

$(\text{3}.\text{11}) O_{n e w} = \frac{d_{1} \cdot e^{m_{1} - m_{n e w}} \cdot O_{1} + d_{2} \cdot e^{m_{2} - m_{n e w}} \cdot \left(\right. \text{Block 2 }的局部注意力结果 \left.\right)}{d_{n e w}}$

观察上式，已经没有了Softmax 概率值，已经完全表述为了全局 $O_{n e w}$ 和局部 $O_{i}$ 的函数关系。

### 4\. FlashAttention 到底讲透了什么，省了什么？

现在我们把整个过程连起来看。

在传统的 Attention 中： 算完 $S = Q K^{T}$ 后，需要 **写回显存** 。做 Softmax 时，需要来回读写 $S$ 、最大值、概率矩阵 $P$ 。最后算 $P \times V$ 时，又要读取 $P$ 和 $V$ 。这就导致了极其恐怖的显存读写带宽消耗，且复杂度是 $O \left(\right. N^{2} \left.\right)$ 。

而在 FlashAttention 中： GPU 将序列切分成一个个 Block，装进高速缓存（SRAM）中。

1. 加载一小块 $Q$ 留在 SRAM 里。
2. 循环加载 $K$ 和 $V$ 的 Block（流式输入）。
3. 算完当前 Block 的 $Q K^{T}$ ，立刻在 SRAM 里算出局部的最大值 $m$ 、局部分母和 $d$ 和局部注意力结果。
4. **利用上面的“缩放因子”公式，直接在 SRAM 里修正之前的** $O$ **。**
5. 当前的 $K$ 和 $V$ 块用完了？直接丢弃！不需要写回显存！
6. 继续加载下一块 $K$ 和 $V$ ，直到循环结束。

由于我们在流式计算的过程中，始终通过 $e^{m_{o l d} - m_{n e w}}$ 去 **动态纠正** 过去的误差，最终循环结束时，SRAM 里的 $O$ 就是完全等价于全局算出来的精准 $O$ 。

### 5\. 总结与通俗比喻

假设我们要统计全国各省的“相对富裕指数加权平均”。因为数据太多，只能一个省一个省地算。

- 传统方法（Safe Softmax）：先跑遍全国，找出全国首富（最大值）。然后再跑遍全国，用每个人的财富减去全国首富的财富去算指数，最后再求和。太折腾了（HBM 读写灾难）。
- Online Softmax 方法：在浙江省先算出浙江首富，直接把浙江的中间结果算出来记在小本子上。然后到了江苏省，发现江苏首富比浙江首富多 100 万。这个时候 **不需要重新计算浙江的数据** ，只需要用一个基于“100万差额”的公式（ $e^{m_{o l d} - m_{n e w}}$ ），把小本子上的浙江中间结果 **打个折** ，再加上江苏的结果，就等于把两个省合并计算了。

在 FlashAttention 中，这个“小本子”就是 GPU 速度极快的 SRAM。依靠这个数学公式的等价变换，我们把 $N \times N$ 级别的巨大中间矩阵读写，变成了只需要在 SRAM 中维护 $m$ 、 $d$ 、 $O$ 几个极小的向量，从而实现了性能的飞跃。

## 四. 由一维向量到二维矩阵的推广

现在将上述的推导推广到二维空间，上述的最大值 $m$ 和分母和 $d$ 有了新的含义。注意，这里的最大值绝对不是整个 $N \times N$ 输入矩阵的全局唯一最大值。在注意力机制中，Softmax 是 **按行（Row-wise）** 计算的。因此， **每一行都有自己的最大值** $m$ **和自己的分母** $d$ 。当我们把它拓展到矩阵分块计算时，所谓的“最大值”其实是一个 **列向量（Vector）** 。为了彻底看清 FlashAttention 中矩阵维度的 Online Softmax 是怎么运作的，我们把视角切到矩阵运算上来。

### 1\. 明确目标：Softmax 在矩阵中是怎么算的？

假设注意力分数矩阵 $S = Q K^{T}$ 是一个 $N \times N$ 的矩阵。

- 矩阵的 **每一行** 代表一个 Query（查询词）。
- 矩阵的 **每一列** 代表一个 Key（被查询词）。
- 我们要做的，是让 **每一个 Query 去对所有的 Key 求 Softmax** 。也就是说，Softmax 每一行的加和必须等于 1。

因此， **第 1 行有一个最大值** $m^{\left(\right. 1 \left.\right)}$ **，第 2 行有一个最大值** $m^{\left(\right. 2 \left.\right)}$ **… 整个矩阵有** $N$ **个最大值** ，它们拼起来就是一个长度为 $N$ 的列向量 $\mathbf{m}$ 。同理，分母也是一个长度为 $N$ 的列向量 $\mathbf{d}$ 。

### 2\. 矩阵分块：FlashAttention 的双重循环

在实际硬件（GPU）中，由于 SRAM 容量极其有限，我们既不能一次性塞入所有的 Query，也不能一次性塞入所有的 Key。FlashAttention 的做法是： **将** $Q$ **按行切成一块块（Row block），将** $K$ **和** $V$ **按照列也切成一块块（Column block）。**

![](https://picx.zhimg.com/v2-3e46da8633c9634c62abf2a912d55d39_1440w.jpg)

Flash Attention: Tiling and Online-Softmax Iteration Diagram

假设我们拿出 $Q$ 的第 1 块，记作 $Q_{b l o c k}$ （假设包含 $N_{1}$ 个 Query，即 $N_{1}$ 行）。 现在，我们要为这 $N_{1}$ 个 Query 计算出最终的注意力输出。

### 步骤一： Qblock 乘以 K 的第 1 块（记作 K1）

1. **算分数：** $S_{1} = Q_{b l o c k} \times \left(\right. K_{1} \left.\right)^{T}$ 。此时 $S_{1}$ 是一个 $N_{1} \times N_{1}$ 的局部小矩阵。
2. **找局部最大值：** 对 $S_{1}$ 的 **每一行** 找最大值，得到一个长度为 $N_{1}$ 的列向量 $\mathbf{m}_{1}$ 。
3. **算局部分母：** 对 $S_{1}$ 每一行减去对应的 $m_{1}$ 算指数和，得到长度为 $N_{1}$ 的列向量 $\mathbf{d}_{1}$ 。
4. **算局部输出：** 拿到对应位置的 $V_{1}$ ，计算 $O_{1} = \left(\right. exp ⁡ \left(\right. S_{1} - \mathbf{m}_{1} \left.\right) \left.\right) \times V_{1}$ ，此时 $O_{1}$ 是 $N_{1} \times d_{h e a d}$ 的矩阵。

### 步骤二： Qblock 乘以 K 的第 2 块（记作 K2）

现在，这 $N_{1}$ 个 Query 遇上了下一批 Key。

1. **算分数：** $S_{2} = Q_{b l o c k} \times \left(\right. K_{2} \left.\right)^{T}$ 。同样是个 $N_{1} \times N_{1}$ 的矩阵。
2. **找局部最大值：** 对 $S_{2}$ 的每一行找最大值，得到列向量 $\mathbf{m}_{2}$ （长度 $N_{1}$ ）。
3. **算局部分母：** 得到列向量 $\mathbf{d}_{2}$ （长度 $N_{1}$ ）。
4. **算局部输出：** 计算 $O_{2} = \left(\right. exp ⁡ \left(\right. S_{2} - \mathbf{m}_{2} \left.\right) \left.\right) \times V_{2}$ ，此时 $O_{2}$ 也是 $N_{1} \times d_{h e a d}$ 的矩阵。

### 步骤三：矩阵级别的合并（Online Softmax）

现在，我们要把这两部分的进度合并。 **注意，接下来的所有操作都是对这** $N_{1}$ **行“独立且并行”地进行的（按元素进行运算）：**

1. **更新每一行的最大值：** $\mathbf{m}_{n e w} = max \left(\right. \mathbf{m}_{1} , \mathbf{m}_{2} \left.\right)$ （逐元素取大，得到一个新的长度为 $N_{1}$ 的向量）。
2. **计算每一行的“补偿系数”向量：** 计算 $\mathbf{r}_{1} = exp ⁡ \left(\right. \mathbf{m}_{1} - \mathbf{m}_{n e w} \left.\right)$ 和 $\mathbf{r}_{2} = exp ⁡ \left(\right. \mathbf{m}_{2} - \mathbf{m}_{n e w} \left.\right)$ 。 *注：由于* $m_{1} , m_{2} , m_{n e w}$ *都是长度 64 的列向量，所以* $\mathbf{r}_{1} , \mathbf{r}_{2}$ *也是长度 64 的列向量，代表这* $N_{1}$ *个 Query 各自的缩放比例。*
3. **更新全局分母向量（按行缩放相加）：** $\mathbf{d}_{n e w} = \mathbf{d}_{1} \bigodot \mathbf{r}_{1} + \mathbf{d}_{2} \bigodot \mathbf{r}_{2}$ *（* $\bigodot$ *表示逐元素相乘，即每一行的旧分母乘上这一行独有的补偿系数）。*
4. **更新全局输出矩阵（按行缩放相加）：** $O_{n e w} = O_{1} \bigodot \mathbf{r}_{1} + O_{2} \bigodot \mathbf{r}_{2}$ *（这里的* $\mathbf{r}$ *会通过广播机制 Broadcast 作用于* $O$ *的每一行之上）。*

就这样，这 $N_{1}$ 个 Query 依次遍历完所有的 $K$ 和 $V$ 块之后，最后只需要执行一步除法，对于这 $N_{1}$ 行中的每一行 $i$ ，最终输出矩阵的第 $i$ 行 ：

$(*) \frac{O_{n e w} \left[\right. i \left]\right.}{\mathbf{d}_{n e w} \left[\right. i \left]\right.}$

注意以下几点：

1. **最大值** $m$ **和分母** $d$ **不是标量（Scalar），而是列向量（Column Vector）。**
2. FlashAttention 的矩阵 Online Softmax，本质上就是把“一维在线更新法则”， **在矩阵的每一行上同时、并行地应用了一遍** 。
3. 正是因为它是 **按行独立** 的，所以我们在切分矩阵时，把 Query 切成小块存进 SRAM 中是极其合理的。每个 Query 在看完了所有的 Key 之后，这单独一行的 Softmax 也就完美算出了，中间丝毫不需要借助 HBM（全局慢速内存）来存储那个庞大的 $N \times N$ 分数矩阵。

## 五. 分子分母分开计算

仔细看上述的计算，会发现最终计算的 $O_{t o t a l}$ 其实是一个分子分母分别独立计算的过程， **在局部计算** $O_{1}$ **和** $O_{2}$ **时，确实不需要** $d$ **参与** 。

为什么会这样？因为这里隐含了 FlashAttention 算法设计中一个极其巧妙的思想： **“分子与分母分离，并行独立更新”** ，这也是 FlashAttention-2 相比于 FlashAttention-1 最大的性能优化点之一。下面，我们来详细拆解一下为什么 $d$ 在前期的计算中“隐身”了。

### 1\. 传统思路的直觉陷阱

按照我们正常的数学直觉，Softmax 输出的是“概率分布”，概率相加必须等于 1。 所以，直觉上我们算第一块的局部输出时，应该长这样：

$(\text{5}.\text{1}) O_{1 _ 归 一 化} = \frac{exp ⁡ \left(\right. S_{1} - m_{1} \left.\right)}{d_{1}} \times V_{1}$

在这个直觉公式里， $d_{1}$ 确实参与计算了。如果采用这种方式，我们维护的就是一个 **“已经被归一化的局部结果”** 。初代的 FlashAttention-1 确实是这么做的，它在每一步循环里都除以了 $d$ 。

### 2\. 为什么现在的 FA（尤其是 FA-2）不让 d 参与 O 的局部计算？

如果我们在每处理一块数据时，都提前除以 $d_{1} , d_{2}$ ，会带来两个致命问题：

1. **合并非常麻烦** ：如果你拿到的是归一化后的 $O_{1 _ 归 一 化}$ ，在和第 2 块合并时，你必须把它“反向乘回去”恢复成分子，再加上第 2 块的分子，然后再除以新的分母。公式会变得极其复杂且啰嗦。
2. **硬件极其低效** ：在 GPU 中， **除法（Division）是非常慢的指令** 。如果我们在内层循环（每处理一个小块）都做一次除法，极大地拖慢了计算速度。

### 3\. 最优解：分子分母“兵分两路”，各算各的

为了避免上述问题，FlashAttention-2 算法做了一个惊艳的决定： **我们要维护的** $O$ **，根本就不是最终的注意力结果，而仅仅是“分子（Numerator）的累加和”！**

我们把注意力公式劈成上下两半来看： 最终结果：

$(\text{5}.\text{2}) \frac{所有块的分子累加}{所有块的分母累加}$

**这也就意味着，在处理每一个小块时，** $O$ **（分子）和** $d$ **（分母）是两条平行线，互不干扰！**

### 在处理 Block 1 时

我们算出一个局部向量， **这是分母 1 的雏形** ： $(\text{5}.\text{3}) d_{1} = \sum exp ⁡ \left(\right. S_{1} - m_{1} \left.\right)$

我们算出一个局部矩阵， **这是分子 1 的雏形，并没有除以** $d_{1}$ ： $(\text{5}.\text{5}) O_{1} = exp ⁡ \left(\right. S_{1} - m_{1} \left.\right) \times V_{1}$

### 在处理 Block 2 时

算局部向量， **这是分母 2 的雏形** ：  
$(\text{5}.\text{5}) d_{2} = \sum exp ⁡ \left(\right. S_{2} - m_{2} \left.\right)$

- 算局部矩阵， **这是分子 2 的雏形，并没有除以** $d_{1}$ ：  
	$(\text{5}.\text{6}) O_{2} = exp ⁡ \left(\right. S_{2} - m_{2} \left.\right) \times V_{2}$  
	  
	在合并阶段（Online 更新）

分母和分子依然是 **各自独立** 用同样的“补偿系数”进行缩放合并的：

**合并分母** ： $(\text{5}.\text{7}) d_{n e w} = d_{1} \bigodot exp ⁡ \left(\right. m_{1} - m_{n e w} \left.\right) + d_{2} \bigodot exp ⁡ \left(\right. m_{2} - m_{n e w} \left.\right)$

**合并分子** ： $(\text{5}.\text{8}) O_{n e w} = O_{1} \bigodot exp ⁡ \left(\right. m_{1} - m_{n e w} \left.\right) + O_{2} \bigodot exp ⁡ \left(\right. m_{2} - m_{n e w} \left.\right)$

你看， $O_{1}$ 、 $O_{2}$ 、 $O_{n e w}$ 在整个内层循环更新的过程中，都只是纯粹的 **分子（未归一化的加权和）** 。既然是分子， $d$ 当然没有任何理由去干涉它。

### 4\. 那么，d 什么时候才登场？

$d$ 就像一个耐心积攒能量的计数器，它在经历了切块1、切块2、切块3……一直到这 64 个 Query 遍历完了 **所有** 的 Key 之后，得到了最终的全局分母 $d_{f i n a l}$ 和全局分子 $O_{f i n a l}$ 。

只有在这 **最后一刻（出了循环之后）** ，才执行唯一的一次除法：

$(\text{5}.\text{9}) 最终\text{ Attention Output} = \frac{O_{f i n a l}}{d_{f i n a l}}$

FlashAttention-2 相比 FlashAttention-1 的核心性能飞跃点，就是将除以 $d$ 的操作（scaling）从内层循环中 **剥离** 出来，推迟到最后一步才执行。在计算局部 $O_{1}$ 和 $O_{2}$ 时，只算分子不除分母。这样不仅省去了大量昂贵的 GPU 除法操作，还让数学合并的逻辑变得前所未有的干净利落！

## 六. 矩阵分块与内外双层循环的对应关系

弄懂了内外循环的对应关系，才能真正明白 FlashAttention 是如何在 GPU 物理硬件（SRAM 和 HBM）上压榨出极限性能的。 **在目前广泛使用的 FlashAttention-2（以及后续版本）中，外层循环是 Query (Q) 分块，内层循环是 Key (K) 和 Value (V) 分块。** 但如果去看最早的 FlashAttention-1 论文，我们会惊讶地发现 **它是反过来的** ！这就是 FA 演进史上最精彩的工程优化之一。

![动图封面](https://pic1.zhimg.com/v2-e4790de9828d138eba94de87dd521be8_b.jpg)

FA\_2 loop

以下我们将这两个版本做对比分析。

计算输出 $O = \text{Softmax} \left(\right. Q K^{T} \left.\right) V$ ，有一个核心维度对应关系： **输出矩阵** $O$ **的行数，是跟着** $Q$ **走！** 一个 Query 对应最终的一个 Output 向量。为了算出这一个完整的 Output 向量，这个 Query 必须和 **所有的 K 和 V** 发生计算。

### 1\. 现代标准：FlashAttention-2 的双层循环（外 Q 内 KV）

在 FA2 中，作者 Tri Dao 修改了循环顺序。这也是目前业界所有高效 Attention（比如 vLLM, TensorRT-LLM 里的实现）的标准做法。

![](https://pic4.zhimg.com/v2-abc409a3758d9cf79374a3bdd7654a35_1440w.jpg)

FA 2

**对应关系：**

- **外层循环（Outer Loop）** ：遍历 **Query (Q)** 的分块。
- **内层循环（Inner Loop）** ：遍历 **Key (K) 和 Value (V)** 的分块。

**GPU 是怎么干活的？**

1. **【外层循环开始】** ：GPU 从显存 (HBM) 中读取 **Block** $Q_{1}$ ，放进高速缓存 (SRAM)。
2. 在 SRAM 里，为 $Q_{1}$ 准备好对应的 $O_{b l o c k}$ （初始为 0）、 $m_{b l o c k}$ （初始为极小值）、 $d_{b l o c k}$ （初始为 0）。
3. **【内层循环开始】** ：
- 读取 **Block** $K_{1} , V_{1}$ 进 SRAM。
- 计算 $Q_{1} \times K_{1}^{T}$ ，利用 Online Softmax 的“整体缩放”魔术，更新 SRAM 里的 $O_{b l o c k} , m_{b l o c k} , d_{b l o c k}$ 。
- 丢弃 $K_{1} , V_{1}$ 。
- 读取 **Block** $K_{2} , V_{2}$ 进 SRAM。
- 计算 $Q_{1} \times K_{2}^{T}$ ，继续更新 SRAM 里的 $O_{b l o c k} , m_{b l o c k} , d_{b l o c k}$ 。
- 丢弃 $K_{2} , V_{2}$ 。
- ……一直遍历完所有的 KV 分块。
1. **【内层循环结束】** ：此时，SRAM 里的 $O_{b l o c k}$ 已经和所有的 KV 都计算过了，它 **彻底“熟”了** ，变成了最终精确的注意力输出。
2. **【外层循环收尾】** ：把这块彻底算完的 $O_{b l o c k}$ ， **一次性** 写回显存 (HBM)！
3. 进入下一个外层循环（读取 $Q_{2}$ ……）。

**为什么这么设计极其巧妙？** 因为中间不断更新的 $O_{b l o c k} , m_{b l o c k} , d_{b l o c k}$ 始终安安全全地待在超高带宽的 SRAM 里。对于任何一块 $O$ ，它 **只被写入显存一次** ！这把显存的读写压力降到了最低。

### 2\. 历史的弯路：FlashAttention-1 为什么慢一点？（外 KV 内 Q）

在初代 FA1 论文中，循环顺序是反的：

- **外层循环（Outer Loop）** ：遍历 Key (K) 和 Value (V) 的分块。
- **内层循环（Inner Loop）** ：遍历 Query (Q) 的分块。
![](https://picx.zhimg.com/v2-40e435b41b7a09ced1fdbfe0f33a1165_1440w.jpg)

FA 1

**当时的逻辑是这样的：**

1. 读取 Block $K_{1} , V_{1}$ 进 SRAM。
2. 内部循环遍历所有的 $Q_{1} , Q_{2} . . . Q_{N}$ 。
3. 算 $Q_{1}$ 和 $K_{1} V_{1}$ 的结果，更新 $Q_{1}$ 对应的 $O_{1}$ 。
4. 算 $Q_{2}$ 和 $K_{1} V_{1}$ 的结果，更新 $Q_{2}$ 对应的 $O_{2}$ 。

**灾难在哪里？** 当在算 $Q_{1}$ 时，你需要把 $O_{1} , m_{1} , d_{1}$ 从显存读进 SRAM，更新完后， **必须马上写回显存** （因为 SRAM 太小，装不下所有的 $O$ ）。等外层循环推进到 $K_{2} , V_{2}$ 时，内层循环又要重新算 $Q_{1}$ ，你又得把 $O_{1}$ 从显存读出来，更新完，再写回去！

如果 K 和 V 被切成了 10 个 Block，那么你的每一个输出块 $O_{i}$ 就要在 SRAM 和显存之间 **来回搬运读写 10 次** ！

**总结一下 FA1 和 FA2 的本质区别：**

- **FA1** ：我拿着同一块数据（KV），去挨个更新所有人（Q）的进度条。结果是频繁更新显存里的进度条（O）。
- **FA2** ：我揪住一个人（Q），让他一次性把所有数据（KV）看完，当场得出最终结果。结果是显存只被写一次。

### 3\. 还有一个隐藏的红利：Causal Mask（因果掩码）

在大模型推理和训练（GPT架构）中，只能看到前面的 Token，不能看后面的 Token（即下三角矩阵）。

如果使用 **FA2（外 Q 内 KV）** 的逻辑，有一个天然的巨大优势：

假设我们现在外层循环处理的是第 3 块 $Q$ ：当内层循环去遍历 KV 的时候，它只需要遍历第 1 块、第 2 块、第 3 块的 KV 就可以 **直接 break 提前终止了** ！因为第 4 块及以后的 KV 在未来，直接被 Mask 掉了，根本不需要发生内存读取和矩阵乘法。这种 “ **及早停止** ”（Early Stop）在 FA2 的外 Q 内 KV 架构下实现起来极其自然，直接砍掉了一半的计算量。

### 4\. 示例代码

```python
import torch
import time

def simulate_flash_attention():
    # ---------------- 1. 参数配置 ----------------
    N = 4096        # 序列长度 Sequence Length
    D = 128         # 注意力头维度 Head Dimension
    Br = 64         # Query 的分块大小 Block size for Q
    Bc = 64         # Key/Value 的分块大小 Block size for K/V
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前运行设备: {device}")

    # 初始化 Q, K, V (为了公平，均使用相同的随机数)
    Q = torch.randn(N, D, device=device)
    K = torch.randn(N, D, device=device)
    V = torch.randn(N, D, device=device)

    Tr = N // Br    # Q 的块数
    Tc = N // Bc    # K/V 的块数
    scale = D ** -0.5 # \sqrt{d} 缩放因子

    # ---------------- 2. FA1：外层 K/V，内层 Q ----------------
    def fa1_outer_kv(Q, K, V):
        # 申请驻留在 HBM(全局显存) 中的变量
        O_global = torch.zeros(N, D, device=device)
        m_global = torch.full((N, 1), float('-inf'), device=device)
        d_global = torch.zeros(N, 1, device=device)

        # 外层循环：遍历 K/V 分块
        for j in range(Tc):
            K_j = K[j*Bc : (j+1)*Bc]
            V_j = V[j*Bc : (j+1)*Bc]
            
            # 内层循环：遍历 Q 分块
            for i in range(Tr):
                Q_i = Q[i*Br : (i+1)*Br]
                
                # 💥 性能灾难点：必须从全局显存读取历史 O, m, d
                O_i = O_global[i*Br : (i+1)*Br]
                m_i = m_global[i*Br : (i+1)*Br]
                d_i = d_global[i*Br : (i+1)*Br]

                # --- 核心数学计算（Online Softmax） ---
                S_ij = torch.matmul(Q_i, K_j.T) * scale
                m_local, _ = torch.max(S_ij, dim=-1, keepdim=True)
                
                m_new = torch.maximum(m_i, m_local)
                exp_diff = torch.exp(m_i - m_new)
                P_ij = torch.exp(S_ij - m_new)
                d_local = torch.sum(P_ij, dim=-1, keepdim=True)
                
                # 更新状态
                d_new = d_i * exp_diff + d_local
                O_new = O_i * exp_diff + torch.matmul(P_ij, V_j)
                
                # 💥 性能灾难点：必须将更新后的 O, m, d 频繁写回全局显存
                O_global[i*Br : (i+1)*Br] = O_new
                m_global[i*Br : (i+1)*Br] = m_new
                d_global[i*Br : (i+1)*Br] = d_new
                
        return O_global / d_global # 最后除以最终的分母 d

    # ---------------- 3. FA2：外层 Q，内层 K/V ----------------
    def fa2_outer_q(Q, K, V):
        # 申请驻留在 HBM(全局显存) 中的变量
        O_global = torch.zeros(N, D, device=device)
        m_global = torch.full((N, 1), float('-inf'), device=device)
        d_global = torch.zeros(N, 1, device=device)

        # 外层循环：遍历 Q 分块
        for i in range(Tr):
            Q_i = Q[i*Br : (i+1)*Br]
            
            # ✨ 性能红利点：在本地(模拟 SRAM) 初始化局部的 O, m, d
            O_local = torch.zeros(Br, D, device=device)
            m_local = torch.full((Br, 1), float('-inf'), device=device)
            d_local_val = torch.zeros(Br, 1, device=device)
            
            # 内层循环：遍历 K/V 分块
            for j in range(Tc):
                K_j = K[j*Bc : (j+1)*Bc]
                V_j = V[j*Bc : (j+1)*Bc]
                
                # --- 核心数学计算（与 FA1 完全一致） ---
                S_ij = torch.matmul(Q_i, K_j.T) * scale
                m_block, _ = torch.max(S_ij, dim=-1, keepdim=True)
                
                m_new = torch.maximum(m_local, m_block)
                exp_diff = torch.exp(m_local - m_new)
                P_ij = torch.exp(S_ij - m_new)
                d_block = torch.sum(P_ij, dim=-1, keepdim=True)
                
                # 直接在局部变量(SRAM) 上累加
                d_local_val = d_local_val * exp_diff + d_block
                O_local = O_local * exp_diff + torch.matmul(P_ij, V_j)
                m_local = m_new
                
            # ✨ 性能红利点：内层循环彻底结束，当前 Q_i 熟了，只往显存写回 1 次！
            O_global[i*Br : (i+1)*Br] = O_local
            m_global[i*Br : (i+1)*Br] = m_local
            d_global[i*Br : (i+1)*Br] = d_local_val
            
        return O_global / d_global

    # ---------------- 4. 性能测试 (预热 + 多次测量) ----------------
    print("\n正在验证两种方法结果的一致性...")
    out1 = fa1_outer_kv(Q, K, V)
    out2 = fa2_outer_q(Q, K, V)
    # 检查两个矩阵输出的最大误差
    max_diff = torch.max(torch.abs(out1 - out2)).item()
    print(f"最大计算误差 (应接近0): {max_diff:.8e}")

    # CUDA 预热
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    runs = 10
    
    # 测 FA1 时延
    start_time = time.perf_counter()
    for _ in range(runs):
        fa1_outer_kv(Q, K, V)
        if torch.cuda.is_available(): torch.cuda.synchronize()
    time_fa1 = (time.perf_counter() - start_time) / runs

    # 测 FA2 时延
    start_time = time.perf_counter()
    for _ in range(runs):
        fa2_outer_q(Q, K, V)
        if torch.cuda.is_available(): torch.cuda.synchronize()
    time_fa2 = (time.perf_counter() - start_time) / runs

    print("\n========= 时延对比 (Python 模拟层级) =========")
    print(f"FA1 (外层 K/V, 内层 Q) 时延: {time_fa1 * 1000:.2f} ms")
    print(f"FA2 (外层 Q, 内层 K/V) 时延: {time_fa2 * 1000:.2f} ms")
    print(f"提速比 (Speedup): {time_fa1 / time_fa2:.2f}x")

if __name__ == "__main__":
    simulate_flash_attention()
```

运行输出：

```
当前运行设备: cuda

正在验证两种方法结果的一致性...
最大计算误差 (应接近0): 0.00000000e+00

========= 时延对比 (Python 模拟层级) =========
FA1 (外层 K/V, 内层 Q) 时延: 679.73 ms
FA2 (外层 Q, 内层 K/V) 时延: 404.28 ms
提速比 (Speedup): 1.68x
```

实际运行结果 **完美地印证了我们之前的理论推导** ！深度解剖一下跑出来的这三个关键数据：

### 5\. 结果分析

### 1\. 最大计算误差：0.00000000e+00

这证明了我们上一篇反复推导的 **Online Softmax 的“缩放魔术”在数学上是绝对无损的** ！虽然 FA2 （外 Q）和 FA1（外 K/V） 的计算顺序完全不同，累加的过程完全不同，但凭借着 $e^{m_{o l d} - m_{n e w}}$ 这个极其精妙的缩放因子，最终算出来的每一个浮点数都和标准步骤严丝合缝，没有一丁点精度损失。

### 2\. 时延骤降：从 679.73 ms 降到 404.28 ms

**在做了一模一样多的矩阵乘法（算力总量完全一致）的情况下** ，仅仅是把外层循环和内层循环的位置调换了一下，速度就直接 **提升了 68%** ！ 这就是算法工程学（Algorithm Engineering）的魅力。在底层，它们计算的乘加操作（FLOPs）一模一样，省下的这 275 毫秒， **全是少做“无用功（反复读写显存）”省下来的时间** 。

### 3\. 深度思考：为什么提速比“只有” 1.68x？

在真实的 FlashAttention 论文中，FA2 比标准 Attention 快了数倍，也比 FA1 明显快得多。为什么在我们的代码里，提速是 1.68 倍？

因为 **PyTorch 的 Python API 限制了硬件的真正实力** 。

在我们的模拟代码里，有一层“物理隔离”是我们用 Python 无法突破的：

- 当我们写下 `O_local = O_local * exp_diff + torch.matmul(P_ij, V_j)` 时，虽然我们 **逻辑上** 希望 `O_local` 待在高速 SRAM 里，但 PyTorch 每执行一个加法或乘法，底层都会生成一个新的 Tensor，并 **把它写回全局显存（HBM）** ！
- 此外，Python 的 `for` 循环每次迭代都会产生极大的 CPU 调度开销（Kernel Launch Overhead）。

**但在真正的 CUDA C++ / Triton 实现中，发生了什么？** 在真正的硬件底层， `O_local` 、 `m_local` 、 `d_local` 会被直接锁死在 GPU 计算单元旁边的 **寄存器（Registers）** 或 **共享内存（Shared Memory / SRAM）** 里！ 在内层遍历 KV 的几百次循环中，它们 **一次都不会触碰显存** ！速度比显存快 10 倍以上。只有当内层循环彻底结束时，才会执行唯一的一次显存写入。

因此，跑出的这 `1.68x` 的加速，仅仅是省去了 **“PyTorch 层面大矩阵切片拼接和局部赋值的显存开销”** 。如果是写到底层 CUDA 里，由于彻底避免了 HBM 的 IO 墙，加速比会更加恐怖！

**一个小实验建议：** 可以把代码里的序列长度 `N = 4096` 改成 `N = 8192` 或更大，再跑一次。这个时候我们会惊奇地发现， **随着序列长度的增加，提速比会越来越大（比如飙升到 2.5x 甚至更高）** 。因为 FA1 对全局显存的写回次数是随切块数量呈平方级爆炸的，而 FA2 始终是线性的。

### 总结

| 维度 | 外层循环 (Outer Loop) | 内层循环 (Inner Loop) |
| --- | --- | --- |
| 对应分块 | Query (Q) 分块 | Key (K) / Value (V) 分块 |
| SRAM 常驻内容 | Q\_{block}，累加的 O\_{block}, m\_{block}, d\_{block} | 每次循环新读入的 K\_{block}, V\_{block} |
| HBM 写回频率 | 每次外循环结束写回 1次 完整的 O\_{block} | 根本不需要写回 KV |
| 循环逻辑一句话 | 拿出一个 Q 块不动 | 像流水线一样把所有 KV 块灌进来更新 Q |

弄懂了这个循环逻辑，不仅透彻理解了 FA2 的代码架构，也真正明白了上一篇里 Online Softmax 那个 $d_{n e w}$ 和 $O_{n e w}$ 到底是在哪个循环层级里发生更新的。

## 参考文献

- [From Online Softmax to FlashAttention](https://link.zhihu.com/?target=https%3A//courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [GPU MODE Lecture 12: Flash Attention – Christian Mills](https://link.zhihu.com/?target=https%3A//christianjmills.com/posts/cuda-mode-notes/lecture-012/)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | Princeton NLP Group](https://link.zhihu.com/?target=https%3A//princeton-nlp.github.io/flash-atttention-2/)
- [What is flash attention? | Modular](https://link.zhihu.com/?target=https%3A//docs.modular.com/glossary/ai/flash-attention/)

还没有人送礼物，鼓励一下作者吧

编辑于 2026-04-12 10:56・江苏[源码（源代码）](https://www.zhihu.com/topic/20031938)[PyTorch](https://www.zhihu.com/topic/20075993)