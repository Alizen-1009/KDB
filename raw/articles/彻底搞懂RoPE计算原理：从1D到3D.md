---
title: "彻底搞懂RoPE计算原理：从1D到3D"
source: "https://mp.weixin.qq.com/s/8_0V6Yxw-_03lCY3ujPVtA"
author:
  - "[[kaiyuan]]"
published:
created: 2026-05-11
description: "RoPE是一种专门为Transformer设计的旋转位置编码：它不通过额外添加位置向量，而是直接在query和key上执行按维度分频的旋转变换，使得向量点积能够体现相对位置差，从而支持超长上下文、平移不变的注意力模式，并在多模态场景中自然扩展至2D/3D位置。"
tags:
  - "clippings"
---
kaiyuan *2026年4月8日 07:30*

![图片](https://mmbiz.qpic.cn/mmbiz_gif/uIP3tuXZx8BXuw5wK7j3avmicib6aZVBqicxsN6ricR1AIiam4Jh59fsLhzU4cREAxrfxPqILpAuUfrVROyIugfOofqucCL4XR5XlfxQg6IPNibmw/640?wx_fmt=gif&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

点击 **蓝字** ，关注我们

RoPE（Rotary Position Embedding） <sup>[1]</sup> 是一种专门为Transformer设计的 **旋转位置编码** ：它不通过额外添加位置向量，而是直接在query和key上执行按维度分频的旋转变换，使得向量点积能够体现相对位置差，从而支持超长上下文、平移不变的注意力模式，并在多模态场景中自然扩展至2D/3D位置。要理解大模型中的RoPE，可分三个层次：二维向量的点积与旋转运算、多维空间中按平面分组的线性变换，以及在实际大模型中的高效实现与视觉扩展。

**本文探讨以下问题：**

- 为什么RoPE能让Attention“只在乎相对位置”，而无需显式存储相对位置矩阵？
- RoPE中频率序列的取值依据是什么？它具体控制了哪些方面？
- 多维向量是如何被“拆分成一系列二维平面”进行旋转变换的？这与代码中的rotate\_half 操作有何关系？
- 1D RoPE如何自然地扩展到图像中的2D场景，以及视频中的3D位置编码？

**01**

**向量乘法预备知识**

在了解RoPE之前，先回顾一些基础的向量乘法知识，循序渐进才能抓住RoPE的关键内容；基础较好的读者可直接跳至第2节。

相关的知识的代码用例参考 <sup>[2]</sup> ：https://github.com/CalvinXKY/InfraTech/blob/main/models/modules/rope\_principle.ipynb

**1.1 二维向量的点积（内积）**

神经网络中有大量矩阵运算，矩阵乘法的分量运算可视作向量之间点积运算。为了理解方便，向量点积可从二维向量开始，再扩展到多维。两个向量点积运算的计算过程：

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

用平面空间表达向量点积

定义：设 ，点积：

 $\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2$

几何含义：

 $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \, \|\mathbf{b}\| \, \cos\theta$

其中θ是两向量之间的夹角。

- 若点积为正 → 夹角小于 90°（方向大致相同）
- 若点积为零 → 两向量垂直（正交）
- 若点积为负 → 夹角大于 90°（方向相反）

另一种几何解释：点积等于一个向量在另一个向量上的投影长度乘以另一个向量的长度。

说明：RoPE中所谓的旋转作用的就是定义中的，通过调整theta角来达到注入位置信息。

**1.2 二维向量的旋转**

二维向量在平面空间中绕原点旋转一个角度theta，得到新的向量，新坐标的计算可借助旋转矩阵来完成。

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

向量b逆时针旋转一个角度theta得到b'

二维旋转矩阵的作用就是将一个二维向量绕原点旋转一个指定的角度。

例如，二维向量

 $\mathbf{v}=\begin{bmatrix}x \\ y\end{bmatrix}$

用旋转矩阵

 $R(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}$

左乘它：

 $\mathbf{v}' = R(\theta) \cdot \mathbf{v} = \begin{bmatrix}x\cos\theta - y\sin\theta \\ x\sin\theta + y\cos\theta\end{bmatrix}$

得到的新向量就是原向量逆时针旋转θ角后的结果。旋转矩阵的核心性质是：

- 行列式为 +1（保持面积/体积不变）
- 满足正交性：
	 $R^T R = I$

**1.3 向量点积与旋转的关系**

在RoPE中采用的关键知识：“两个向量分别旋转后再做点积”的含义。

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

两个向量分别旋转后再做点积，旋转角度转化为相对变化：

过程：设 a、b的极角分别为，则原始夹角。

旋转后：新夹角

 $\alpha' = \phi_{b'} - \phi_{a'} = (\phi_b + \theta_2) - (\phi_a + \theta_1) = \alpha + \theta_2 - \theta_1$

点积：

 $\mathbf{a}' \cdot \mathbf{b}' = |\mathbf{a}'||\mathbf{b}'| \cos\alpha' = |\mathbf{a}||\mathbf{b}| \cos(\alpha + \theta_2 - \theta_1)$ 

若写成，则与上式一致。

两个向量的绝对位置差异，经点积运算后转化为相对位置的变化（即），这正是RoPE所依赖的关键数学原理。

**02**

**Attention运算引入旋转角度**

在Attention计算中，用Q与做矩阵乘法得到分数（scores）这一步骤，本质上是矩阵乘法；对于单个score而言，它对应两个向量的点积。结合前述向量乘法的知识，假设文本序列经编码后隐藏层维度 hidden\_size=2，编码数据的形状为 \[seq\_len, hidden\_size\]，此时每个token可视为一个二维向量。

举个例子，假设输入序列为“我是ky”，编码后有4个token，则shape=\[4, 2\]。

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

QK相乘计算Sores的例子

QK乘法中若不引入位置编码，计算无法分别tokens的相对位置。先来看一个score计算的例子：Q中token‘k’与K中token‘是’、token‘k’做点积求对应的score。从距离上看，token‘k’离token‘是’显然比离自己更远，能否在分数上体现这种差异？

即，如何标记序列中不同位置的token，从而影响最终score的大小？

答案是：每个位置旋转一个不同的角度。

为了体现距离，现在给字符串从左到右各token依次递增旋转2°。即token‘我’旋转0°，token‘是’旋转2°，最后一个token‘y’旋转6°。因为是二维数据，可通过坐标图展示，如下所示：

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

所有向量旋转指定角度

在计算点积时，可使用公式：

 $\mathbf{a}' \cdot \mathbf{b}' = |\mathbf{a}'||\mathbf{b}'| \cos\alpha' = |\mathbf{a}||\mathbf{b}| \cos(\alpha + \theta_2 - \theta_1)$ 

很显然，序列位置越接近（θ <sub>1</sub> 与θ <sub>2</sub> 越接近）时，越小，对位置差的响应方式就越能体现相对关系。因此，score中便融入了相对位置信息。

同时可以看到，各向量上的旋转角θ本身是绝对的，但点积中体现的是相对关系，从而使attention中每个token都能感知与之相乘的另一token的位置。这正是RoPE的核心思想。

在计算query(q <sub>m</sub>)与key(k <sub>n</sub>)的内积时，RoPE的效果等价于引入一个相对位置依赖的旋转：

 $(R_m q_m)^\top (R_n k_n) = q_m^\top R_m^\top R_n k_n = q_m^\top R_{n-m} k_n$ 

其中分别为施加在位置上query与key的旋转矩阵（R为正交矩阵，且），故内积只依赖相对位置。

这里还有两个关键问题没有回答：

- Q/K的hidden\_size通常远大于2（如1024、2048），已不是把整个向量当作二维来旋转的情形，如何引入旋转位置信息？
- theta角度设置多少合适？

接下来依次分析：

**2.1 QK的多维向量如何旋转？**

通过线性代数知识，可知：

- 多维向量支持旋转操作；
- 多维向量的旋转方式不止一种。

这里我们仅关注部分坐标（平面）上的旋转，以四维情形为例。四维旋转矩阵是4×4的正交矩阵（行列式为+1），作用在四维向量上。

一个简单的四维旋转例子是在由两个坐标轴张成的平面内旋转（例如平面），而保持另外两个坐标不变。比如在平面内旋转角度θ，同时保持z和w不变。矩阵如下：

 $R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 & 0 \\ \sin\theta & \cos\theta & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$ 

具体数值举例：取，则：

 $R = \begin{bmatrix} 0 & -1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$ 

用它旋转四维向量 v=（1,0,0,0）：

 $\mathbf{v}' = R \cdot \mathbf{v} = \begin{bmatrix} 0\cdot1 + (-1)\cdot0 + 0\cdot0 + 0\cdot0 \\ 1\cdot1 + 0\cdot0 + 0\cdot0 + 0\cdot0 \\ 0\cdot1 + 0\cdot0 + 1\cdot0 + 0\cdot0 \\ 0\cdot1 + 0\cdot0 + 0\cdot0 + 1\cdot0 \end{bmatrix} = \begin{bmatrix}0 \\ 1 \\ 0 \\ 0\end{bmatrix}$ 

即（1,0,0,0）→（0,1,0,0），这是在平面内旋转90°。而z和w分量完全不变。

更一般的四维旋转

四维空间中的旋转可以同时发生在两个相互正交的平面上（例如平面和平面），且两个平面的旋转角度独立。这样的矩阵是块对角的：

 $R(\theta, \phi) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 & 0 \\ \sin\theta & \cos\theta & 0 & 0 \\ 0 & 0 & \cos\phi & -\sin\phi \\ 0 & 0 & \sin\phi & \cos\phi \end{bmatrix}$ 

这就是四维旋转矩阵的典型形式。它满足（因为每个2×2子块的行列式都是1，乘积为1）。

在QK的乘法运算中，RoPE把d维向量拆成d/2个二维子空间，在每个子空间内做平面旋转（各子空间正交）。

设token向量的维度为d，各维下标为0,1,…，d-1。当d为偶数时，一种便于画图的划分是把相邻两维成对：\[0,1\]、\[2,3\]、…、\[d-2,d-1\]，共d/2个平面；而常见实现则把第i维与第i+d/2维配成一对，数学上仍是块对角正交变换，只是维度的编号方式不同。下面先用相邻两维成对的写法给出块对角形式的：

设d = 2n，则旋转矩阵为：

 $R_m = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\ \sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\ 0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\ 0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \cdots & \cos m\theta_n & -\sin m\theta_n \\ 0 & 0 & 0 & 0 & \cdots & \sin m\theta_n & \cos m\theta_n \end{pmatrix}$ 

其中是预先设定的频率

注意：在实际RoPE计算中，配对不是按相邻下标，而是前半段与后半段一一对应，即第i维与第i+d/2维配成一对（i=0,1,…,d/2-1），这种配对方式便于计算。

**2.2 theta角度设置多少合适？**

由上面的讨论可知，RoPE用位置m与多组基频共同决定各平面上的旋转。下面先用周期性与序列长度说明：为何需要随维度i变化的，而不能只靠每位置固定转一小角度这一种尺度。

对任一固定频率，位置m带来的相位是；由于cos以2π为周期，有

 $\cos(m\theta_i+\Delta)=\cos(m\theta_i+2\pi+\Delta)$

若只用一种很粗的步长（直觉上类似每步转一度的比喻），序列很长时，不同位置m在某些频率上容易落在同一等价类里，难以区分。

而在大模型中，序列长度可达128K/256K甚至更长，因此需要多组随维度i变化的基频（见下式），使不同位置在各频率上的组合相位在所需长度范围内尽可能可区分；这不是靠每个位置多转一度这种单一尺度能解决的。

除了位置，当hidden\_size>2时还存在多个二维平面，需要区分各平面对结果的影响，因此把平面序号（维度索引）也引入旋转角度的计算。

在RoPE中，θ由位置索引m与维度索引i共同决定。具体计算公式如下：

对于位置m（token在序列中的索引，从0开始），以及第i个旋转平面（i=0,…,d/2-1；与实现中rotate\_half一致时，该平面由第i维与第i+d/2维分量张成），该平面上的旋转角为：

 $\theta(m, i) = m \cdot \theta_i = m \cdot \left( 10000^{-\frac{2i}{d}} \right)$ 

其中：

- d为注意力头维度，即hidden\_size/num\_heads
- i的取值范围是0,1,…，d/2-1
- 是该维度对的基础频率（与位置无关，由维度索引决定）

这种分配方式，使得即使序列长度很大，所分配的旋转角度θ也几乎不会出现重叠。

**2.3 RoPE的计算过程**

**2.3.1 降低重复计算**

在大模型中，RoPE的运算通常分成两步：先预计算“系数矩阵”，前向传播时再用该矩阵乘Q或K，而不是显式地采用如下运算：

 $(R_m q_m)^\top (R_n k_n)$ 

关键在于计算效率：降低构造矩阵的成本。再看一下的内容：

 $R_m = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\ \sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\ 0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\ 0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & 0 & 0 & \cdots & \cos m\theta_n & -\sin m\theta_n \\ 0 & 0 & 0 & 0 & \cdots & \sin m\theta_n & \cos m\theta_n \end{pmatrix}$ 

其中：

 $\theta(m, i) = m \cdot \theta_i = m \cdot \left( 10000^{-\frac{2i}{d}} \right)$ 

基础频率的含义：每个元素代表一对正弦/余弦分量（对应一个旋转角度），即：

矩阵中各元素对每个Q/K位置的运算是相同的，尤其是其中cos与sin的取值。 因此可按最长序列预先构造好cos、sin，在attention迭代中复用。

一般步骤：

- 根据最长索引值L\_max构造频率表（freq\_table），即由构成的数据矩阵，shape为\[L\_max, d/2\]
- 计算freq\_table的cos与sin值。

**2.3.2避免显式矩阵乘法**

是块对角矩阵，若按完整稠密矩阵做乘法会造成大量无效计算。 RoPE的高效实现通常采用复数乘法形式，避免显式计算旋转矩阵。对于位置m，第i个平面内的分量（在相邻维记号下可记为）视为复数，旋转后为：

 $z' = z \cdot e^{i m \theta_i} = (x_{2i} + i x_{2i+1}) \cdot (\cos(m\theta_i) + i \sin(m\theta_i))$ 

展开得到实数部分即为旋转后的两个分量：

 $\begin{aligned} x'_{2i} &= x_{2i}\cos(m\theta_i) - x_{2i+1}\sin(m\theta_i) \\ x'_{2i+1} &= x_{2i}\sin(m\theta_i) + x_{2i+1}\cos(m\theta_i) \end{aligned}$ 

结合前面的分析，其中的和可以提前算好。多维通用公式如下 <sup>[3]</sup> ：

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

RoPE原文中的计算公式

常见的计算过程代码示例：

1) cos与sin值的构造：

```ini
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)position_ids_expanded = position_ids[:, None, :].float()freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)emb = torch.cat((freqs, freqs), dim=-1)cos = emb.cos() * self.attention_scalingsin = emb.sin() * self.attention_scaling
```

其中freqs经cat拼接后，最后一维与hidden\_size一致。

2) Q/K值做位置运算：

```python
def rotate_half(x):    """Rotates half the hidden dims of the input."""2
2
    return torch.cat((-x2, x1), dim=-1)@use_kernel_func_from_hub("rotary_pos_emb")def apply_rotary_pos_emb(q, k, cos, sin):    """Applies Rotary Position Embedding to the query and key tensors.    Args:        q (\`torch.Tensor\`): The query tensor.        k (\`torch.Tensor\`): The key tensor.        cos (\`torch.Tensor\`): The cosine part of the rotary embedding.        sin (\`torch.Tensor\`): The sine part of the rotary embedding.    Returns:        \`tuple(torch.Tensor)\` comprising of the query and key tensors rotated using the Rotary Position Embedding.    """    q_embed = (q * cos) + (rotate_half(q) * sin)    k_embed = (k * cos) + (rotate_half(k) * sin)    return q_embed, k_embed
```

其中rotate\_half用于构造与sin相乘的另一部分；对单个元素而言，计算如下：

```apache
q_i'   = q_i * cosθ - q_{i+dim/2} * sinθq_{i+dim/2}' = q_{i+dim/2} * cosθ + q_i * sinθ
```

**03**

**视觉数据（2D/3D）如何进行RoPE位置编码？**

文字是1D数据，其位置信息通过位置索引m表达。但对于图片/视频这样的数据，位置信息如何表达？

不同于文字数据中位置单调递增的特点，视觉数据中的图片位置可能出现循环，即两张不同的图片可能具有重叠的位置信息。

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

这里需要考虑：

- 区分t、h、w对位置变化的影响。例如，同一像素位置（w和h相同）在不同帧之间的差异。
- 不同视觉样本之间要可区分。例如，两个视频中t、w、h取值相同的位置仍应能区分。

如果将图片视为只有一帧的视频，视觉数据通常可用\[t, h, w\]定位，其中：

- t：视频帧索引
- h：高度方向索引
- w：宽度方向索引

进一步问题是：3D视觉数据的信息如何进行位置编码，即如何把旋转位置加进去？

最朴素的思想是把视觉数据展平为1D数据

 $\theta(m', i) = m' \cdot \theta_i = m' \cdot \left( 10000^{-\frac{2i}{d}} \right)$ 

其中为各轴上的离散索引；通常取各维的步长/尺度，使成为把三维坐标线性映射为一维索引的一种方式）

 $m' = t_i \cdot T + h_i \cdot H + w_i \cdot W$ 

这种方式比较直接，但存在一个问题：视觉数据与文字数据混合到一起时，其位置信息如何区分？

举个例子：在某一组取值下，视频位置(t=0、h=0、w=1)与文本中某一位置索引算出的可能相同，从而发生碰撞，需要额外机制区分模态或样本。

下面介绍常见的视觉RoPE编码的计算方式：

**3.1 2D位置编码**

2D位置编码一般用于图片，即不考虑时间维度T。以Qwen VL视觉塔中的2D位置编码为例 <sup>[4]</sup> ，采用宽高索引对半的分配逻辑。对图片数据而言，每个token有二维索引(h, w)，对应视觉塔中的row/col；若head\_dim=128，其分频逻辑是：

- 128先分为64对二维向量构成独立的平面；
- 前32对的频率分量由row索引决定，后32对的频率分量由col索引决定。

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

更具体一点，每个维度的归属如下：

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

然后cos = emb.cos()，sin = emb.sin()，形状都是(seq\_len, 128)。规律：

- 前32个旋转对（覆盖q\[0:32\]与q\[64:96\]）由行坐标r决定相位。
- 后32个旋转对（覆盖q\[32:64\]与q\[96:128\]）由列坐标m决定相位。

其中r与c由token在图片中的行列位置决定。row与col共用freqs；max\_len通常取max(R,C)，其中R、C分别为row与col方向的最大索引。

位置编码在Q/K上的计算代码示例如下：

1 freqs的计算：

```python
class Qwen3VLVisionRotaryEmbedding(nn.Module):    inv_freq: torch.Tensor  # fix linting for \`register_buffer\`    def __init__(self, dim: int, theta: float = 10000.0) -> None:        super().__init__()        self.dim = dim        self.theta = theta        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))        self.register_buffer("inv_freq", inv_freq, persistent=False)    def forward(self, seqlen: int) -> torch.Tensor:        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)        freqs = torch.outer(seq, self.inv_freq)        return freqs
```

2 坐标与偏移的映射关系：

```sql
offset = 0for num_frames, height, width in grid_thw_list:    merged_h, merged_w = height // merge_size, width // merge_size    block_rows = torch.arange(merged_h, device=device)  # block row indices    block_cols = torch.arange(merged_w, device=device)  # block col indices    intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets    intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets    # Compute full-resolution positions    row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]    col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]    row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)    col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)    coords = torch.stack((row_idx, col_idx), dim=-1)    if num_frames > 1:        coords = coords.repeat(num_frames, 1)    num_tokens = coords.shape[0]    pos_ids[offset : offset + num_tokens] = coords    offset += num_tokensembeddings = freq_table[pos_ids]  # lookup rotary embeddingsembeddings = embeddings.flatten(1)
```

3 Q/K值做位置运算(与1D基本相同)：

```python
def rotate_half(x):    """Rotates half the hidden dims of the input."""2
2
    return torch.cat((-x2, x1), dim=-1)def apply_rotary_pos_emb_vision(    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:    orig_q_dtype = q.dtype    orig_k_dtype = k.dtype    q, k = q.float(), k.float()    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()    q_embed = (q * cos) + (rotate_half(q) * sin)    k_embed = (k * cos) + (rotate_half(k) * sin)    q_embed = q_embed.to(orig_q_dtype)    k_embed = k_embed.to(orig_k_dtype)    return q_embed, k_embed
```

**3.3 视觉3D位置编码**

视觉3D位置编码是在2D位置编码的基础上，进一步引入了时间维度；此外，3D编码通常还需考虑多模态融合的问题。当前主流模型的3D-RoPE编码采用的并非直接展平的编码方式，这里介绍QwenVL中用到M-RoPE <sup>[5]</sup> 。

关键问题还是对嵌入维度(head\_dim)的平面分组处理。在M-RoPE中，有三个独立的角度：

- 时间维度角度：angle\_t = t \* theta\_i
- 高度维度角度：angle\_h = h \* theta\_i
- 宽度维度角度：angle\_w = w \* theta\_i

计算时，特征向量在通道维上切分为三段，分别旋转。

- 1、按mrope\_section将通道维划分为三段，每段对应时间、高度、宽度之一；
- 2、各段内部再按二维子空间（相邻两维成对）做平面旋转；
- 3、各段使用对应轴向的位置索引与频率计算旋转角度。

以Qwen2 VL为例，mrope\_section参数设置为\[16, 24, 24\]时：

- 前16个通道使用angle\_t进行旋转。
- 中间24个通道使用angle\_h进行旋转。
- 最后24个通道使用angle\_w进行旋转。

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

这样，最终的旋转位置编码就融合了三维坐标的信息。

**3.4 3D编码计算兼容文本编码**

输入为纯文本序列（没有图像/视频）时，将(t, h, w)坐标都设为m，三个维度的位置ID完全相同，都等于该token在序列中的绝对位置。此时上述计算变为：

- 前section\[0\]个通道用angle\_t = t \* theta\_i = m \* theta\_i旋转；
- 中间section\[1\]个通道用angle\_h = h \* theta\_i = m \* theta\_i旋转；
- 最后section\[2\]个通道用angle\_w = w \* theta\_i = m \* theta\_i旋转。

所有通道的旋转角度都是m \* theta\_i，这完全等价于标准的1D RoPE。

**3.5 M-RoPE频率分布不均的改进**

Qwen2 VL分段的方式有一个问题，就是频谱不均，时间T占用的都是高频通道，W占用的都是低频通道。

作为改进，Qwen2.5/Qwen3 VL采用了Interleaved-MRoPE，其核心思想如下：

在1D RoPE中，嵌入层分成了d/2组，每一对连续元素(x\[2j\], x\[2j+1\])视为一个二维子空间，j = 0, 1,..., d/2 - 1。

确定第j个子空间应使用t、h、w中的哪一个轴进行旋转

采用轮询（Round‑Robin）分配：

 $\text{axis}(j) = j \bmod 3$ 

其中定义映射：0→t，1→h，2→w。因此：

- 当j % 3 == 0时：该子空间用t轴的旋转角度angle = t \* θ\_j
- 当j % 3 == 1时：该子空间用h轴的旋转角度angle = h \* θ\_j
- 当j % 3 == 2时：该子空间用w轴的旋转角度angle = w \* θ\_j

这种方式将时间、高度、宽度三个维度的信息交错融入整个特征空间，从而使多模态模型能更全面、更高效地理解数据中的复杂时空结构。

频率的构造代码示例(参考Qwen3 VL)：

```python
def forward(self, x, position_ids):    # In contrast to other models, Qwen3VL has different position ids for the grids    # So we expand the inv_freq to shape (3, ...)    if position_ids.ndim == 2:        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)    inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)    position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"    with maybe_autocast(device_type=device_type, enabled=False):  # Force float32        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)        freqs = apply_interleaved_mrope(freqs, self.mrope_section)        emb = torch.cat((freqs, freqs), dim=-1)        cos = emb.cos() * self.attention_scaling        sin = emb.sin() * self.attention_scaling    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

其中apply\_interleaved\_mrope的作用如下：

```python
def apply_interleaved_mrope(self, freqs, mrope_section):    """Apply interleaved MRoPE to 3D rotary embeddings.    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to    interleaved [THWTHWTHW...TT], preserving frequency continuity.    args:        x: (3, bs, seq_len, head_dim // 2)        mrope_section: (3,)    returns:        x_t: (bs, seq_len, head_dim // 2)    """# just overwrite the first dimension T
    for dim, offset in enumerate((1, 2), start=1):  # H, W        length = mrope_section[dim] * 3        idx = slice(offset, length, 3)        freqs_t[..., idx] = freqs[dim, ..., idx]    return freqs_t
```

**附1**

**RoPE远程衰减特性**

RoPE远程衰减，就是随着两个 token 在序列中的相对距离增大，它们之间的注意力分数会平均而言逐渐降低。这符合我们对语言的直观认知：一个词与邻近词的联系，通常比与远处词的联系更紧密。（这一点的解释可以参考原文的说明 <sup>[1]</sup> ）

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

---

**参考:**

- \[1\]abpaper https://arxiv.org/pdf/2104.09864
- \[2\]https://github.com/CalvinXKY/InfraTech/blob/main/models/modules/rope\_principle.ipynb
- \[3\]https://kexue.fm/archives/8265
- \[4\]Qwen3VLVisionRotaryEmbedding https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3\_vl/modeling\_qwen3\_vl.py
- \[5\]https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3\_vl/modeling\_qwen3\_vl.py

想深耕AI Infra领域？欢迎访问InfraTech库！内容涵盖大模型基础、PyTorch/vLLM/SGLang框架入门、性能加速等核心方向，配套50+知识干货及适合初学者的notebook练习： **https://github.com/CalvinXKY/InfraTech**

扫码关注我们，了解更多AI Infra基础知识。

![图片](data:image/svg+xml,%3C%3Fxml version='1.0' encoding='UTF-8'%3F%3E%3Csvg width='1px' height='1px' viewBox='0 0 1 1' version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'%3E%3Ctitle%3E%3C/title%3E%3Cg stroke='none' stroke-width='1' fill='none' fill-rule='evenodd' fill-opacity='0'%3E%3Cg transform='translate(-249.000000, -126.000000)' fill='%23FFFFFF'%3E%3Crect x='249' y='126' width='1' height='1'%3E%3C/rect%3E%3C/g%3E%3C/g%3E%3C/svg%3E)

继续滑动看下一个

InfraTech

向上滑动看下一个