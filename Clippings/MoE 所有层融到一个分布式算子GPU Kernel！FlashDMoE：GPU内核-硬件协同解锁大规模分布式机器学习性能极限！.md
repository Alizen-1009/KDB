---
title: "MoE 所有层融到一个分布式算子GPU Kernel！FlashDMoE：GPU内核-硬件协同解锁大规模分布式机器学习性能极限！"
source: "https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888883&idx=1&sn=14a76e02fc523d30f613d4a2219c0fea&scene=21&poc_token=HJluTGqjkvK25Ye6TZTUfySJWRGgzSxSZP-AzAlp"
author:
  - "[[康奈尔大学]]"
published:
created: 2026-07-08
description: "这是首个将整个混合专家（MoE）算子融合到单个 GPU 内核中的系统。通过在统一内核中嵌入计算、通信和调度，与最先进系统相比，实现了 6 倍速度提升、9 倍的 GPU 利用率提升，以及分布式 MoE 场景下 5.7 倍吞吐量提升。"
tags:
  - "clippings"
---
康奈尔大学 NeuralTalk *2025年6月22日 19:08*

关键词：算子融合、性能优化、大模型、GPU、软硬件协同设计、分布式机器学习

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4X8TBuRMTIBwoaXs6g5hz2qzmxbEgXZsib2wELwKcyFhsMj5lmIyNC2kQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

本文 8140 字，阅读需 32 分钟，播客 9 分钟见文末

本文是上一篇 [所有层融合成一个 GPU kernel：消除 GPU 空闲周期！为 Llama-1B 设计低延迟的大内核！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888839&idx=1&sn=a77ed7a2987f874b3e52e253993fdf89&scene=21#wechat_redirect) 在分布式环境下的延申。

> 混合专家模型（MoE）的计算稀疏性使计算成本随模型规模增长呈亚线性增长，从而为训练大规模神经网络提供了可扩展的路径。

然而，现有实现主要由于 CPU 管理的调度、主机发起的通信和频繁的内核启动，存在 **GPU 利用率低、显著的延迟开销** 和根本 **无法利用任务局部性** 的问题。

为了克服这些限制，我们开发了 **FlashDMoE** ，这是一种 **完全驻留在 GPU 上的 MoE 算子** ，它将专家计算和跨 GPU 通信 **融合到【单个】持久的 GPU 内核** 中。FlashDMoE **实现了调度、计算和合并阶段的细粒度流水线** ，消除了启动开销并减少了空闲间隙。

与现有工作不同，FlashDMoE **避免了用于单边设备发起的跨 GPU（R）DMA 传输的批量同步集合操作** ，从而实现了有效负载效率，我们在稀疏激活层中消除了臃肿或冗余的网络有效负载。

当在具有多达 128 个专家和 16K token 序列的 MoE 模型的 8-H100 GPU 节点上进行评估时， **尽管使用 FP32 而基线使用 FP16（！？🐎）** ，FlashDMoE 与最先进的基线相比，实现了高达 **9 倍的 GPU 利用率、6 倍的延迟降低、5.7 倍的吞吐量提高和 4 倍的重叠效率。**

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XdTHuHud22W10AASUmOROpyqOIQzWOl7xKyVJq6BH1cGYQANsG9doUg/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4Xcsy3EhwX1e69yvibE4FtcWLY5TqSwmKG6QCDHI942tQOu56q58DzHbA/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)

> FlashDMoE 表明， **【有原则的】GPU 内核-硬件协同设计** 是解锁大规模分布式机器学习性能上限的关键。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XUGfkDIdyurl65bKydh4RalRKIZ2C2qg3Scia1ykSd2oDv4dtaNzicUUA/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=3)

## 本文目录

- 本文目录
- 一、引言
- MoE 中的通信开销
	- MoE 中的内核启动开销
	- 1.1 我们的贡献：单个内核中的 DMoE
- 二、动机
- 2.1 同步通信与延迟差异
	- 2.2 内核启动开销
- 计算-通信重叠与内核融合
- 三、融合 MoE 内核设计
- 3.1 基于 Actor 的模型
	- 3.2 FlashDMoE 中的 Actor 间交互
	- 3.3 对称张量布局与通信效率
- 四、实验
- 4.1 实验设置
	- 4.2 前向延迟
	- 4.3 GPU 利用率
	- 4.4 重叠效率
	- 4.5 吞吐量
	- 4.6 专家可扩展性
	- 4.7 内存开销
- 五、局限性和未来工作
- 5.1 编程复杂性
	- 5.2 FP16 支持和共享内存访问模式
	- 5.3 缺乏反向传播和训练支持
	- 5.4 多节点场景的网络缓冲区溢出
	- 5.5 未来优化方向
- 六、结论
- 参考文献
![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XgRAAUJW2zX1oUcEdTB6eP8liahibry7lOWPKbEphMibQXW2jXHfmuabvQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4)

交流加群请在 NeuralTalk 公众号后台回复：加群

## 一、引言

包括 DeepSeek-v3\[1\]、LLama4\[2\]、DBRX\[3\]和 Snowflake Arctic\[4\]在内的最先进的大型语言模型（LLMs）因其计算效率和在许多任务上的强大性能而采用了混合专家（MoE）架构。传统的 Transformer 块由一个自注意力模块和一个密集前馈网络（FFN）\[5\]组成。

相比之下，MoE 架构将这个单一的 FFN 替换为相同大小的 FFN，也称为专家（图 2（b））。一个称为门函数的可训练神经网络通过在运行时将输入 token 动态路由到选定的专家来稀疏激活这些专家。模型参数的增加（更多 FFN）提高了模型质量，而没有相应的计算成本增加。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XnRNlY6bHLOgbl8qxGJicbEMbiaZib79YbdGIeWSvyLXjx9ZVufYhNbIAg/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)

### MoE 中的通信开销

随着 MoE 模型规模的增长，GPU 内存限制使得无法在单个设备上托管所有专家。标准做法是使用专家并行性（EP）将专家分布在多个 GPU 上，这需要通过 AllToAll 或 AllGather 中的多对多通信进行 token 路由\[1,4,3,6\]。还需要另一轮上述多对多通信，以将专家处理的置换 token 恢复到序列中的原始顺序。

> 现有工作已经观察到这些通信操作占总运行时间的 68%\[7,8\]，在此期间 GPU 完全空闲，除非实现明确地与计算重叠。

**这种流水线形式难以高效实现，因为它需要异步的 GPU 驱动通信和内核融合以最大化重叠效率** 。通常，PyTorch 等框架中可用的跨 GPU 通信 API 并非此类，而是 CPU 驱动的\[9\]。

### MoE 中的内核启动开销

通信重叠的效率进一步受到从 CPU 启动许多内核的开销的限制。具体来说，现有实现\[10-13\]需要在单层传递中启动大量内核（见表 1）。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4Xcsy3EhwX1e69yvibE4FtcWLY5TqSwmKG6QCDHI942tQOu56q58DzHbA/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=6)

**频繁的内核启动会对性能产生负面影响** ：

1. 在 GPU 之间创建不确定的内核启动时间，加剧延迟差异问题；
2. 引入不必要的同步点，导致 GPU 在继续之前等待对等方或 CPU；
3. 在内核边界处引起重复的全局内存往返。

尽管 CUDA 图\[14\]可以在静态工作负载中部分缓解第一个问题，但它们与 MoE 的动态专家路由模式不兼容。

> 解决剩余问题需要新的解决方案，我们通过完整的内核融合和异步设备发起的通信在这项工作中提供了这些解决方案。

### 1.1 我们的贡献：单个内核中的 DMoE

为了克服最先进的 MoE 模型中的这些基本效率问题，我们开发了 FlashDMoE，其中我们将所有 DMoE 计算和通信任务集成到单个持久的 GPU 内核中，即一个在整个 MoE 算子期间保持活动的内核（图 3 左下）。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XcbzlibaxNhMRWLagzLtvwx6YlyVkicHhfZicF77r8p4QwyRcOMj7DDJZQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=7)

**与需要 CPU 协调的多个内核启动不同，FlashDMoE 只需要启动一个内核** ，显著减少了 CPU 的参与。在融合的内核中， **FlashDMoE 实现了一种响应式编程模型** ，以实现 **细粒度的并行性** 和 **数万 GPU 线程** 之间的 **松耦合** 、 **非阻塞执行** 。

#### 内核内块调度和 tile 并行性

> FlashDMoE 实现了 tile 级并行性，这意味着它将输入 token 矩阵划分为称为 tile 的更小、独立的单元，这些单元由块处理但由线程束（warp）管理（调度和构造）。

除了一个块之外，我们将每个线程块专门化为执行计算的处理器。此外，我们指定一个专用的操作系统（OS）块（4 个线程束）来执行以下管理任务：

1. 向处理器调度计算工作（调度器）
2. 从其他 GPU 接收的消息中解码计算任务（订阅者）。

这种设计 **允许 FlashDMoE 根据就绪状态动态地将任务分配给 GPU 块，确保在 DMoE 算子的整个生命周期内没有 GPU SM 保持空闲。** FlashDMoE 选择 tile 尺寸以最大化 GPU 算术强度，同时仍受益于高度并行性。

#### 异步和有效负载高效的通信

通过从头开始 **重新设计 MoE 算子** ，FlashDMoE **解决了传统 MoE 执行流水线中固有的基本效率问题** 。

一个值得注意的效率低下问题是通信期间的 token 填充。为了简化编程复杂性和由于集合通信 API 的对称性约束，现有实现必须将 token 有效负载零填充以匹配预定义的缓冲区大小。当 token 不对称地路由到专家时，就会发生这种情况，导致 GPU 接收到的内容远低于预期容量。

然而，这些空有效负载浪费了通信带宽，膨胀了数据传输延迟，并可能在某些实现中导致对空矩阵的不必要计算。 **FlashDMoE 通过仅向具有主动选择专家的 GPU 发送非填充 token 来引入有效负载高效的通信** ，节省了通信和计算资源。

#### 技术挑战

实现 FlashDMoE 的单内核设计需要解决几个技术挑战以实现高性能：

1. 轻量级计算依赖管理；
2. 导航最佳 SM 占用配置；
3. 实现设备内 BLAS 操作；
4. 最小化设备间和设备内的同步开销；
5. 在可用时通过统一虚拟寻址（UVA）利用 DMA 实现传输感知。

在解决这些挑战时，FlashDMoE 的设计与传统的同步 AllToAll 集合完全不同，在传统集合中，GPU 在层执行期间表现出显著的空闲时间。

- 对于 **设备发起的通信，FlashDMoE 使用 NVSHMEM\[16\]在所有 GPU 之间建立全局地址空间** ，以实现上述直接内存访问（DMA）或远程 DMA（RDMA）通信。
- 对于设备内 BLAS，FlashDMoE 通过 CUTLASS\[17\]开发了自定义的高性能 GEMM 操作。

#### 结果

我们在 **跨多个节点的多个 GPU 上评估** 了 FlashDMoE。我们的评估表明， **与最先进的实现相比，FlashDMoE 实现了 6 倍的延迟加速、9 倍的 GPU 利用率提高、4 倍的弱扩展效率和 5.7 倍的吞吐量增加。**

> 我们预计这些性能提升在 **多节点** 场景中会变得 **更好** ，其中节点间通信使用较低带宽的节点间链路（如 RDMA、InfiniBand）进行。

## 二、动机

### 2.1 同步通信与延迟差异

当前 MoE 框架中使用的 AlltoAll 或 AllGather 通信是一种同步集合操作，其完成需要所有参与 GPU 的参与。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XCU7EaW9IZynM25Y6AYkicfNicolJUACbVlt5WHm6hvzkf22exl5KicEcQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=8)

在此过程中，GPU 之间处理速度或内核调度的差异会引发延迟差异效应（图 4），这对（1）集合操作的性能和（2）端到端性能都不利，因为停滞的 GPU 在集合操作终止之前无法进行下游的依赖或独立任务。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XnricL51Tyxe2DBCwT27GNlVwg8KjCOCTDQaBthGGaDCkWXibNVNNyr1g/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=9)

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XuL6ZHxDsck5kNbEjCEnoVmFc0LMicstd7WHia0xhyTQV9JF3xUAtxbvw/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=10)

具体来说，如图 15 所示，对于在 32 个 A100 GPU 上分布式训练 13 亿参数的 GPT-3 MoE 模型，与图 15b 中的平均实际内核时间相比，我们观察到 P95 通信性能下降了 1.32 倍。由于底层硬件是针对“软件抖动”\[18\]进行了良好调优的超级计算机，这种性能下降相对温和。

然而，我们在单节点虚拟机（VM）中观察到了更严重的 p95 性能损失，达 11 倍。与之前的高性能计算工作\[19,20\]一致， **我们认为消除这种同步集合通信中固有的障碍将允许 GPU 将观察到的空闲时间重新用于下游计算，如图 4 所示。**

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XzNBc41XOvSH6lF3MSrrPRiczFD2r2OicOoFT78MgicTFaXHN6Q6drEsxQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=11)

### 2.2 内核启动开销

我们比较了 FlashDMoE 和现有基线之间的内核启动开销。 **表 1 显示了单次前向传播期间的内核启动次数** ： **FlashDMoE 仅启动一个持久内核，而基线在执行相同计算时最多启动 550 个短生命周期内核。**

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XhxBQZSPnHibVkw7JbyAn1kUibVnRDo3XqLjcz4iceib9h5hXSERvJ5gbaw/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=12)

**图 5 使用 Nsight Systems 捕获的 CUDA API 跟踪提供了可视化比较，说明了 FlashDMoE 和 DeepEP 之间的差异。** DeepEP 表现出大量的小 CUDA API 调用，各个算子之间频繁停顿，导致 GPU 空闲时间增加（图 5a）。

相比之下， **FlashDMoE 通过避免启动开销和同步间隙保持了高 GPU 利用率——实现了 93.17%的 GPU 利用率** ，而 DeepEP 为 14%。实验结果见后文。

## 计算-通信重叠与内核融合

> 为了减少分布式 DNN 训练中同步通信的开销，许多研究工作致力于增加计算和通信的重叠。

对于不含 MoE 层的通用 Transformer 模型， **许多工作\[41-49\]提供了划分和调度计算与通信操作的见解和技术** ，旨在实现更细粒度的重叠。为了解决 MoE 训练中 AllToAll 通信和专家并行性带来的挑战：

- Tutel\[50\]和 FasterMoE\[13\]将 AllToAll 与专家计算重叠。
- Lancet\[51\]还支持前向传播中的非 MoE 计算和反向传播中的权重梯度计算与 AllToAll 重叠。

尽管有这些重叠，由于带屏障的阻塞式同步集合通信，这些方法的性能在实际中受到限制。相比之下， **FlashDMoE 通过在单个内核内将异步的、设备发起的数据传输与分块计算重叠** ， **从根本上消除了这些效率问题。**

> FlashDMoE 还与 COMET\[11\]和 DeepEP\[1\]等最先进的工作不同，后者也使用这种内核发起的通信，但粒度较粗且没有完全的内核融合。

## 三、融合 MoE 内核设计

现代分布式 MoE 系统存在两个限制：

1. 关键路径上 **频繁的多对多（AlltoAll 或 AllGather）集合操作**
2. **重复内核启动** 带来的显著开销。
![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XqUle2nPHL6sOvYDs4qI3LX46S1jq0gTGuLGRkChBVFgdf2ib2ibC0ibjw/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=13)

我们在 FlashDMoE 中解决了这些问题，这是一个 **完全融合的 MoE 算子，实现为单个持久的 GPU 内核。** 与之前的方法\[11,1,10,12,21,8,22-26\]不同，FlashDMoE 是 **第一个实现完全融合的分布式 MoE 内核的解决方案** ，通过 **仅需要一次内核启动** ![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4Xcsy3EhwX1e69yvibE4FtcWLY5TqSwmKG6QCDHI942tQOu56q58DzHbA/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=14)

### 3.1 基于 Actor 的模型

FlashDMoE 的设计基于并发计算的 Actor 模型\[27-29\]。我们通过将 GPU 线程块和线程束（warp）专门化为三种不同的 Actor 角色来实现该模型：（1）处理器（§E.1），（2）订阅者（§E.3），和（3）调度器（§E.2）。

- 处理器执行计算（GEMM 和元素级操作）和 tile 通信。我们使用 CUTLASS\[17\]作为高性能 BLAS 例程的基础架构，使用 NVSHMEM\[16\]进行内核发起的通信。
- 订阅者和调度器执行管理功能。
- 调度器将计算任务分配给可用的线程块。 **我们的关键创新是使调度器既支持多线程以实现高调度吞吐量，又支持工作守恒以确保持续高 GPU SM 利用率。**
	- 订阅者从对等 GPU 解码 tile 数据包到任务描述符（§3.1）。在 GPU 的 N 个线程块中，我们将 N-1 个专门化为处理器角色，最后一个块专门化为操作系统（OS）块。在该块内，我们将三个线程束专门化为订阅者角色，一个线程束专门化为调度器角色。

这种线程块在 Actor 之间的分配是有意的： **我们的目标是使用很少的资源进行管理任务，同时保留大部分资源用于执行 MoE 计算任务。** 图 6 总结了 FlashDMoE 架构及其组成 Actor，而算法 1 给出了该系统在代码中的非常接近的翻译。

请注意， 是输入 token 矩阵； 是输出矩阵； 是专家权重的三维张量，其中 E 表示执行 GPU 的本地专家数量，H 是嵌入维度，D 是 FFN 中间维度，S 是序列长度。 是路由表数据结构，其中 表示位置 c 的 token i 调度到专家 θ，w 是合并权重（公式 2），c 是专家容量。 的元组结构是实现细节。 捕获门函数产生的亲和度分数（公式 3）。

### 3.2 FlashDMoE 中的 Actor 间交互

FlashDMoE 以 tile 为粒度分解 MoE 计算和通信，tile 是张量的静态大小分区，以实现并行执行和任务的有效重叠。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XZ9PS0yd4DNsDZnohwwv0vlHWVoAFD0mj9KpLWLh7Kld453uuPzk8HA/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=15)

每个 tile 映射到由任务描述符封装的离散工作单元。订阅者从其接收的远程 tile 数据包中解码这些任务描述符。同时，调度器接收关于可用任务的通知，并将它们调度给执行器 Actor 执行这些任务定义的计算，即前馈网络（FFN）和专家合并操作。 **上图图 7 展示了 Actor 交互链，演示了 FlashDMoE 如何强制执行 DMoE 功能依赖关系。**

### 3.3 对称张量布局与通信效率

在单个 GPU 设备内，FlashDMoE 中的 Actor 通过 GPU 的内存子系统进行通信（见图 8）。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XgYC5gGF6qXYV0DJVMJfasJicDVGION41ibvwVhbZgSK8pY1nNrtWbf9g/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=16)

具体来说， **调度器和订阅者 Actor 通过快速共享内存交换数据，而其他 Actor 对通过全局内存通信** 。对于跨多个设备的通信，FlashDMoE 使用设备发起的通信，利用单边 PGAS（分区全局地址空间）编程模型\[34\]。然而， **在 PGAS 中实现可扩展且正确的单边内存访问而不进行昂贵的同步是一个已知的挑战** \[1,35\]。我们 **通过一种可证明正确且可扩展的解决方案来解决这一挑战** ：对称张量布局 L，支持完全非阻塞的内存访问。我们将 L 定义为：

其中：P 是专家并行世界大小，R 标识通信轮次（即两轮，一轮用于 token 调度，一轮用于合并），B 是暂存缓冲区数量，E 是本地专家数量，c 是放大的专家容量（§3.2.1），H 是 token 嵌入维度。

**我们实现非阻塞通信的核心见解是时间缓冲** 。具体来说，我们为底层 token 矩阵预分配至少 倍的内存，其中 r 是依赖图中的通信轮次数，因子 2 考虑了每个通信轮次内传入和传出数据的单独缓冲区。对于 MoE 模型，我们有 。这种适度的内存使用增加消除了单边数据传输期间同步的需要。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XGr4SRhxLdqXWyStoibdAk0GWB6Ws5GnRaQ62ebRoxiaYHagGTCmWbBfw/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=17)

图 9b 说明了该对称张量布局中的单元如何被索引并用于直接内存访问（DMA）和远程 DMA（RDMA）操作。正如定理 3.1 所强调的，L 上的这种索引方案是允许完全非阻塞访问而无需同步的基础机制，因为所有访问都是无写冲突的。证明见§C。

> 定理 3.1：对称张量布局 L 是无写冲突的。

为了构造 L，我们从原始 token 缓冲区 开始，其中 S 是序列长度，H 是 token 嵌入维度。我们首先将序列维度 S 重组为三个子维度，表示专家容量（C）、本地专家槽（E）和专家并行世界大小（W），即：

在均匀专家分布的典型情况下（如图 9a 所示），我们有 和 ，其中 是模型中的专家总数。因此，token 缓冲区的大小为 。在图 9a 中，每个标记为 （）的单元是大小为 的矩阵。在先前工作\[33,11\]的基础上，我们引入了额外的时间维度 R（通信轮次）和 B（暂存缓冲区）。每个通信轮次有两个固定的暂存槽：一个用于传出 token，另一个用于传入 token。每个槽按维度 P 索引，形成形状为 的张量。因此，张量大小 通常至少是原始 token 缓冲区大小的四倍，在均匀专家分布的情况下正好大四倍。根据经验，我们发现：

#### 3.3.1 in-place 填充以提高有效负载效率

由于 MoE 调度中 token 的动态和不均匀分布\[36\]，GPU 通常接收的 token 少于其预定义的专家容量。当前的 MoE 框架\[10\]通常在计算前用空 token 填充这些缓冲区，不必要地增加了通信有效负载并降低了性能。相比之下， **我们提出 in-place 填充，直接在本地对称张量缓冲区内执行填充，从而消除了多余的网络通信。**

如图 9a 作为参考所示，每个单元 的大小根据专家容量 c 确定。我们进一步调整该容量以确保可被 tile 块大小 整除，确保处理远程 token 的处理器线程能够安全且对齐地读取内存。这种 in-place 填充策略略微增加了 L 的内存占用，如下所述：

## 四、实验

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4X8frib32hyrNPjbReId1Fw5PUefVibCs71NGteC5dFK9aZ5HpUFNObK1Q/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=18)

### 4.1 实验设置

我们在配备 8 块 NVIDIA H100 80G GPU 的服务器上进行实验，这些 GPU 通过 NVLink 互联，并配备 125GB 内存和 20 个 vCPU。

- 我们使用 PyTorch 2.6.0、CUDA 12.8 和 Ubuntu 22.04 系统。
- 所有实验均使用配置有 16 个注意力头、2048 维嵌入维度和 2048 维 FFN 中间尺寸的 MoE Transformer 模型。
- 我们对所有实验应用分布式数据并行（DDP）和专家并行（EP）。
- 我们仅在单个 MoE 层上执行前向传播，并在 32 次热身传递后测量 32 次传递的平均运行时间。
- 我们使用容量因子为 1.0 的 Top-2 路由策略。

我们将 FlashDMoE 与几个最先进的 MoE 系统进行比较：（1）Comet\[11\]，（2）FasterMoE\[13\]，（3）Megatron-CUTLASS\[37\]，以及（4）MegatronTE（使用 Transformer Engine 的 Megatron-LM）\[38\]。Comet 依赖 cudaMemcpyPeerAsync\[39\]，而 FasterMoE 和 Megatron-LM 仅使用 NCCL 进行通信。

### 4.2 前向延迟

我们首先在 4 块和 8 块 GPU 设置下，测量不同序列长度下 FlashDMoE 的前向延迟（图 10）。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XhNjce7e8ZW3qpGcAibAmXY3bM1HeYog4mEiaqlv2GxycMAr4OiaEqlJRQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=19)

**FlashDMoE 始终优于所有基线** ，在更长的序列长度下表现出尤其显著的改进。

- 在 **4 块 GPU 上** ，它在 16K token 时比 Megatron-TE 快 4.6 倍，比 FasterMoE 快 2.6 倍。
- 在 **8 块 GPU 上，增益更为明显** ，FlashDMoE 保持低延迟，而基线由于 token 缓冲区按比例增加导致通信成本上升，性能急剧下降，FlashDMoE 比基线快高达 6.4 倍。

> 这些结果凸显了 FlashDMoE 在扩展 token 吞吐量时避免其他实现所面临的通信惩罚的能力。

### 4.3 GPU 利用率

为了量化 GPU 效率，我们测量前向传播期间的流多处理器（SM）利用率（图 11）。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XaictfShNcOx666OreZCiaYlF78AlW6X1MChX4RbWFiaiaT54RKbUFRlVVw/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=20)

**FlashDMoE 实现了 93.17%的平均 SM 利用率**

- 比 FasterMoE（9.67%）高 9 倍以上
- 比 DeepEP+Megatron-LM（13.55%）高 6.8 倍 = 比 Megatron-TE（59.11%）高 4 倍
- 比 Comet（42.31%）高 2.2 倍

> 这一改进源于我们完全融合的内核架构以及计算和通信任务的细粒度流水线。通过消除因内核启动导致的空闲间隙并实现内核内任务调度，FlashDMoE 确保 SM 在执行期间始终忙于有效工作。

### 4.4 重叠效率

我们通过测量随 GPU 数量增加的弱扩展效率来评估 FlashDMoE 重叠通信和计算的程度（图 12b）。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XHfXlpkQxpBdib7GYtFlfNl8DI2giawCH77Qo4NY2pIeS1h5kVNT4iccnA/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=21)

我们注意到大多数基线无法在单个 GPU 上执行，因此我们使用 2 块 GPU 作为参考点。我们观察到 Megatron-CUTLASS 和 Megatron-TE 性能显著下降，当 GPU≥4 块时，重叠效率降至 0.5 以下。

相比之下，FlashDMoE 在 4 块和 8 块 GPU 时分别实现高达 3.88 倍和 4 倍的更高效率。图 12a 进一步说明了这种效率，因为 FlashDMoE 显示出稳定的前向延迟增长，而基线 Megatron-CUTLASS 和 Megatron-TE 经历了近似线性的延迟放大，而 FasterMoE 表现出亚线性扩展。

> 我们将这种次优性能归因于延迟差异效应和暴露的通信。

相比之下，FlashDMoE 表现出预期的均匀延迟，因为在这个弱扩展实验中每个 GPU 的工作负载是固定的。这些结果进一步证实，即使在扩展时，FlashDMoE 基于 Actor 的设计和异步数据移动也能实现近乎理想的重叠。

### 4.5 吞吐量

吞吐量以每秒百万 token 数（MTokens/s）衡量，反映端到端系统效率。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XeGFB0ffXzj1XCrQ6gG5Ef2hanv1RPGkdVY3NNjUibI45vBiaEQOjHaPw/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=22)

**如图 13 所示，FlashDMoE 与 GPU 数量呈线性扩展，在 8 块 GPU 时达到 17.7 MTokens/s。**

- 这比 FasterMoE 高 5.7 倍以上
- 比 Megatron-TE 和 Megatron-CUTLASS 高 4.9 倍。

值得注意的是， **这些结果是在 FlashDMoE 完全使用 FP32 的情况下实现的，而基线使用 FP16。**

> 这表明 FlashDMoE 的设计 **不是通过** 利用更低的精度来消除吞吐量瓶颈，而是 **通过最大化硬件利用率和消除主机驱动的低效率** 。

### 4.6 专家可扩展性

我们分析了在固定序列长度（T=16K）下，随着专家数量增加，FlashDMoE 的扩展情况。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XtJujPYgtXJ0RhkJJgmWMSDWjYuSW5CHkhMajC65dicDQhCQJsGsrIibQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=23)

**如图 14 所示，即使专家数量从 8 增加到 128，FlashDMoE 也能保持预期的低均匀延迟。相比之下，基线由于内核启动开销增加，表现出超线性延迟增长。**

在 4 块 H100 和 8 块 H100 上，当专家数量为 128 时，FlashDMoE 分别比这些基线快 4 倍和 6.6 倍。 **FlashDMoE 的有效负载高效通信和调度器驱动的内核内调度使其能够维持专家并行性，而不会产生其他系统中出现的通信和协调惩罚** 。这些结果强化了 FlashDMoE 在超稀疏 MoE 配置中的可扩展性。

### 4.7 内存开销

我们测量了 FlashDMoE 的对称张量 L 和运行时记账状态所需的 GPU 内存。内存开销主要取决于 tile 大小、专家容量（EC）和专家数量（E）。 **表 3 总结了各种配置下的内存开销，确认 FlashDMoE 保持适度且可预测的内存占用。**

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XZ1RpAepCL8azcGlJIvdvupwl5uAsJicvXNSZ69wfXdYOkGFjWPaxibkQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=24)

## 五、局限性和未来工作

### 5.1 编程复杂性

> 开发完全融合的持久内核是一项艰巨的工程任务。

尽管 FlashDMoE 证明了此类内核的可行性和优势，但其 **构建需要 GPU 架构、同步和分布式协议以及内存层次结构的深入专业知识。** 这种高进入门槛限制了其采用。

> 未来的工作可能会考虑 **编译器级抽象或领域特定语言（DSL）** 来普及这种技术。

### 5.2 FP16 支持和共享内存访问模式

尽管现代 GPU 原生支持半精度计算，但将 FlashDMoE 适配到 FP16 对于处理器的计算算子来说并非易事。具体来说，我们 **手动调整的共享内存布局并不是** 我们用于实现设备内 GEMM 的 CUTLASS Collective Mainloop 算子的 **最有效模板参数** 。 **这种次优配置降低了内存吞吐量，如下图所示。**

克服 Ampere 及以下 GPU 的这一问题需要对最佳布局进行仔细研究，但对于 Hopper 及以上 GPU，我们预计在未来的改进中使用 CUTLASS 提供的构建器接口。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XgCjWdkRowO2cibHZWibLPf5ib7upwAfe1aVHTY71gTThgib1jiaRuI2PJHA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=25)

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4X1oX3kSKOWbW3cOS1H2EtYfEmwgy3c7bm2ibXXfTqicmrS92U1Nxqe6Lg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=26)

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XyjuzpJxQ2ECpKuVBy1JWYzbPCHl4jTW0eQGgGdmcxCce1MSCXaMbvQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=27)

### 5.3 缺乏反向传播和训练支持

虽然这项工作侧重于推理，但启用训练需要将反向计算和梯度通信融合到内核中。支持这一点需要对内存记账和任务描述符定义进行重大更改。 **尽管如此，这仍然是将该系统扩展到完全支持端到端训练的一个令人兴奋的方向。**

### 5.4 多节点场景的网络缓冲区溢出

在多节点实验中，我们观察到当 token 数>2048 时，应用程序由于未接收到预期消息而无法终止。我们假设这种故障是由于网络硬件层的缓冲区溢出造成的，这在生成许多大消息的应用程序中很常见\[18\]，如我们的系统。我们注意到这种故障可以通过调整硬件配置\[52\]来解决，但我们认为这种探索与本工作无关。

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4Xc4xYibpyW2Q1PKckL77n0IWNsjlLCTk30v1gq3a0NAkaMtickTY3HiaDA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=28)

### 5.5 未来优化方向

未来的工作将集中在以下几个方面：

1. **优化 FP16 的共享内存布局** ，以提高内存吞吐量和计算效率。
2. 开发 **编译器辅助的内核融合** 技术，降低编程复杂性。
3. 扩展对反向传播的支持，实现端到端的 MoE 训练。
4. 改进多节点通信协议，避免缓冲区溢出并提升扩展性能。

## 六、结论

> 本文介绍了 FlashDMoE，这是首个将整个混合专家（MoE）算子融合到单个持久化 GPU 内核中的系统。

我们证明， **主流 MoE 实现存在两个关键效率问题** ：

1. 由 CPU 管理的同步通信导致互联利用率低下；
2. 通过多个 GPU 内核的碎片化执行引入开销和同步延迟。

相比之下， **FlashDMoE 通过在统一内核中嵌入计算、通信和调度** ，采用了 **GPU 自主执行模型** 。它利用 Actor 风格的并发机制、线程束（warp）专用化和异步（R）DMA 传输，实现了 **细粒度的通信-计算重叠** 。

我们的评估表明，与最先进的系统相比，FlashDMoE 实现了 6 倍的速度提升、9 倍的 GPU 利用率提升，以及分布式 MoE 场景下 5.7 倍的吞吐量提升。

FlashDMoE 挑战了分布式深度学习中的主流执行范式，并为构建未来的 GPU 原生系统提供了一个引人注目的模板。尽管存在编程复杂性和缺乏 FP16 支持等限制，但这项工作为内核内分布式计算的新时代奠定了基础。

> 未来的系统可能会在此基础上 **实现整个训练流水线的内核融合** ，推动从 CPU 编排到完全自主 GPU 执行的设计转变。

## 参考文献

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XiajyDXRfRgR59dontd1fJLwnW3rYm4TC2LFfm8ROluEEQabQd4oE9MA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=29)

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4Xf91NnicEgTDH3sIgxLvIHv0V9Juf4tXyhw7p1wyUaqicEKI8O8ibo2DEg/640?wx_fmt=png&from=appmsg#imgIndex=30)

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XDBeRXhqxichqtarnn1A7AiaomqaDib4yxXPkLicNBicia2FG0R6t9Zx3VxXg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=31)

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4Xa1cr61MvzGaEYzKeg7ZzeTGAaIMMBicP2EfCBblXwoPxDhAVULibdWMg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=32)

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XwaZChlC6I1G9QstHeD1icia35BkXcCVtyRfYst57ic1XPjoNMhdrnrTgQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=33)

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4Xia1udHR5MibXagQbLxRwgicbSK0Ozn4pMBSzOQ0XVsvgLFxgwld6YicTDQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=34)

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XZenpQbjBKVauVm62fNqwflwKQYRAwYaXJLhyYdmG9xxcIY52DIaL1Q/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=35)

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4Xm3icuk6znLia2mvPSibmem2KRWiclVNkI5f1fhx7xfPA674pGoKv3qmcRg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=36)

![图片](https://mmbiz.qpic.cn/mmbiz_png/JiaD3ALapU7D26jr9v76BbmcPl4dzFz4XqKHaJQ4L1YS9unwHb3kCSxY356GJ895Vsosam7Tqk0z1oao7aNqLNw/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=37)

交流加群请在 NeuralTalk 公众号后台回复：加群

,9分钟

GPU · 目录

文中音频