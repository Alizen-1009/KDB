---
title: "超越 vLLM 与 SGLang！Event Tensor：以动态 MegaKernel 消除重编译，解锁GPU核间通信-计算重叠"
source: "https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447900086&idx=1&sn=fb8be688929f686853cc12c815e759ae&scene=21&poc_token=HBFuTGqjhuhjA8uDzfzJVpqBJ7pdVgIHMltq4k33"
author:
  - "[[CMU,NVIDIA,UCB等]]"
published:
created: 2026-07-07
description: "当 CUDA Graph 还在为动态 Shape 焦头烂额时，Event Tensor 已经用一套编译器抽象将整个 LLM 解码过程塞进了一个“永不退出”的内核。1.48 倍推理加速、3.5 倍预热缩减的背后，是一场关于 GPU 调度模型的范式转移。"
tags:
  - "clippings"
---
CMU,NVIDIA,UCB等 NeuralTalk *2026年4月18日 19:00*

关键词： **MegaKernel** 、 ***动态形状调度*** 、LLM 推理、 **编译器抽象** 、细粒度同步、形状泛化

,22分钟

> 在现代 GPU 工作负载中，尤其是大语言模型（LLM）推理， **内核启动开销和粗粒度同步已成为限制端到端效率的主要瓶颈。**

尽管最近的“超级内核”（Megakernel）技术通过将多个算子融合到单个持久化内核中来消除启动间隙并暴露内核间并行性，但它们在实际工作负载中处理动态形状和数据依赖计算时却显得力不从心。

当你在 vLLM 或 SGLang 上跑 Qwen3-32B 时，是否注意过服务启动后那长达 **2 分钟甚至 10 分钟** 的“预热期”？或者每当输入 Shape 发生变化时，为什么 CUDA Graph 需要反复重新捕获？

![表 1 不同 graph 捕获方法下 Qwen3-32B 模型服务的预热时间。该表对比 SGLang、vLLM（JIT）与 ETC（AOT）的预热耗时和 JIT 图捕获次数，ETC 仅 35 秒且 0 次捕获，远优于传统方案。核心原因是 ETC 基于事件张量的符号形状实现 AOT 编译，无需运行时重复捕获 CUDA 图；传统方案需为不同形状捕获大量静态图，产生高额预热开销。事件张量的动态形状抽象彻底消除 JIT 与图捕获开销，大幅降低 LLM 服务部署的预热成本。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHib6Rz6YPG3UbGVZjBhLsMYgknibQp3Sbfz7TQcY5CseXpbwleggCtrKjtWgGmB3FWp0xOPQKricEgrGLIo1BnQUchicIul5DKwQZY/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

表 1 不同 graph 捕获方法下 Qwen3-32B 模型服务的预热时间。该表对比 SGLang、vLLM（JIT）与 ETC（AOT）的预热耗时和 JIT 图捕获次数，ETC 仅 35 秒且 0 次捕获，远优于传统方案。核心原因是 ETC 基于事件张量的符号形状实现 AOT 编译，无需运行时重复捕获 CUDA 图；传统方案需为不同形状捕获大量静态图，产生高额预热开销。事件张量的动态形状抽象彻底消除 JIT 与图捕获开销，大幅降低 LLM 服务部署的预热成本。

这些看似无关紧要的工程细节，实际上暴露了当前 LLM 推理基础设施在 GPU 调度模型上的一场慢性危机——而《Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel》这篇论文，正是对这一危机的正面宣战。

![图片](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHicu19T4K513jmOibh03NG2aHAo8lxGRpUYnzO1wQvk5JFypQ1kcwGzMdLBDHGq7o9tXn0Z6QTK9AV2r7QXicP5SgOlKOfSl4jecY/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1)

- **Event Tensor: A Unified Abstraction For Compiling Dynamic Megakernel**
- https://arxiv.org/pdf/2604.13327
- 1.2 万字，阅读 40 分钟， **播客 22 分钟**

,22分钟

MegaKernel 相关推荐

> 该研究直指当前 LLM 推理系统的两大死穴：

- **内核启动开销** ：每次 ，而最快的内核执行仅需
- **内核边界强制的隐式同步** ：明明可以流水线并行，却硬生生被切成了串行

更致命的是，随着 MoE 架构和连续批处理（Continuous Batching）的普及，计算图本身变得动态且数据依赖—— ***CUDA Graph 那一套“捕获-重放”的静态范式彻底失效了。***

![图 2：Event Tensor 抽象总览。计算图被划分为分块算子任务，Event Tensor 以一等公民的符号形状对象捕获任务间细粒度依赖，处理 LLM 服务的两类核心动态性：形状动态性与数据依赖动态性。符号维度适配动态批次等形状变化，运行时索引映射适配 MoE 等数据依赖路由。它将同步事件组织为多维张量，让编译器用统一逻辑处理两类动态性，解决传统巨型内核无法适配真实场景动态计算的痛点，支撑 ETC（事件张量编译器）编译器实现静态 / 动态调度一体化。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHibpxXD6ON6DOBh5njicgSaXR7rmbbGfYQwt4iasp9nQZrdQ7hKWICg05mQnJic7pXHqkXlDtpws9Nic1ocBjuhrahe4vYd6ObtsYU0/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)

图 2：Event Tensor 抽象总览。计算图被划分为分块算子任务，Event Tensor 以一等公民的符号形状对象捕获任务间细粒度依赖，处理 LLM 服务的两类核心动态性：形状动态性与数据依赖动态性。符号维度适配动态批次等形状变化，运行时索引映射适配 MoE 等数据依赖路由。它将同步事件组织为多维张量，让编译器用统一逻辑处理两类动态性，解决传统巨型内核无法适配真实场景动态计算的痛点，支撑 ETC（事件张量编译器）编译器实现静态 / 动态调度一体化。

Event Tensor 的核心洞察堪称优雅： **既然同步事件本身可以看作任务完成状态的集合，为什么不将它们组织成与数据张量同构的多维数组，享受符号 Shape 和编译器优化的全套红利？** 这正是该工作的本质贡献—— ***将细粒度同步原语提升为编译器 IR 中的一等公民，用一个符号化的“事件张量”模板，同时驯服了 Shape 动态性（通过符号维度）和数据动态性（通过运行时索引表达式）。***

> 评估结果证实了这一范式的威力：

- 在 MoE 层推理上， ***ETC（Event Tensor Compiler，事件张量编译器）*** 生成的 **超级内核相较 Triton 和 FlashInfer 基线最高提速 ；**
- 在低 Batch 解码场景， **端到端延迟比 vLLM 降低 ；**
- 而得益于真正的 AOT 编译，系统预热时间 **从 vLLM 的 锐减至 —— *这不是百分比的优化，而是工作流程的重构。***

本文将从 Event Tensor 的语言构造、编译器调度变换到运行时极简设计，逐层拆解这一可能重新定义 LLM 推理编译器范式的技术方案。

![图16. 单块B200上Qwen-30BA3B的原始内核相对性能结果。该柱状图对比了 vLLM、SGLang 与 ETC（ours）在 Qwen3-30B-A3B 服务任务中不同批大小下的相对性能，以 vLLM 为基线（1.0）。ETC 在所有批大小下均表现最优，尤其在小批（如 1、8）场景下性能提升显著，最高较基线提升约 48%；随批大小增大，优势虽收窄，但仍保持领先，展现出在不同负载下的稳定性能优势。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHicnHiauJQcneYzaJoj7a2AD93mK37NAOrelHaGmIzLaUiaiaPic2ic73gLStsIx9o9LnvL2XmiaoUmSpWTAtvtFvkVX14VW6YCpdBFHA/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=3)

图16. 单块B200上Qwen-30BA3B的原始内核相对性能结果。该柱状图对比了 vLLM、SGLang 与 ETC（ours）在 Qwen3-30B-A3B 服务任务中不同批大小下的相对性能，以 vLLM 为基线（1.0）。ETC 在所有批大小下均表现最优，尤其在小批（如 1、8）场景下性能提升显著，最高较基线提升约 48%；随批大小增大，优势虽收窄，但仍保持领先，展现出在不同负载下的稳定性能优势。

![图17. 单块B200上Qwen-32B的原始内核相对性能结果。该柱状图对比了TP=1配置下，vLLM、SGLang与ETC（ours）在Qwen3-32B服务任务中的相对性能，以vLLM为基线（1.0）。ETC在所有批大小下均保持领先，尤其在小批场景优势显著；SGLang在大批（64、128）下性能明显下降，而ETC始终稳定优于两者，展现出单卡部署下的性能优势与负载适应性。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchH8spmCFe33oK24rku9ZST3QrTKAvwLIe6rzqGyIEYvPjJFA0PwZ6Ar4Ptk8ApiadkDJCx2BiaT6SP2yVE5wBicFHqVaQRsAFrqQsc/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4)

图17. 单块B200上Qwen-32B的原始内核相对性能结果。该柱状图对比了TP=1配置下，vLLM、SGLang与ETC（ours）在Qwen3-32B服务任务中的相对性能，以vLLM为基线（1.0）。ETC在所有批大小下均保持领先，尤其在小批场景优势显著；SGLang在大批（64、128）下性能明显下降，而ETC始终稳定优于两者，展现出单卡部署下的性能优势与负载适应性。

## 本文目录

- 一、内核的“柏林墙”：为什么你的 GPU 有一半时间在空转
- 1.1 两道枷锁：启动开销与隐式同步
	- 1.2 CUDA Graph 的权宜之计与致命短板
	- 1.3 超级内核的“动态困局”
- 二、Event Tensor 的核心抽象：当同步原语成为一等公民
- 2.1 语言构造的三板斧
	- 2.2 驯服 Shape 动态性：从静态图到符号模板
	- 2.3 拥抱数据依赖：当 MoE 遇上动态事件
- 三、编译器的魔法：将事件图熔铸为高效内核
- 3.1 静态调度：为可预测负载定制的零开销流水线
	- 3.2 动态调度：为 MoE 量身定制的负载均衡器
	- 3.3 极简运行时：将“操作系统”编译进内核
	- 3.4 端到端编译流程一览
- 四、性能的铁证：从通信重叠到 MoE 霸权
- 4.1 通信与计算：打破 Tensor 并行的气泡
	- 4.2 MoE 层：动态性的终极考验
	- 4.3 低 Batch 解码：重新定义实时交互的延迟下限
	- 4.4 预热开销：AOT 对 JIT 的降维打击
	- 4.5 动静之辩：两种调度哲学的实证对决
- 五、相关工作：在巨人的肩膀上眺望
- 5.1 深度学习编译器与算子融合
	- 5.2 LLM 推理服务系统
	- 5.3 任务并行模型与 GPU 超级内核
- 六、结论与展望：迈向完全动态的 GPU 计算时代
- 6.1 结论总结
	- 6.2 进阶分析
	- 6.3 未来工作
![图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/GxIgp4icchHibFtrHeibsGP9O2wZT5VvmUhD00tUHx2FFZX2ibFHxRFaUib4cCYMfVE72XzBGeNGrg30hdHpLw8s2qXTN9lbrCex5hDMbMdaYia1E/640?wx_fmt=jpeg&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)

## 一、内核的“柏林墙”：为什么你的 GPU 有一半时间在空转

> 现代机器学习部署的核心矛盾已从“算得不够快”转向“调度得不够聪明”。当 GPU 的单次内核执行时间已经压缩到微秒级时，传统的内核边界和 CPU 侧调度模型反而成了最大的性能黑洞。

### 1.1 两道枷锁：启动开销与隐式同步

任何优化 LLM 推理性能的工程师最终都会撞上两面墙。

**第一面墙是内核启动开销** 。在当前主流的 PyTorch 等框架中， ***GPU 内核由 CPU 串行发射，如下图 1 左上。***

![图 1：不同的 GPU 调度模型。逐个内核（Kernel-by-kernel）和 CUDA Graph 调度模型强制执行粗粒度的顺序执行。超级内核（Megakernel）将操作分解为更小的任务，实现了内核间并行性。从图中可以清晰看到，传统模型下 SMs（流式多处理器）大量时间处于空闲等待状态，而超级内核模型通过任务级流水线将 SMs 的利用率推向极致。这正是 Event Tensor 要解决的核心问题域。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHibLWiboic6xoKkfIU9ObibAWibLibIOJJGicpq1zrkA6KUM2nqsicb6fKymPVgsPBkhItjfec8Jk1PSGiaYllu6uuKu3dp8MIdGfQ2OnRM/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=6)

图 1：不同的 GPU 调度模型。逐个内核（Kernel-by-kernel）和 CUDA Graph 调度模型强制执行粗粒度的顺序执行。超级内核（Megakernel）将操作分解为更小的任务，实现了内核间并行性。从图中可以清晰看到，传统模型下 SMs（流式多处理器）大量时间处于空闲等待状态，而超级内核模型通过任务级流水线将 SMs 的利用率推向极致。这正是 Event Tensor 要解决的核心问题域。

每次发射都需要穿越 PCIe 总线、更新硬件状态、设置参数——这些操作累积起来约 。 **问题在于，一个 LLM 解码步骤可能涉及数百个细粒度算子，而其中最快的 GEMM 或 Norm 内核可能只需 就能跑完。 *这意味着，GPU 有超过一半的时间不是在计算，而是在等待 CPU“发号施令”。***

**第二面墙更为隐蔽，也更难逾越——内核边界强制执行的隐式同步** 。

在传统的 kernel-by-kernel 执行模型中，每个内核必须完全结束后，下一个内核才能开始。 ***但真实的计算图依赖关系远非如此粗暴：后续算子往往只依赖于前序算子的部分输出。理论上，这些算子完全可以流水线执行或重叠执行，以提升吞吐量*** 。然而，内核边界就像一道柏林墙，彻底阻断了这种细粒度的内核间并行性。

### 1.2 CUDA Graph 的权宜之计与致命短板

> 针对第一面墙，业界拿出了 CUDA Graph。它的思路很简单： **先把一连串内核操作录制下来，形成一个静态的执行图谱，后续推理时直接重放这个图谱，从而将多次内核启动的开销压缩到单次** 。从效果上看，CUDA Graph 确实大幅降低了启动延迟。

但它有一个致命的前提假设：计算图必须是静态的。这意味着， ***所有张量的 Shape、控制流分支、甚至指针地址都必须在录制时确定*** 。 **一旦 Shape 发生变化（例如 Batch Size 从 32 变成 33），整个 Graph 必须重新捕获** 。这不仅引入了难以忍受的运行时抖动，在面对 MoE 这种数据依赖的控制流时更是直接“缴械投降”——你无法为一个依赖于输入 Token 内容的专家路由网络预先录制静态图。

### 1.3 超级内核的“动态困局”

> 既然问题出在“内核太多”上，最激进的思路就是：为什么不把所有算子都塞进一个内核里，让它一直运行，永不退出？这就是“超级内核”（Megakernel）的思想。

通过将每个算子分解为 CTA（Cooperative Thread Array，即 SM 上的线程块）级别的 Tile 任务，所有任务在一个持久化内核中执行，任务间的同步通过轻量级的 GPU 端信号量完成。这样既消灭了 CPU 侧的内核启动，又因为任务颗粒度足够细，可以暴露算子间的流水线并行。

理想很丰满，现实却遭遇两大挑战：

1. **动态 Shape 难题：超级内核在编译时必须确定任务网格的大小** 。如果 Batch Size 变化了， ***传统方案要么重新编译（这在生产环境中是灾难性的），要么为每一种可能的 Shape 预编译一个版本*** （面对连续批处理的海量 Shape 组合，这根本不现实）。
2. **数据依赖难题** ：在 MoE 层中，Token 如何分配给专家是由前一层的输出动态决定的。 ***哪些 Tile 任务依赖于哪些事件，在编译时是未知的。***

Event Tensor 正是为解决这两个“动态困局”而生。它的 ***核心思想是：将同步事件本身建模为一种支持符号 Shape 的“张量”* 。既然数据张量可以通过符号维度 `[B, L, H]` 来泛化所有 Shape，为什么事件依赖图不行？**

## 二、Event Tensor 的核心抽象：当同步原语成为一等公民

> Event Tensor 的本质，是将传统并行编程中散落各处的、手工管理的信号量，提升到编译器 IR 的层面，赋予其张量的代数结构与符号能力。这不是简单的“把事件放进数组”，而是为依赖图赋予了模板化的编译时优化空间。

### 2.1 语言构造的三板斧

在深入编译器魔法之前，我们需要理解 Event Tensor 程序的基本构成。它并非另起炉灶，而是对现有算子编程模型的一次精巧扩展。

- **Device Function（设备函数）** ：定义了在 GPU 上并行启动的 Tile 任务网格。每个任务由一个多维坐标标识，运行在一个 SM 上。它可以包含 Warp 特化、Tensor Core 调用等高级特性。
- **Event Tensor（事件张量）** ：一个多维结构，其元素代表“一组任务的完成状态”。每个事件元素维护一个初始等待计数器（Wait Count），记录它依赖的任务数量。它支持两个核心操作： `E[i].notify()` （信号完成，计数器减 1）和 `E[i].wait()` （阻塞直至计数器归零）。在动态调度模式下，它还能主动触发依赖它的任务。
- **Graph Function（图函数）** ：描述整体计算图，包含 `call_device` 调用。与传统计算图的本质区别在于，它不仅包含数据张量，还显式包含 Event Tensor。每次设备函数启动都可以标注输入/输出依赖，并通过坐标映射精确描述任务间的同步关系。

> 图 3 展示了一个经典的分裂-K 归约（Split-K Reduction）示例。

![图 3 为一个基于Event Tensor的两级并行求和示例程序，清晰展示了生产者任务如何通过Event Tensor 与消费者任务建立声明式依赖关系。左侧代码中，生产者任务以二维坐标为参数，对输入张量的子块执行局部求和，生成形状为的中间张量；消费者任务以一维坐标为参数，对中对应子块执行全局求和，最终输出形状为的张量。主程序中声明的是调度核心：的通过的类einsum映射，将同一下4个任务的完成事件聚合到的第个位置，匹配的等待条件；的再通过映射，触发对应消费者任务。右侧示意图直观呈现了这一过程：中间张量的个tile经并行处理后，事件信号通过完成聚合与分发，驱动执行。这种声明式依赖描述让编译器能显式感知任务间的并行关系，精准优化任务级并行度，避免不必要的全局同步，实现高效流水线调度。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHicNPhzTAxziadBRzDfbicicSUzj42mLDaicppVibMtTJHrl0w0hc1tkGMXckm6PwPNpyia5lk23w73icaErNDdcFKVKo2WZeibiaqz1cTzA/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=7)

图 3 为一个基于Event Tensor的两级并行求和示例程序，清晰展示了生产者任务如何通过Event Tensor 与消费者任务建立声明式依赖关系。左侧代码中，生产者任务以二维坐标为参数，对输入张量的子块执行局部求和，生成形状为的中间张量；消费者任务以一维坐标为参数，对中对应子块执行全局求和，最终输出形状为的张量。主程序中声明的是调度核心：的通过的类einsum映射，将同一下4个任务的完成事件聚合到的第个位置，匹配的等待条件；的再通过映射，触发对应消费者任务。右侧示意图直观呈现了这一过程：中间张量的个tile经并行处理后，事件信号通过完成聚合与分发，驱动执行。这种声明式依赖描述让编译器能显式感知任务间的并行关系，精准优化任务级并行度，避免不必要的全局同步，实现高效流水线调度。

我们要计算 。传统做法是，第一阶段所有部分和任务全部完成后，第二阶段才开始。 ***但通过引入形状为 `[n]` 的 Event Tensor ，我们建立起细粒度依赖：。* 这意味着，只要某一行 的所有部分和任务完成，该行的最终归约任务就可以立即开始，无需等待其他行。**

### 2.2 驯服 Shape 动态性：从静态图到符号模板

> Event Tensor 最关键的突破，在于它让依赖图也具备了“符号 Shape”的能力。传统 CUDA Graph 是运行时为具体 Shape 实例化的一张固定依赖图。而 Event Tensor 图则是一个 **符号模板** 。

![这张图展示了基于Event Tensor的多头注意力依赖调度流程。输入的Q RoPE、K RoPE张量（维度含批次B与头数H=2）通过映射，V Proj张量通过映射，将各任务完成事件聚合到形状为的Event Tensor中。该ETensor在运行时动态实例化，为后续注意力任务提供依赖同步：当批次B=1时，每个头h=0、h=1各生成一个依赖单元，等待对应Q、K、V的完成信号；当批次B=2时，则为每个批次内的每个头分别生成独立依赖单元，实现批次与头维度的细粒度同步。这种声明式依赖映射让编译器可根据批次大小动态实例化依赖关系，在运行时精准控制注意力任务的触发条件，无需硬编码同步逻辑，同时支持批处理与多头维度的并行调度优化。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHibypfmiaoRMEIPXVackSzo98OCHzlcS34Geu60S4huZrI8dLU1XM8NDRdNCibfurkYuQ1aFSA4ySPMTlgickS4qxKB0TSibMMicm7c8/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=8)

这张图展示了基于Event Tensor的多头注意力依赖调度流程。输入的Q RoPE、K RoPE张量（维度含批次B与头数H=2）通过映射，V Proj张量通过映射，将各任务完成事件聚合到形状为的Event Tensor中。该ETensor在运行时动态实例化，为后续注意力任务提供依赖同步：当批次B=1时，每个头h=0、h=1各生成一个依赖单元，等待对应Q、K、V的完成信号；当批次B=2时，则为每个批次内的每个头分别生成独立依赖单元，实现批次与头维度的细粒度同步。这种声明式依赖映射让编译器可根据批次大小动态实例化依赖关系，在运行时精准控制注意力任务的触发条件，无需硬编码同步逻辑，同时支持批处理与多头维度的并行调度优化。

如图 4 所示，一个 Event Tensor 的维度可以是符号变量（如 Batch Size ）。 **这个符号化的图在编译时定义了一种“依赖关系的生成规则”** ，而不是具体的依赖边。在运行时，当具体的 Shape（如 或 ）传入时，这套规则会动态实例化出对应规模的依赖图—— 或 的网格。

**关键在于，符号维度作为通用模板，这一切发生在不重新编译、不重新捕获 Graph** 的前提下，彻底摆脱 CUDA Graph 需重复捕获的缺陷，可实例化不同批次大小的任务图。实验表明该设计 ***在低批次动态推理场景下，保持高性能同时降低引擎预热开销最高 3.5 倍，证明符号抽象让动态巨型内核实现预编译*** ，消除运行时编译开销。

### 2.3 拥抱数据依赖：当 MoE 遇上动态事件

> 如果说 Shape 动态性还只是“规模”的变化，那么 MoE 带来的数据依赖则是“拓扑结构”的变化。 **下图 5 对比了常规工作负载与 MoE 工作负载的本质差异。**

![图 5：Event Tensor 处理数据依赖动态性。该图对比了基于Event Tensor的两种任务图模型：(a)常规任务图与(b)数据依赖任务图。(a)中生产者任务通过固定映射更新Event Tensor（ETensor），ETensor以静态计数器触发映射的消费者任务，依赖关系预先确定。(b)以MoE路由为例，Token生产者任务通过数据依赖映射更新ETensor；ETensor结合运行时与的值，通过动态映射触发GroupGeMM消费者任务，实现数据驱动的动态依赖同步。图例中，蓝色方块为任务Tile，黄色方块为运行时值，红色单元为带计数器的ETensor，箭头区分常规/数据依赖的ETensor更新与任务触发逻辑，直观展现了ETensor同时支持静态与动态任务调度的能力。实验中该设计让 MoE 层融合为单个巨型内核，性能比专用库最高提升 1.23 倍，证明其能高效处理不规则任务图，突破传统方案无法适配数据依赖计算的局限。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHicEBjmpib7rBJrPqPJU5IY8dD65jUWtma6a5Pb3R6eGEg1dia2FxT3iaz8dBMgLynSQ2kcmhK87dsY3Ctybl2uoDic19NWrRwz7ibx4/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=9)

图 5：Event Tensor 处理数据依赖动态性。该图对比了基于Event Tensor的两种任务图模型：(a)常规任务图与(b)数据依赖任务图。(a)中生产者任务通过固定映射更新Event Tensor（ETensor），ETensor以静态计数器触发映射的消费者任务，依赖关系预先确定。(b)以MoE路由为例，Token生产者任务通过数据依赖映射更新ETensor；ETensor结合运行时与的值，通过动态映射触发GroupGeMM消费者任务，实现数据驱动的动态依赖同步。图例中，蓝色方块为任务Tile，黄色方块为运行时值，红色单元为带计数器的ETensor，箭头区分常规/数据依赖的ETensor更新与任务触发逻辑，直观展现了ETensor同时支持静态与动态任务调度的能力。实验中该设计让 MoE 层融合为单个巨型内核，性能比专用库最高提升 1.23 倍，证明其能高效处理不规则任务图，突破传统方案无法适配数据依赖计算的局限。

在 MoE 中，Token 到专家的路由结果存储在运行时张量 `topk` 中。哪些 GroupGEMM Tile 处理哪些 Token、需要触发多少个后续 Tile——这些信息只有在 `topk` 计算完成后才能确定。

> 但是，Event Tensor 通过两个核心机制应对这一挑战：

1. **数据依赖的事件更新（Data-Dependent Event Update）： *事件计数器的初始值不再是编译时常量，而是根据运行时 `topk` 结果动态计算*** ，例如，每个专家的事件计数器初始化为路由到该专家的 Token 数量。
2. **数据依赖的任务触发（Data-Dependent Task Triggering） *：一个事件可以触发数量不等的消费者任务*** 。如上面图 5b 所示，张量 `exp_indptr` 存储了每个专家需要触发的 GroupGEMM Tile 的起止索引。专家 的事件会触发范围在 `(exp_indptr[i], exp_indptr[i+1])` 内的所有 Tile。

## 三、编译器的魔法：将事件图熔铸为高效内核

抽象定义了“是什么”，编译器则决定了“怎么做”。

ETC 的核心竞争力在于，它 **提供了一套从 Event Tensor 抽象到具体调度策略的自动化变换流水线** 。开发者只需描述任务与依赖，编译器负责选择静态或动态调度，并生成极简的运行时支撑代码。

### 3.1 静态调度：为可预测负载定制的零开销流水线

> 静态调度的哲学是“一切尽在掌握”。 ***它适用于那些 Tile 执行时间相对均匀、依赖模式固定的工作负载*** ，例如密集模型的 MLP 层或通信模式固定的 All-Gather + GEMM 融合。

![算法 1：ETC 静态调度转换算法。该算法以包含事件张量依赖的分块级数据流图模块为输入，通过复制原模块、生成静态调度方案、创建持久化内核、将预计算调度存入全局内存、遍历任务网格并添加调度、分块、输入等待与输出通知逻辑，最终输出融合静态调度巨型内核的更新模块，其核心是提前为 GPU 的每个流式多处理器（SM）分配固定任务队列，依托计数器型信号量与事件触发等待实现细粒度同步，将多算子融合为单持久内核，彻底消除内核启动开销与边界粗同步，适配稠密 LLM 等规则计算场景，调度开销极低，在低批量推理中可显著降低服务延迟。下面是具体的步骤](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchH9HC7qw1qyLZQw17ib5Qrosl5pQat9XWEldhyBtDrzUZxKZ18sRfj02xTvKoq1gvEiatP2hINaOZEvmgpoVjkJ6x3vhnwXX9UoRg/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=10)

算法 1：ETC 静态调度转换算法。该算法以包含事件张量依赖的分块级数据流图模块为输入，通过复制原模块、生成静态调度方案、创建持久化内核、将预计算调度存入全局内存、遍历任务网格并添加调度、分块、输入等待与输出通知逻辑，最终输出融合静态调度巨型内核的更新模块，其核心是提前为 GPU 的每个流式多处理器（SM）分配固定任务队列，依托计数器型信号量与事件触发等待实现细粒度同步，将多算子融合为单持久内核，彻底消除内核启动开销与边界粗同步，适配稠密 LLM 等规则计算场景，调度开销极低，在低批量推理中可显著降低服务延迟。下面是具体的步骤

ETC 的静态调度变换遵循三步走（见算法 1）：

1. **构建每 SM 执行队列** ：编译器 ***在 Host 端根据任务图和 Shape 信息，预先计算出每个 SM 应该执行的任务序列*** 。这类似于提前排好一张精确到每个工位的“生产排期表”。
2. **生成持久化主循环** ：生成一个“永不退出”的 GPU 内核。每个 SM 在这个内核中循环，从自己的私有队列中取出任务并执行。
3. **降低 Event Tensor 依赖** ： ***将高层的 `out_edges` 和 `in_edges` 注解，具体化为 `notify()` 和 `wait()` 调用*** 。在 GEMM + Reduce-Scatter 的例子中如下图 6，编译器自动在 GEMM Tile 末尾插入 `notify()` ，在 Reduce-Scatter Tile 开头插入 `wait()` 。
![图 6 静态调度变换前后 GEMM 与 Reduce-Scatter 操作。两个独立的设备函数被融合为单个持久化函数，并通过在事件张量上执行显式通知与等待调用，来协调依赖关系。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHic6MGy9DyK2CJC7aj6D4qgKErvANU2vEOO32SN9EIINDJV1vib7mj9fDia0Sshr6iaOKbmCMApnPZWHxEweyDNdeS1veEuI4tBQyU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=11)

图 6 静态调度变换前后 GEMM 与 Reduce-Scatter 操作。两个独立的设备函数被融合为单个持久化函数，并通过在事件张量上执行显式通知与等待调用，来协调依赖关系。

> 下面图 7 生动地展示了这一过程的执行时序。假设每个 Reduce-Scatter Tile 依赖于两个 GEMM Tile（计数器初值为 2）。

![图 7：静态融合的GEMM（MM）+ Reduce-Scatter（RS）内核的执行时序示意图。SM0和SM1的并发执行与等待机制清晰可见。当SM0因依赖未满足而自旋等待时，SM1仍在执行有效计算，从而实现了计算与等待的重叠。这种“忙等”在超级内核中是可接受的，因为等待时间极短，且避免了上下文切换的开销。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchH9PviaO1Yxwib6J94j8T3ubq1gWvdD4tZ0uGcmqmDcI8Iz7xdicHFiaLo1DhQticvdJ280j1nOcAfaE2wFuBibeTbf4VM1mLRic3op3ag/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=12)

图 7：静态融合的GEMM（MM）+ Reduce-Scatter（RS）内核的执行时序示意图。SM0和SM1的并发执行与等待机制清晰可见。当SM0因依赖未满足而自旋等待时，SM1仍在执行有效计算，从而实现了计算与等待的重叠。这种“忙等”在超级内核中是可接受的，因为等待时间极短，且避免了上下文切换的开销。

不同时刻的执行过程如下：

- 在 时刻，SM0 上的 MM0 完成并通知事件（计数器减 1）。此时 SM0 上的 RS 任务虽然已被调度，但会进入自旋等待（Spin-Wait）。
- 在 到 期间，SM1 继续执行自己的 MM0 任务，GPU 保持忙碌。
- 直到 时刻 SM1 完成，计数器归零，SM0 上的 RS 任务才被唤醒执行。

这种 SM 级别的细粒度交叠，是传统内核边界调度无法实现的。

**对于动态 Shape，静态调度采用了一种“向上对齐”的保守策略： *为未见过的 Shape 复用下一个更大采样 Shape 的执行队列。对于数据依赖，则假定最坏情况来预留资源*** 。虽然保守，但在可预测的密集计算场景下，它带来了近乎为零的调度开销。

### 3.2 动态调度：为 MoE 量身定制的负载均衡器

> 面对 MoE 这种 Tile 执行时间高度不确定、任务图拓扑在运行时才揭晓的场景，静态调度就显得力不从心了。ETC 的动态调度变换将调度器从“编译时”搬到了“GPU 上”，见下面算法 2。

![算法 2：ETC 动态调度转换算法。该算法同样以带事件张量依赖的分块级数据流图模块为输入，经复制模块、创建持久化内核、初始化 GPU 侧轻量调度器、添加任务出队逻辑、遍历任务网格并绑定事件完成与任务入队逻辑后，输出融合动态调度巨型内核的更新模块，它在芯片内实现轻量任务调度，事件完成后自动将依赖任务入队，空闲 SM 可原子取任务执行，无需 CPU 预计算任务队列，能有效解决 MoE 动态路由带来的 SM 负载失衡问题，完美适配数据依赖型 LLM 计算场景，仅付出极小的队列操作开销即可提升动态 workload 执行效率。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHibjo9se8SydRYWJ1II05m5iczodDRPWDEJqg1KibzibiaBDMcf5epfGK3ZibfiaOZUNDsSRFW1P9JvHBvkReIiabsGbxWBMBqORsftK6E/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=13)

算法 2：ETC 动态调度转换算法。该算法同样以带事件张量依赖的分块级数据流图模块为输入，经复制模块、创建持久化内核、初始化 GPU 侧轻量调度器、添加任务出队逻辑、遍历任务网格并绑定事件完成与任务入队逻辑后，输出融合动态调度巨型内核的更新模块，它在芯片内实现轻量任务调度，事件完成后自动将依赖任务入队，空闲 SM 可原子取任务执行，无需 CPU 预计算任务队列，能有效解决 MoE 动态路由带来的 SM 负载失衡问题，完美适配数据依赖型 LLM 计算场景，仅付出极小的队列操作开销即可提升动态 workload 执行效率。

![图8：经过动态调度变换后的GEMM + Reduce-Scatter。任务push和pop被插入，任务的执行由一个片上调度器动态协调。与静态调度的“预排期”不同，这里的执行顺序完全由运行时的事件触发关系决定。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchH8TjNRP7xGzvL6ZwiaNbwHEZiaicMicyGgXZZSrHnVNYoKNjYWoibqqamWX77FyEhlE9VY20M3UvTIqPg1JyEPK6Klx4E9vnyNIKjpE/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=14)

图8：经过动态调度变换后的GEMM + Reduce-Scatter。任务push和pop被插入，任务的执行由一个片上调度器动态协调。与静态调度的“预排期”不同，这里的执行顺序完全由运行时的事件触发关系决定。

如图 8 所示，编译器会为每个任务引入 `push` 和 `pop` 操作，并将它们与一个全局的任务队列关联。其核心机制如图 9 所示：

![图9：动态调度的推-弹（Push-and-Pop）机制。Producer任务完成后，通过原子操作将Consumer任务推入全局队列。空闲的SM则从队列中拉取任务执行。这种机制天然实现了负载均衡：执行快的SM会自然拉取更多任务。论文在附录E中提到，他们采用了“提前推”（Early Push）策略——一旦Producer被调度，就立即推送Consumer，从而将Push操作的开销重叠在Producer执行期间，避免增加关键路径长度。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHicrTZtGicX9mEuibyyz2fd1HjQLwHtNjB91wuic2vcZztibZZ3qhQ6zFC49VrtQWAocd8xSnrHqCqUriaGcAFsumiaQIcibXaY7ftc1P8/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=15)

图9：动态调度的推-弹（Push-and-Pop）机制。Producer任务完成后，通过原子操作将Consumer任务推入全局队列。空闲的SM则从队列中拉取任务执行。这种机制天然实现了负载均衡：执行快的SM会自然拉取更多任务。论文在附录E中提到，他们采用了“提前推”（Early Push）策略——一旦Producer被调度，就立即推送Consumer，从而将Push操作的开销重叠在Producer执行期间，避免增加关键路径长度。

- **Producer 任务** ：执行完毕后，原子更新事件计数器。当计数器归零时，调用 `scheduler.push_tasks` ，将依赖它的 Consumer 任务推入全局就绪队列。
- **Consumer 任务** ：当 SM 空闲并从队列中 `pop` 到一个任务时，首先执行 `event.wait()` 以确保所有 Producer 确已完成（这是一种双重检查机制），然后再执行实际的计算负载。

**静态调度与动态调度的选择体现了一种经典的权衡。**

- 静态调度胜在零开销，适合规整负载；
- 动态调度赢在灵活性，适合不规则负载。

ETC 的价值在于，它让开发者可以通过选择编译 Pass，在同一套 Event Tensor 描述上无缝切换这两种策略。

### 3.3 极简运行时：将“操作系统”编译进内核

> 一个值得深思的设计是，ETC 的运行时环境极其精简，见下图 10。

![图10：运行时架构对比。（a）传统运行时执行器中的任务图在内存中物化，只有tiled operators被编译。（b）ETC将调度逻辑编译进超级内核，无需运行时物化任务图。这种“编译进”的方式，使得ETC的内核几乎不依赖任何重量级的运行时库，部署极其轻便。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchH8eVrEmx5WSdDFRJr2o9H682ZWu5Uz4XuZEKBV1pdcN1KtdT9C1icFvxuiaDT7HfPj7icOzWBHLuJUNrqG4doa8SQEe3LgKunnwns/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=16)

图10：运行时架构对比。（a）传统运行时执行器中的任务图在内存中物化，只有tiled operators被编译。（b）ETC将调度逻辑编译进超级内核，无需运行时物化任务图。这种“编译进”的方式，使得ETC的内核几乎不依赖任何重量级的运行时库，部署极其轻便。

在传统的任务图运行时（如 Legion、Realm）中，整个任务图需要在内存中显式物化，由一个通用的执行器遍历图谱、发射内核。这种方法虽然灵活，但引入了可观的内存和调度开销。

> ETC 走了一条完全不同的路：它将调度逻辑 **编译进内核本身** 。具体而言：

- Event Tensor 被直接降低为一个整数张量，复用现有的张量数据结构。
- `notify()` 实现为 `atomicSub` （原子减）。
- `wait()` 实现为一个对计数器值的自旋循环。
- 运行时状态仅由这些整数张量和调度器的任务队列组成。

这种设计哲学—— **用编译器生成取代运行时解释** ——是 ETC 能够实现超低延迟和快速预热的根本原因之一。

### 3.4 端到端编译流程一览

> 下图 15 是完整的 ETC 编译流程，始于一个未优化的、已标注 Event Tensor 的计算图。

![图15：ETC的端到端编译流程。从带Event Tensor注解的计算图出发，经历图优化、Tile级优化、调度变换，最终生成单个持久化内核。这一流程将过去需要手工完成的复杂融合与调度工程，系统化为可复用的编译器Pass。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHicOtOiblqTacZaxuDc9CfbkicibOn6wybRok6AbGic9DEhIuZyKCCSEwzEjl6GicysxMlUhz0SRlJeZNiap3ibwv1nu49gtJNdz2GKiaA4/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=17)

图15：ETC的端到端编译流程。从带Event Tensor注解的计算图出发，经历图优化、Tile级优化、调度变换，最终生成单个持久化内核。这一流程将过去需要手工完成的复杂融合与调度工程，系统化为可复用的编译器Pass。

该图首先经过标准的图优化如内存规划。接着是 Tile 级优化，为每个算子确定具体的硬件指令映射和流水线策略。然后，用户可以选择应用静态调度 Pass 或动态调度 Pass。最后，生成的融合设备函数以持久化内核的形式输出为 GPU 代码。

一个可选的预取（Prefetching）Pass 还可以插入权重预取逻辑，使得每个 Tile 能在输入激活到来之前就提前加载权重，进一步隐藏内存延迟。

## 四、性能的铁证：从通信重叠到 MoE 霸权

> 数据不会说谎。在与工业级强基线（vLLM、SGLang、cuBLAS、Triton）的对决中，ETC 在通信-计算重叠、MoE 动态路由、低 Batch 解码三大战役中均交出了统治级的成绩单。 **更重要的是，它从根本上消灭了 CUDA Graph 的预热噩梦。**

### 4.1 通信与计算：打破 Tensor 并行的气泡

在 Tensor 并行推理中，GEMM 后的 Reduce-Scatter 和 All-Gather 后的 GEMM 是两个核心模式。理想的执行应该是通信与计算完全重叠，不留气泡。

ETC 在 8 张 B200 上，分别使用动态调度（应对通信抖动）和静态调度（应对规则的 Ring 算法），与 cuBLAS+NCCL、TP-Async、Triton Distributed、cuBLASMp 进行了对比。

#### GEMM + Reduce-Scatter

> ETC 在所有 MLP 配置上均优于所有基线：

![图11：在8张B200上使用动态调度器的GEMM + Reduce-Scatter性能结果。ETC在所有配置下均取得了显著加速。这一优势源于Event Tensor将整体操作分解为深度流水化的任务图，并通过动态调度有效应对了潜在的通信延迟波动。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHicTa7YK3Fa3iaSib0tbKQJdR7Np2qZoezz8hicv5o8SbibUGxkv5pD1tN8YicWOtW8qNXzia1bnR3PH85vEVACEbhn6XibdTicicy1DzdQ4/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=18)

图11：在8张B200上使用动态调度器的GEMM + Reduce-Scatter性能结果。ETC在所有配置下均取得了显著加速。这一优势源于Event Tensor将整体操作分解为深度流水化的任务图，并通过动态调度有效应对了潜在的通信延迟波动。

- 在最大模型配置上 **相比 cuBLAS+NCCL 基线实现了最高 的执行时间加速** 。
- TP-Async 的粗粒度切分导致 Tile 过小或过大，重叠效果不佳；
- Triton-Dist 对 B200 的支持尚不成熟。

ETC 的细粒度任务流水线展现了压倒性优势。

#### All-Gather + GEMM

> 趋势和 GEMM + Reduce-Scatter 保持一致，ETC 同样在大多数配置下保持领先，最高加速同样达到 **。**

![图12：在8张B200上使用静态调度器的All-Gather + GEMM性能结果。ETC再次展现了接近或超越所有基线的性能。静态调度在这里被用于精确编排计算与Ring算法的通信顺序，以极低的运行时开销实现了近乎完美的重叠。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHibv5GWHhTqJiaPZM8vvP5ibjLw1bWJsmnlvbF7smFezicDTNpiahhxNsiakCvfVLRPFdArEHuoptZkLO2nHhlxOartoIo1OuLvS6ESc/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=19)

图12：在8张B200上使用静态调度器的All-Gather + GEMM性能结果。ETC再次展现了接近或超越所有基线的性能。静态调度在这里被用于精确编排计算与Ring算法的通信顺序，以极低的运行时开销实现了近乎完美的重叠。

### 4.2 MoE 层：动态性的终极考验

> 这是 Event Tensor 能力的试金石。Qwen3-30B-A3B 的 MoE 层包含 128 个专家，Top-8 路由，是典型的数据依赖负载。 **ETC 使用动态调度，将整个 MoE 数据流融合进单个超级内核。**

![图13：在单张B200上MoE层的性能结果。ETC的动态调度超级内核在所有Token数量下均表现最佳。数据依赖的Event Tensor和动态调度器共同作用，不仅实现了算子间的流水线并行，还通过运行时负载均衡解决了专家路由带来的计算不规则性。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchH9WcSBtFBWxLu78WjbiadBPxBZ9r0JnVMuUkm1ibu6fHBQibP87REhwaaibO2aeAqcmdkibcfAAnGHyHmrMzSibRcC3w9cXhPMTEGv98/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=20)

图13：在单张B200上MoE层的性能结果。ETC的动态调度超级内核在所有Token数量下均表现最佳。数据依赖的Event Tensor和动态调度器共同作用，不仅实现了算子间的流水线并行，还通过运行时负载均衡解决了专家路由带来的计算不规则性。

如图 13 所示，ETC 在不同 Token 数量下 **均显著超越 Triton 和 FlashInfer** 。值得注意的是，作者分析指出 FlashInfer 的 GroupGEMM 针对较大 Token 数优化更充分，而 Triton 的 GroupGEMM 融合了 gather/scatter 操作，因此两者在不同 Token 数下的相对排名会发生变化。

值得注意的是， ***FlashInfer 在大 Token 数下对 GroupGEMM 的优化更好，而 Triton 则在融合 gather/scatter 上有优势*** 。 **ETC 之所以能全面胜出，在于两点** ：

- 其一，Event Tensor 打破了 MoE 两阶段 GroupGEMM 间的全局同步屏障，实现了细粒度流水线；
- 其二，片上动态调度器为不规则的专家负载提供了比任何静态分配都更优的负载均衡，最大限度地减少了 SM 空闲时间。

### 4.3 低 Batch 解码：重新定义实时交互的延迟下限

> 在实时智能体、代码助手等低 Batch 场景，延迟就是生命。我们将 ETC 编译的超级内核集成到 **Qwen3-30B-A3B 和 Qwen3-32B 的完整解码流程** 中（包括 Attention、MoE、MLP、KV-Cache 等所有算子）， **与 vLLM 和 SGLang 这两个工业级顶流系统进行端到端比较。**

![图14：Qwen3-30B-A3B和Qwen3-32B模型服务的端到端性能（数值越低越好）。ETC在单卡场景下全面领先，尤其在MoE和低Batch Size下优势巨大。这证明了超级内核架构在消除内核边界、暴露算子间并行性方面的根本性优势。四卡Tensor并行下性能持平，则指出了未来在分布式CPU调度层面的优化空间。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHib1JUw7X0rrQRptVibMib64ykfibYLXC1CEUqb1HvA3VNJEXqfOBNEYEg32VXkVy2icTalOzzqBVHIJlrlOa5jabts4iapxPCBvYmyQ/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=21)

图14：Qwen3-30B-A3B和Qwen3-32B模型服务的端到端性能（数值越低越好）。ETC在单卡场景下全面领先，尤其在MoE和低Batch Size下优势巨大。这证明了超级内核架构在消除内核边界、暴露算子间并行性方面的根本性优势。四卡Tensor并行下性能持平，则指出了未来在分布式CPU调度层面的优化空间。

- **Qwen3-30B-A3B (MoE)：** 在 Batch Size 为 1 时，ETC 的 TPOT（每输出一个 Token 的时间） **比 vLLM 快，比 SGLang 快， *见图 14 左。这是对 MoE 动态负载优化能力的直接体现。***
- **Qwen3-32B (Dense)：** ETC 在所有 Batch Size 下均保持最低延迟， **在 Batch Size 为 1 时比 vLLM 快， *见图 14 中。***
- **Qwen3-32B (TP=4)：** 在 4 卡 Tensor 并行下，ETC 性能与 vLLM 持平（ 到 ）。SGLang 在此场景下表现更优， ***论文分析这归因于其高度优化的 CPU 侧调度器，而非 GPU 内核本身*** 。ETC 的 GPU 内核性能在这里与最强基线已无本质差距。
![表 4 MLP 模型配置（S = 序列长度，H = 隐藏层维度，I = 中间层维度）：该表列出 8 组源自主流 LLM 的 MLP 配置，序列长度固定 8192，覆盖 8B-405B 模型的隐藏层与中间层维度，用于验证 ETC 计算 - 通信融合核性能。这些配置覆盖从小型到超大型 LLM 的计算特征，ETC 在全配置中均实现性能提升。事件张量抽象与调度转换，对不同规模 LLM 的 GEMM + 通信融合优化具备普适性，适配全尺寸 LLM 推理场景。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchH94H0qRyicCLQvwjlZ1P8vWKE1icNFek4WJYNiaehypWhRbIOo6dTZrPWj7vPQgXrIjSePbIWLbCsX3qmQZcReibD7n3vXDNvpoHCA/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=22)

表 4 MLP 模型配置（S = 序列长度，H = 隐藏层维度，I = 中间层维度）：该表列出 8 组源自主流 LLM 的 MLP 配置，序列长度固定 8192，覆盖 8B-405B 模型的隐藏层与中间层维度，用于验证 ETC 计算 - 通信融合核性能。这些配置覆盖从小型到超大型 LLM 的计算特征，ETC 在全配置中均实现性能提升。事件张量抽象与调度转换，对不同规模 LLM 的 GEMM + 通信融合优化具备普适性，适配全尺寸 LLM 推理场景。

### 4.4 预热开销：AOT 对 JIT 的降维打击

> 如果说延迟优化是“加速”，那么消除预热开销就是“减负”。在云服务场景中，模型加载后的 JIT 编译和 CUDA Graph 捕获是巨大的时间黑洞。

如下表 1 所示，vLLM 预热耗时 期间捕获了 67 个不同的 CUDA Graph ，SGLang 更是需要 捕获 351 个 graph。 **而 ETC，凭借其 AOT 编译出的单一、形状泛化的超级内核，预热时间仅需 。**

![表 1 不同 graph 捕获方法下 Qwen3-32B 模型服务的预热时间。该表对比 SGLang、vLLM（JIT）与 ETC（AOT）的预热耗时和 JIT 图捕获次数，ETC 仅 35 秒且 0 次捕获，远优于传统方案。核心原因是 ETC 基于事件张量的符号形状实现 AOT 编译，无需运行时重复捕获 CUDA 图；传统方案需为不同形状捕获大量静态图，产生高额预热开销。事件张量的动态形状抽象彻底消除 JIT 与图捕获开销，大幅降低 LLM 服务部署的预热成本。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHicYtqbuewEXkiczkauy44OYkxptkB2tVzf4Lj1NlyHFeZQU0JlaLgeFv5WUU1nxnBPFJeab6n0NhIDJEQqxMt8rlR5UOPrxLia90/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=23)

表 1 不同 graph 捕获方法下 Qwen3-32B 模型服务的预热时间。该表对比 SGLang、vLLM（JIT）与 ETC（AOT）的预热耗时和 JIT 图捕获次数，ETC 仅 35 秒且 0 次捕获，远优于传统方案。核心原因是 ETC 基于事件张量的符号形状实现 AOT 编译，无需运行时重复捕获 CUDA 图；传统方案需为不同形状捕获大量静态图，产生高额预热开销。事件张量的动态形状抽象彻底消除 JIT 与图捕获开销，大幅降低 LLM 服务部署的预热成本。

这组数据揭示了 ETC 在生产环境中的巨大潜力。 **它将服务启动时间从“分钟级”拉低到“秒级”，对于弹性伸缩、无服务器推理等场景具有革命性意义** 。这不仅仅是性能优化，而是部署范式的根本转变。

### 4.5 动静之辩：两种调度哲学的实证对决

> 表 2 和表 3 定量分析了静态调度与动态调度在不同负载下的表现。

- **在 MoE 这类数据依赖负载上（表 2）** ：动态调度凭借其负载均衡能力，在大多数 Batch Size 下优于静态调度，在 Batch Size=1024 时领先达 。 ***这清晰地表明，面对不规则任务，灵活的运行时调度利大于弊。***
![表 2 不同 ETC 调度方法在 MoE 层相对未融合巨型内核的性能。该表对比静态 / 动态调度在 MoE 层的相对性能，动态调度在 128-4096 令牌场景下最优，达 1.08 倍，静态调度仅小幅提升。MoE 存在数据依赖的令牌路由，静态调度预分配任务易导致 SM 负载失衡，动态调度则通过 GPU 内调度器实时分发任务平衡负载。可以看出，数据依赖类动态工作负载中，动态调度的负载均衡优势显著，静态调度仅适配无数据依赖的规则场景。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHib3P9Sw9pgicsuic8ibTlmHpoXYJB6B95jpsgoCiaMiaBCltbRuwqEK2QKViatiaTa2iaoLON8kvNpAYLzytbhyhy19iak2ibn2qLiaSicoFug/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=24)

表 2 不同 ETC 调度方法在 MoE 层相对未融合巨型内核的性能。该表对比静态 / 动态调度在 MoE 层的相对性能，动态调度在 128-4096 令牌场景下最优，达 1.08 倍，静态调度仅小幅提升。MoE 存在数据依赖的令牌路由，静态调度预分配任务易导致 SM 负载失衡，动态调度则通过 GPU 内调度器实时分发任务平衡负载。可以看出，数据依赖类动态工作负载中，动态调度的负载均衡优势显著，静态调度仅适配无数据依赖的规则场景。

- **在密集的 Tensor 并行负载上（表 3）** ：静态调度完胜。动态调度的队列 Push/Pop 开销在分布式环境下被放大，导致其性能显著落后于静态调度。 ***静态调度对 ETC-unfused（单事件全局同步）的 领先，则纯粹来自于 Event Tensor 带来的细粒度流水线并行。***
![表 3 张量并行 TP=4 时不同 ETC 调度方法在 Qwen3-32B 相对未融合巨型内核的性能。该表展示 TP=4 分布式场景下静态 / 动态调度性能，静态调度提升 6%-9%，动态调度性能下降。分布式环境中动态调度的任务队列推送 / 弹出会产生远程通信开销，静态调度预分配任务无运行时调度损耗，更适配密集 Transformer 规则工作负载。张量并行的规则分布式工作负载优先选静态调度，动态调度因通信开销不适用，验证调度策略需匹配工作负载特性。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchH8flscZsw6ys5C8logbxlMx0IGkK7dsIk2gSB5SicBkIhjyI3gtQchFkgG0Cnk90YvBJhVopphdHicHicT8Gdccxhfqnn9p0RLnl4/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=25)

表 3 张量并行 TP=4 时不同 ETC 调度方法在 Qwen3-32B 相对未融合巨型内核的性能。该表展示 TP=4 分布式场景下静态 / 动态调度性能，静态调度提升 6%-9%，动态调度性能下降。分布式环境中动态调度的任务队列推送 / 弹出会产生远程通信开销，静态调度预分配任务无运行时调度损耗，更适配密集 Transformer 规则工作负载。张量并行的规则分布式工作负载优先选静态调度，动态调度因通信开销不适用，验证调度策略需匹配工作负载特性。

**这场对比实验深刻揭示了一个事实：没有银弹。动态调度在 MoE 的不规则性中找到了它的主场，而静态调度则在密集计算的规整舞台上捍卫了荣誉** 。ETC 的智慧在于提供了一个统一的框架，让开发者可以基于负载特性做出权衡，而非被绑定在单一策略上。

## 五、相关工作：在巨人的肩膀上眺望

> Event Tensor 并非横空出世，它生长于并行编程模型、深度学习编译器与 LLM 服务系统的交叉地带。理解它与这些领域工作的关系，才能看清它的历史坐标与创新价值。

### 5.1 深度学习编译器与算子融合

- TVM、XLA、PyTorch 2 的 torch.compile 等主流编译器长期致力于图级优化和算子融合。
- Rammer 和 Roller 探索了在 GPU 上进行 Tile 级任务的软件调度。

然而，这些工作普遍缺乏一个显式的、用于跟踪和优化细粒度依赖关系的抽象。它们或者止步于内核边界，或者将融合局限在有限的模式内。

**Event Tensor 可以看作是这一脉络的自然延伸** ——它提供了缺失的“同步语言”，使得编译器能够对跨算子的 Tile 级依赖进行推理和变换。

### 5.2 LLM 推理服务系统

vLLM、SGLang、TensorRT-LLM 代表了 LLM 推理系统的当前巅峰。它们通过 PagedAttention、Continuous Batching、CUDA Graph 等系统级优化取得了巨大成功。然而，它们在 GPU 执行层仍然受制于内核边界和静态图假设。

**ETC 并非要取代这些系统，而是旨在成为它们更强大的后端** 。例如，SGLang 的上层调度逻辑完全可以复用 ETC 生成的超级内核，从而在保持其卓越 CPU 调度能力的同时，获得 GPU 执行效率的质变。

### 5.3 任务并行模型与 GPU 超级内核

> 从 Cilk、Legion 到 Realm，任务并行编程模型历史悠久。但它们多聚焦于 CPU 主导的粗粒度任务调度。

- 近期的 Mirage、Look Ma, No Bubbles!等工作开始探索 LLM 的超级内核化，但它们主要针对单 Batch、Dense 模型，且通常固化了某种特定调度策略。
- Graphene 将线程建模为具有同步能力的张量，概念上与 Event Tensor 相关，但其目标仍是单内核优化，而非多算子超级内核融合。

Event Tensor 的独特贡献在于，它在这些工作的基础上，提供了一个 **统一的编译器抽象** ，系统地解决了 **动态形状** 和 **数据依赖** 这两大挑战，并支持 **动静双态调度** 的自动变换。它补全了从任务并行理论到 LLM 推理生产环境之间的关键拼图。

## 六、结论与展望：迈向完全动态的 GPU 计算时代

### 6.1 结论总结

> Event Tensor 的提出，标志着 GPU 编程模型的一次重要进化。其核心贡献可概括为三点：

1. **抽象创新** ：提出了 Event Tensor， ***将细粒度同步提升为编译器中的一等公民，通过符号 Shape 和运行时索引表达式，首次在超级内核框架下系统性地解决了 Shape 动态性与数据依赖动态性。***
2. **编译方法** ：构建了 ETC 编译器，实现了从 Event Tensor 描述到静态/动态调度超级内核的自动化变换，将复杂的手工融合工程转化为可复用的编译 Pass，并设计了一种极简的、编译进内核的运行时。
3. **实验验证** ：在通信-计算重叠、MoE 层和端到端推理等多个维度，ETC 均展现了超越工业级基线的卓越性能（最高 加速），并通过 AOT 编译将系统预热开销降低了一个数量级（）。

### 6.2 进阶分析

> 抛开论文的叙事，我们需要冷静审视 Event Tensor 的边界条件与潜在成本。

- **问题的解决程度** ：Event Tensor 从根本上解决了 **GPU 端执行的低效问题** ，但它并未解决 **CPU 端的调度开销** 。
![图14：Qwen3-30B-A3B和Qwen3-32B模型服务的端到端性能（数值越低越好）。ETC在单卡场景下全面领先，尤其在MoE和低Batch Size下优势巨大。这证明了超级内核架构在消除内核边界、暴露算子间并行性方面的根本性优势。四卡Tensor并行下性能持平，则指出了未来在分布式CPU调度层面的优化空间。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchH9Ty1GtrQnYwxKnnFxicXKb73wcrn48eOChr0FSdNQiaTS4jYmXfk027RCB28c38XBx8HMBPWnTSlPJATRFYaLRQ0RiamUG4ORIng/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=26)

图14：Qwen3-30B-A3B和Qwen3-32B模型服务的端到端性能（数值越低越好）。ETC在单卡场景下全面领先，尤其在MoE和低Batch Size下优势巨大。这证明了超级内核架构在消除内核边界、暴露算子间并行性方面的根本性优势。四卡Tensor并行下性能持平，则指出了未来在分布式CPU调度层面的优化空间。

如图 14（右）所示，在 TP=4 时，ETC 的 GPU 内核性能已不输甚至略胜一筹， **但端到端延迟仍被 SGLang 拉开。 *作者将此归因于 SGLang 高度优化的 CPU 调度器以及 ETC 当前服务引擎中较高的 CPU 侧开销——这是工程实现层面的问题，而非 Event Tensor 抽象本身的根本局限*** 。

- **方法论的限制** ：静态调度对动态 Shape 的“向上对齐”策略是一种妥协。当实际 Shape 远小于采样 Shape 时，会导致 SM 队列中存在大量空任务槽，造成资源浪费。 ***动态调度的全局任务队列则是一个潜在的竞争热点，当 SM 数量激增（如未来的超大 GPU）时， `push` / `pop` 的原子操作竞争可能成为新的瓶颈。***
- **隐形成本** ：AOT 编译虽然消除了运行时预热，但其 ***编译时间本身可能很长*** ，Qwen3-32B 离线编译需 。对于模型迭代极快的研发阶段，这可能影响开发效率。此外， ***超级内核将大量代码塞进单个函数，可能对指令缓存（I-Cache）造成压力，这在论文中并未被深入讨论。***

客观地说，Event Tensor 在它所定义的“GPU 内核融合与调度”问题域内，给出了近乎完美的解答。它的局限性，恰恰为我们指明了这场性能革命的下一个战场。

### 6.3 未来工作

论文作者展望了未来能够自动从标准计算图生成 Event Tensor 任务图的高级编译 Pass，这将进一步降低超级内核的编程门槛。此外，他们也计划探索与更多领域特定语言（DSL）的集成。

> 视角如果站在 AI Infra 发展的宏观趋势上，NeuralTalk 认为 Event Tensor 的涟漪效应将远超论文本身。

1. **AI Chip 的指令集启示** ：Event Tensor 所表达的细粒度、数据驱动的依赖同步，与数据流架构的理念不谋而合。这是否暗示了未来 AI ***专用芯片在硬件层面可以直接支持类似“事件张量”的同步原语？* 如果硬件能够直接执行 `notify` 和 `wait` 而不需要通过原子操作轮询内存，能效比将再上一个台阶。**
2. **CPU-GPU 协同的终局思考** ：既然 GPU 可以自主调度任务，那么 CPU 的角色是否可以进一步后撤，变为纯粹的“请求提交者”？ ***未来，一个完整的 LLM 推理请求，是否可以完全由一个 GPU 持久化内核从头到尾处理，CPU 仅在首尾参与？*** 这将催生一种全新的、以 GPU 为中心的服务器无感知（Serverless）推理架构。
3. **软件生态的“编译器优先”运动** ：ETC 的成功再次证明，对于极度性能敏感的底层软件，手工优化终有极限。 ***将专家知识编码进编译器，通过高层抽象自动生成最优代码，是通往极致性能的唯一路径*** 。Event Tensor 的抽象有望被 Triton、MLIR 等社区吸收，推动一场更广泛的“编译器优先”运动。

> Event Tensor 不仅是一个技术方案，它更像一个宣言：在动态性与并行性日益成为 AI 计算核心矛盾的今天，我们的编程模型和编译技术必须做出根本性的回应。这场始于 GPU 内核的变革，或许才刚刚拉开序幕。

相关推荐

- [无需手动构建MegaKernels！Luminal 编译生成 MegaKernels：解决 GPU SM 负载不均，消除内核启动开销与内存气泡，适配任意架构！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447896119&idx=1&sn=80dce399f083de55a8cded74e2a54c3d&scene=21#wechat_redirect)
- [性能相比SGLang/vLLM最高提升1.7倍！Mirage Persistent Kernel：首个自动巨核化多GPU LLM推理的编译器-运行时系统，细粒度计算-通信重叠](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895728&idx=1&sn=88a267756d2f2057f868d43f0500841f&scene=21#wechat_redirect)
- [所有层融合成一个 GPU kernel：消除 GPU 空闲周期！为 Llama-1B 设计低延迟的大内核！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888839&idx=1&sn=a77ed7a2987f874b3e52e253993fdf89&scene=21#wechat_redirect)
- [MoE 所有层融到一个分布式算子GPU Kernel！FlashDMoE：GPU内核-硬件协同解锁大规模分布式机器学习性能极限！](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447888883&idx=1&sn=14a76e02fc523d30f613d4a2219c0fea&scene=21#wechat_redirect)
- [HotChips 2025 从摩尔定律到巨型内核：GPU 上机器学习系统优化的十年跃迁，Zhihao Jia 演讲解读](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447891453&idx=1&sn=632b9abf4170c20af52b481ca8ec24e7&scene=21#wechat_redirect)

交流加群请在 NeuralTalk 公众号后台回复：加群

GPU · 目录