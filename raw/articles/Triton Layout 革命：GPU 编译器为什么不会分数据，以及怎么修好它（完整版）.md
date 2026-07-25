---
title: "Triton Layout 革命：GPU 编译器为什么不会分数据，以及怎么修好它（完整版）"
source: "https://mp.weixin.qq.com/s/1oN2r9DXj4OCqVDhLvdWhQ"
author:
  - "[[老许漫谈AIInfra]]"
published:
created: 2026-07-06
description: "一栋楼 100 个工人，工头喊了一嗓子“去装修”但没说谁负责几楼几号房。旧 Triton 就是这个工头。三层修复，从 BlockedEncoding 到 F₂ 线性代数，把 GPU 编译器的数据分配从靠猜变成数学推导。"
tags:
  - "clippings"
---
老许漫谈AIInfra 老许漫谈AIInfra *2026年7月1日 07:10*

📖 阅读时间：约 25 分钟

**本文目录**

**第一章** ：Layout 是什么

**第二章** ：Triton 是什么 —— Python 写 GPU 核函数，但翻译链里藏着一个关键决策点

**第三章** ：GPU 架构详解

**第四章** ：旧 Triton 的问题

**第五章** ：解决方案一：BlockedEncoding

**第六章** ：新问题：N² 爆炸

**第七章** ：终极解法：F₂ 线性代数 —— 用 GF(2) 矩阵把 Layout 变成数学，自动推导一切

**第八章** ：结果与总结

第一章：Layout 是什么，为什么它让你损失了 20-40% 性能

你写了一个 Triton kernel，跑了 GEMM 基准测试，结果是 CUDA 同等实现的 63%。

这不是你的错，也不是 Triton 的 bug。这是一个深埋在编译器里的系统性问题——旧 Triton 在最关键的一步上靠猜。

这一步叫 **Layout** 。

想象一栋楼有 100 个工人，工头喊了一嗓子"去装修"，但没说谁负责几楼几号房。工人们自己猜着分，结果有人扎堆，有人闲着，电梯挤成一锅粥。这就是坏 Layout 在 GPU 上的写照。

Layout 的核心概念很简单： **GPU 上"谁负责哪块数据"的分配方案** 。在一个 256×256 的矩阵乘法里，GPU 有成千上万个线程，每个线程只能处理一小块数据。Layout 决定了：第 0 号线程负责矩阵的哪几个元素？第 1 号线程呢？整个 Warp（32 个线程）是否刚好对齐 TensorCore 的指令粒度？

![Layout 数据分配概念图](https://mmbiz.qpic.cn/sz_mmbiz_png/8bQpc14UAHz4dscMXO5fMXJNBBzIVKNUG2KdWw9Mic5LWHTsUIcEYkmiaDic4P9A6F5HUeNiaQ7zyqCFFLCiaQcGmC9MjBlCI8fZ0ibO4pMuvNbRM/640?from=appmsg&wx_fmt=png&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

Layout 数据分配概念图

Layout 分得好，32 个线程同时访问 32 个不同的内存 bank，TensorCore 一次命中，带宽拉满。Layout 分得差，32 个线程挤在同一个 bank，串行排队，throughput 跌到 1/32。

真实数据并不温和：FlashAttention-3 的作者在开发过程中发现，旧 Triton 的默认 Layout 无法满足 TensorCore 指令对齐要求， **不得不手写 byte permute 作为补偿** 。GEMM 场景下，Triton kernel 跑出 CUDA 60-80% 是常见情况，不是极端案例。

这篇文章讲清楚三件事：旧 Triton 为什么会产生坏 Layout，第一代修复（BlockedEncoding）怎么解决又带来什么新问题，以及终极解法（F₂ 线性代数）怎么用数学一劳永逸。

第二章：Triton 是什么

2.1 Triton 的翻译链

Triton 是 OpenAI 开发的 GPU 编程语言和编译器，目标是让 Python 程序员能写出接近 CUDA 专家水平的 GPU 核函数。

它的翻译链是：

Python DSL

↓

Triton IR（中间表示）

↓

LLVM IR

↓

PTX / AMDGCN / SPIR-V（GPU 汇编）

↓

实际 GPU 指令

每一步都是信息压缩的过程。Python 里的 `tl.dot(a, b)` 最终变成几十条寄存器操作指令。问题在于：这些指令怎么分配给哪个线程，在哪一步决定？

2.2 Triton vs CUDA：价值主张

CUDA 是 C++ 方言，直接控制线程、内存、寄存器。写好 CUDA 需要深刻理解：内存合并（coalesced access）、shared memory bank conflict、TensorCore 指令对齐、warp divergence……每一项都是独立的专业知识。

Triton 的价值主张是 **tile abstraction** ：你只需要描述"我要对这个 tile 做什么运算"，编译器负责把这个 tile 映射到线程、处理内存访问模式、避免 bank conflict、对齐指令。

用一句话说： **Triton 让写 Python 的人，获得 CUDA 专家才能达到的性能** 。

这个承诺在大多数情况下兑现了——但 Layout 决策是它最薄弱的环节。

2.3 翻译链中的关键节点：数据分配

在 Triton IR 到 LLVM IR 的降级过程中，有一个隐藏的关键决策： **这个 tile 里的每个元素，交给哪个线程（或哪组线程）处理？**

这个决策就是 Layout。它必须在编译时做出，因为 GPU 没有运行时调度器——每个线程在启动前就知道自己的任务。

旧 Triton 在这一步怎么做的？我们先把 GPU 架构讲清楚，再来回答这个问题。

第三章：GPU 架构详细介绍

3.1 四层结构与内存层次

GPU 的执行模型是严格分层的：

![GPU 四层内存层次结构](https://mmbiz.qpic.cn/mmbiz_png/8bQpc14UAHwjqOcaic5W9won3EFVzeG90icRQmCvUVb6oMiaSGVQ6SWw8QcyfTwkguiaBDdJMuqJDRfgVg8ia919duVt9eIvq9LDAaDTH47Ajwxg/640?from=appmsg&wx_fmt=png&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1)

GPU 四层内存层次结构

**Grid（网格）** ：一次 kernel 启动的所有工作单元。内存对应 HBM（High Bandwidth Memory，40-80GB，带宽约 2-3 TB/s）。

**CTA / Thread Block（线程块）** ：可以协作的线程组，通常 128-1024 个线程。内存对应 L1 Cache + Shared Memory（32-192KB，带宽约 10-20 TB/s）。同一 CTA 内的线程可以通过 `__syncthreads()` 同步。

**Warp（线程束）** ：32 个线程，GPU 调度的最小单元。这 32 个线程 **同步执行同一条指令** （SIMT 模型）。内存对应寄存器（最快，无访问延迟）。

**Thread（线程）** ：最小执行单元，有自己的寄存器。

这四层结构直接决定了数据分配的粒度。你不能给单个线程一个"聪明的分配"——你必须给整个 Warp 设计一个协调的分配方案。

3.2 Warp 为什么特别关键

三个原因让 Warp 成为 Layout 设计的核心：

**第一，32 个线程同步执行。** 一条 load 指令，32 个线程同时发出内存请求。如果这 32 个请求刚好打到 32 个不同的内存 bank（128 字节对齐），硬件一次全部处理。如果都打到同一个 bank，串行，慢 32 倍。

**第二，TensorCore 指令粒度。** TensorCore 是 NVIDIA 为矩阵运算专门设计的硬件单元，一条指令能做 16×16×16 或更大规模的矩阵乘法。 **TensorCore 指令的操作数，必须由一个 Warp 的 32 个线程协作提供** ，且有严格的位置要求——哪个线程持有矩阵的哪个元素是硬编码的。Layout 必须与这个要求对齐，否则 TensorCore 无法命中。

**第三，Warp Shuffle 通信。** 同一 Warp 内的线程可以通过 `__shfl_xor_sync` 等指令直接交换寄存器值，不需要经过 Shared Memory。这是高性能 reduce 和转置的关键。但前提是 Layout 设计时就考虑到了 Warp 内的数据分布。

3.3 好 Layout vs 坏 Layout：以 GEMM 256×256 为例

![好 Layout vs 坏 Layout 对比](https://mmbiz.qpic.cn/mmbiz_png/8bQpc14UAHzmaexbyHWyVWqQFktOXdg3yNM7jdEqMIcIuEJ51DhX6zQCS1u9hJmXXj0aPEmyDz5OIZlmtTPcsn21icxZqG2o9phuhlneVR3I/640?from=appmsg&wx_fmt=png&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)

好 Layout vs 坏 Layout 对比

**好 Layout** ：一个 Warp 的 32 个线程，分别负责矩阵的 32 个连续列。内存访问合并，TensorCore 指令粒度对齐，一次 `ldmatrix` 完成加载。

**坏 Layout** ：一个 Warp 的 32 个线程，按行分配，每个线程负责一整行的片段。相邻线程访问的内存地址间距 256 字节，远超 cache line 宽度，无法合并。TensorCore 需要的元素散落在多个线程，必须先做一次转换。

在 256×256 的 GEMM 实测中，从好 Layout 切换到坏 Layout，性能差距约 20-40%。这不是理论数字，是实测结果。

**核心认知** ：Layout 不是优化细节，是正确性的门槛。TensorCore 指令要求特定的寄存器分布，如果 Layout 不满足，编译器必须插入补偿转换，性能直接大幅下降。

「老许漫谈AIInfra」 · 持续关注 AI 基础设施工程实践

第四章：旧 Triton 的问题——为什么会产生坏 Layout

为什么会跳过 Warp 层

旧 Triton（2.x 及之前版本）在 Layout 决策上有一个根本性的设计缺陷： **它的表达体系只有 CTA 层和 Thread 层，跳过了 Warp 层** 。

这个设计在 Triton 早期是合理的——彼时 TensorCore 还不是主流，编译器也相对简单。Triton 的思路是：先把 tile 分配给 CTA，再把 CTA 内的元素分配给每个 Thread，Warp 这一层由硬件自动管理。

问题是：TensorCore 的指令粒度恰好是 Warp 级别的。跳过 Warp 层，意味着编译器无法在设计阶段保证 Warp 内的数据排列符合 TensorCore 要求。

结果：大量补偿转换

旧 Triton 的降级路径是：

CTA 级 tile 分配

↓（直接）

Thread 级寄存器分配

编译器生成的默认 Layout，通常是按 row-major 顺序给每个 Thread 分配连续元素。这种 Layout 对 TensorCore 来说是错的。

于是编译器检测到不匹配，自动插入 **layout conversion pass** ——在计算之前，先做一轮数据重排，把线程持有的数据移动到正确位置，然后才能调用 TensorCore 指令。

这个 layout conversion 本身需要时间：Shared Memory 读写、Warp Shuffle、寄存器复制。对于计算密集型 kernel（如 GEMM），这些额外开销占总时间的 20-40% 并不罕见。

**明确结论** ：旧 Triton treatment（CTA→Thread 直接降级，跳过 Warp）→ 坏 Layout（不匹配 TensorCore 指令粒度）→ 大量 layout conversion → 性能损失 20-40%。

「老许漫谈AIInfra」 · 持续关注 AI 基础设施工程实践

第五章：解决方案一——BlockedEncoding（ML-Triton）

ML-Triton（arXiv 2503.14985）提出了第一代系统性修复方案： **BlockedEncoding** ，通过四个参数精确描述 Layout，覆盖从 CTA 到 Thread 的所有层级。

BlockedEncoding 四参数详解

![BlockedEncoding 可视化](https://mmbiz.qpic.cn/mmbiz_png/8bQpc14UAHz6D2x6xiaEBon20l79H3wN4ov7kkj0g25zLY28SCW5NOjPRAjJIOY1WhEzWf7kl0FCc18eWOtagtiaQSXU6ASdqHesUvqDzf7fw/640?from=appmsg&wx_fmt=png&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=3)

BlockedEncoding 可视化

BlockedEncoding(

sizePerThread, # 每个 Thread 持有的数据形状，如 \[1, 8\]

threadsPerWarp, # Warp 内的线程排布，如 \[8, 4\]（8行×4列=32）

warpsPerCTA, # CTA 内的 Warp 排布，如 \[4, 1\]

CTAsPerCGA # 多 CTA 协作时的排布，如 \[1, 1\]

)

这四个参数的乘积，精确定义了 tile 内每个元素归属哪个线程——从 Thread 粒度到 Warp 粒度到 CTA 粒度，层层清晰。

关键改进在 `threadsPerWarp` 参数：它让编译器能够在 Warp 层做出明确决策，确保 Warp 内 32 个线程的数据分布与 TensorCore 的期望格式匹配。

One-Off 决策策略

ML-Triton 的编译策略是"一次性推导全图"——在编译开始时，分析整个计算图的 Layout 需求，做出全局最优的 Layout 分配，而不是逐层默认然后事后补偿。

三级编译流水线

![三级编译流水线](https://mmbiz.qpic.cn/sz_mmbiz_png/8bQpc14UAHzvUjZ8ejUxr4FaWpAYHicUwytfkkG4ZYpibEN8FwLMpOib12B3nUbbZhDENBkJdhoDQc3fEYicSHm9YhViciabslsnT8hD0wicUF0384/640?from=appmsg&wx_fmt=png&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4)

三级编译流水线

1. **分析阶段** ：扫描计算图，识别所有 Layout 约束（TensorCore 指令要求、内存访问模式、转换开销）
2. **求解阶段** ：用约束传播算法推导全局 Layout 方案
3. **代码生成阶段** ：按确定的 Layout 方案生成 PTX 指令，最小化 layout conversion 数量

实测结果

在 Intel PVC GPU 上，ML-Triton 使 Triton 达到硬件理论峰值的 95%+，相比旧 Triton 提升了约 30 个百分点。

**BlockedEncoding 的价值** ：把"猜"变成"算"。通过显式参数化 Warp 层，让编译器在设计阶段就能保证 Layout 与 TensorCore 对齐。

「老许漫谈AIInfra」 · 持续关注 AI 基础设施工程实践

第六章：新问题——N² 爆炸

BlockedEncoding 解决了"猜"的问题，但引入了一个新的工程困境： **Layout 种类爆炸，转换函数维护成本失控** 。

N² 问题的本质

GPU kernel 里有很多不同的操作，每种操作可能需要不同的 Layout：

- GEMM 的矩阵乘法：需要 TensorCore 对齐 Layout
- Softmax 的 reduce：需要 row-major Layout，方便 Warp 内规约
- 转置操作：需要 column-major Layout
- 点乘操作：需要与前一步匹配的 Layout
- ……

假设你有 10 种不同的 Layout，那么理论上就需要 10×10=100 个"A→B"的转换函数。每个转换函数都要手写、测试、优化。

这不是假设。根据论文数据， **ML-Triton 代码库中约 12% 的 bug 来自 layout 转换函数** ——这些函数逻辑复杂、极易出错，而且每次支持新硬件或新操作都要新增。

![N² 爆炸 vs F₂ 统一方案](https://mmbiz.qpic.cn/sz_mmbiz_png/8bQpc14UAHylabPSQLp5b73GicRoIqCiak879gGExiaJ41uSonLq69cY3WoCIibQD1mp2rkFepDadraOMNScLCcMhZDtibYN5Be2WcuxUl0fuIBM/640?from=appmsg&wx_fmt=png&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)

N² 爆炸 vs F₂ 统一方案

FlashAttention-3 的极端案例

FA3（FlashAttention-3）是当今最重要的 GPU kernel 之一。它的作者在集成到 Triton 时遭遇了 N² 问题的极端版本。

FA3 在 attention score 计算后需要做一个数据重排，以便后续的 softmax 操作能高效使用 Warp Shuffle。这个重排在旧 Layout 体系里没有通用的转换函数，作者 **不得不手写 byte permute** ：直接操作寄存器中的字节顺序，按位置换数据。

这是一段 CUDA 内联汇编级别的代码，普通工程师几乎无法独立写出。而它的存在，仅仅是因为 Layout 系统不够通用。

**N² 爆炸的教训** ：当 Layout 种类增加时，手工维护的转换函数以平方速度增长。这不是可持续的工程路径。需要一个统一的数学框架来自动推导任意两种 Layout 之间的转换。

「老许漫谈AIInfra」 · 持续关注 AI 基础设施工程实践

第七章：终极解法——F₂ 线性代数（Linear Layouts）

Linear Layouts（arXiv 2505.23819）提出了根本性的解答： **把 Layout 变成 GF(2) 上的矩阵，把 Layout 转换变成矩阵乘法** 。

7.1 核心洞见：GPU 编号天然是二进制的

在 GPU 上，一切编号都是整数，而整数天然可以用二进制表示：

- Lane ID：0-31，5位二进制（10000 → 11111 覆盖 32 个 thread）
- Warp ID：0-3，2位二进制
- 矩阵行坐标：0-255，8位二进制
- 矩阵列坐标：0-255，8位二进制

Layout 的本质，就是从"硬件编号空间"（thread lane, warp id）到"数据坐标空间"（行，列）的映射。

这两个空间都是二进制整数。它们之间的映射，能不能用二进制运算描述？

**可以。而且完美契合 GF(2) 域的代数结构。**

7.2 F₂ 域基础：XOR=加法，AND=乘法

GF(2)（也写作 F₂）是只有 0 和 1 的域：

加法：0+0=0, 0+1=1, 1+0=1, 1+1=0 → XOR

乘法：0×0=0, 0×1=0, 1×0=0, 1×1=1 → AND

在 GF(2) 上，矩阵乘法就是：用 AND 代替普通乘法，用 XOR 代替普通加法。

这对 GPU 编程来说是天然选择：

1. **XOR 和 AND 是 GPU 最快的指令** ，一个时钟周期完成
2. **二进制向量完美表示线程编号** ，5位表示 lane 0-31
3. **矩阵乘法有逆运算** ，Layout 转换可以自动求解

7.3 Layout 作为 F₂ 矩阵

![F₂ 矩阵 Layout 编码推导](https://mmbiz.qpic.cn/mmbiz_png/8bQpc14UAHyWVIwNnCNj26c9nEKcUicHVoINEfOgzI6eBvFNsDD3N91gzHCBbetb946jJmANg1Bqqjj6zTObjbgUcgclmEs79ly6djvXGzxs/640?from=appmsg&wx_fmt=png&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=6)

F₂ 矩阵 Layout 编码推导

Linear Layout 的核心定义：

**Layout M 是一个 F₂ 矩阵，列向量描述每个"资源 bit"对"tensor 坐标 bit"的贡献。**

举个例子。假设我们有 32 个 lane（5 bit），2 个 warp（1 bit），需要映射到一个 32×4 的矩阵（5 bit 行，2 bit 列）。

Layout 矩阵 M 就是一个 6×7 的 F₂ 矩阵：

- 行：输入 bit（lane 的 5 bit + warp 的 1 bit = 6 bit）
- 列：输出 bit（行坐标的 5 bit + 列坐标的 2 bit = 7 bit）

给定 lane=5（二进制 00101），warp=1（二进制 1），把这 6 bit 拼成向量 \[0,0,1,0,1,1\]，乘以 M 矩阵（XOR-AND 运算），得到目标坐标的 7 bit 表示，拆开就是（行，列）坐标。

这就是"谁负责哪块数据"的完整数学描述。

7.4 三件事自动化

有了这个框架，三件之前需要手工处理的事，变成了矩阵运算：

**Layout 转换 = 矩阵乘法**

![Layout 转换 = F₂ 矩阵乘法](https://mmbiz.qpic.cn/sz_mmbiz_png/8bQpc14UAHxicGc64I0EJo6o05E0vMN1FM12WE2LDISk6hQxFkib0B4ETZ5nQkgkK4SReeicBbwCWZOib2Mb4qu9jpfGNScNbh3vAiaYfxhTZLCg/640?from=appmsg&wx_fmt=png&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=7)

Layout 转换 = F₂ 矩阵乘法

从 Layout A（矩阵 M₁）转换到 Layout B（矩阵 M₂），只需：

转换矩阵 = M₂ × M₁⁻¹

100 个手写转换函数，变成一个矩阵求逆加一个矩阵乘法。M₁⁻¹ 在 GF(2) 上有解析公式，计算极快。

**自动 Swizzling = 行置换**

Swizzling 是一种内存地址扰乱技术，用于避免 Shared Memory bank conflict。在 Linear Layout 框架里，Swizzling 就是对 F₂ 矩阵的行做置换——可以系统枚举所有无冲突的 Swizzling 方案，自动选最优。

**自动 Warp Shuffle**

Warp Shuffle 需要知道 Warp 内哪个 lane 持有目标数据。在 Linear Layout 框架里，这个查询变成：在矩阵 M 里找特定行的 XOR 组合，是纯代数计算，编译时完成。

7.5 BlockedEncoding 是 F₂ 矩阵的特例

这里有一个优雅的联系： **第五章的 BlockedEncoding，可以直接编码为特定形式的 F₂ 矩阵** 。

BlockedEncoding 的四个参数（sizePerThread, threadsPerWarp, warpsPerCTA, CTAsPerCGA）定义了一种严格的块状分割，这种分割对应 F₂ 矩阵的一个特定子集——块对角结构。

这意味着 ML-Triton 和 Linear Layouts 这两篇论文，其实是同一条道路的两个阶段：前者发现了问题，提出了参数化解法；后者把参数化解法统一进一个更完备的代数框架，同时解决了 N² 爆炸。

![F₂ 矩阵描述 Layout 原理](https://mmbiz.qpic.cn/mmbiz_png/8bQpc14UAHwiayXurhe0SWibI0mPZeUSVQaXmw0u79LJRDs1KkFRvopq2a2DXjWTjicjqjKjDC38WdzOKiaIvheCdrJ7X6syaX7JRlnJF1fZqaI/640?from=appmsg&wx_fmt=png&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=8)

F₂ 矩阵描述 Layout 原理

**Linear Layouts 的核心价值** ：Layout 是 F₂ 矩阵，Layout 转换是矩阵乘法。这不是一种优化，而是把"靠猜+手写"换成了"数学推导+自动化"。N² 的手工维护代价降为 O(1) 的矩阵运算。

「老许漫谈AIInfra」 · 持续关注 AI 基础设施工程实践

第八章：结果与总结

数字说话

**ML-Triton（BlockedEncoding，arXiv 2503.14985）**

| 测试场景 | 旧 Triton | ML-Triton | 提升 |
| --- | --- | --- | --- |
| GEMM（Intel PVC） | ~65% peak | 95%+ peak | +30pp |
| 一般 kernel（NVIDIA A100） | ~70% peak | ~88% peak | +18pp |

来源：\[arXiv 2503.14985\](https://arxiv.org/abs/2503.14985)

**Linear Layouts（F₂ 线性代数，arXiv 2505.23819）**

![Linear Layouts 性能数据](https://mmbiz.qpic.cn/sz_mmbiz_png/8bQpc14UAHxUdfJa5f0qjO13dYP0MeqatgTIhO58vjac0jxbVajIGexBMapwoD0YfoicwxjLxMpmTUjAvBzbGwzcnicPwBMbeD6k9ndP9LibTg/640?from=appmsg&wx_fmt=png&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=9)

Linear Layouts 性能数据

| 测试场景 | 基线 | Linear Layouts | 说明 |
| --- | --- | --- | --- |
| GEMM（A100） | CUDA 60-80% | CUDA 95%+ | Layout 转换消除 |
| FA3 集成 | 需手写 byte permute | 自动推导 | N² 问题消除 |
| 转换函数 bug 率 | 12% of bugs | 趋近于 0 | 代码库 bug 减少 |

来源：\[arXiv 2505.23819\](https://arxiv.org/abs/2505.23819)

![GEMM 性能对比](https://mmbiz.qpic.cn/mmbiz_png/8bQpc14UAHyO0ibd978Cickp3nXOfPUicsLKKoPSG4NcKpZDqs90YxIJOPELaKSRAAXmDN5bteWY5IC3cPocTicUb2FBZGge4cTn9HickibM7tg08/640?from=appmsg&wx_fmt=png&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=10)

GEMM 性能对比

三句话总结

这两篇论文合在一起，描述了一条清晰的演进路径：

**问题的根源** ：旧 Triton 在 CTA→Thread 降级时跳过了 Warp 层，无法保证 TensorCore 对齐，系统性地产生坏 Layout，性能损失 20-40%。

**第一代修复** ：BlockedEncoding 用四参数显式表达 Warp 层，把"猜"变成"设计"，在 Intel PVC 上达到 95%+ 效率。但 Layout 种类爆炸导致 N² 个手写转换函数。

**终极解法** ：Linear Layouts 把 Layout 表示为 GF(2) 矩阵，把转换变成矩阵乘法，自动化 Swizzling 和 Warp Shuffle，同时统一了 BlockedEncoding 的表达能力。一个框架，解决所有 Layout 问题。

*参考资料：* *\- \[ML-Triton: Compiler Support for Efficient Machine Learning\](https://arxiv.org/abs/2503.14985)* *\- \[Linear Layouts: A Mathematical Framework for GPU Data Layout\](https://arxiv.org/abs/2505.23819)*

「老许漫谈AIInfra」 · 持续关注 AI 基础设施工程实践

**微信扫一扫赞赏作者**