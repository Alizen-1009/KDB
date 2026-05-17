---
title: "陈巍：DeepSeek 开源Day（1）-FlashMLA 深入分析（收录于：DeepSeek技术详解系列）"
source: "https://zhuanlan.zhihu.com/p/26031898869"
author:
  - "[[陈巍 博士高级职称（清华/中科院） 大模型/存算一体/GPGPU]]"
published:
created: 2026-05-17
description: "本文将浅入深出的分析DeepSeek新开源的FlashMLA原理、架构，解读FlashMLA的贡献。 2月24日，DeepSeek启动“开源周”，首个开源的代码库为FlashMLA。DeepSeek这种挤牙膏式的宣推手段也是很有意思，看来梁文锋团队不…"
tags:
  - "clippings"
---
[收录于 · DeepSeek技术详解系列](https://www.zhihu.com/column/c_1878136088322785280)

138 人赞同了该文章

目录

收起

1 FlashMLA简介

2 FlashMLA的关键技术与未来优化

3 从Memory Bound到Flash Attention和MLA

3.1 Memory Bound与I0-Awareness

3.2 Flash Attention

3.3 MLA

4 FlashMLA微架构分析

4.1 FlashMLA核心技术

4.2 FlashMLA代码结构

5 FlashMLA的价值与意义

延申阅读

本文将浅入深出的分析DeepSeek新开源的 [FlashMLA](https://zhida.zhihu.com/search?content_id=254228681&content_type=Article&match_order=1&q=FlashMLA&zhida_source=entity) 原理、架构，解读FlashMLA的贡献。

2月24日，DeepSeek启动“开源周”，首个开源的代码库为FlashMLA。DeepSeek这种挤牙膏式的宣推手段也是很有意思，看来梁文锋团队不仅仅是技术派，也擅长玩技术流量IP。

![](https://pic4.zhimg.com/v2-93a756d88b3a9e3645dd6b86db44bd65_1440w.jpg)

## 1 FlashMLA简介

FlashMLA是由 depseek-ai （深度求索）开发的一个开源项目，针对 [Hopper架构](https://zhida.zhihu.com/search?content_id=254228681&content_type=Article&match_order=1&q=Hopper%E6%9E%B6%E6%9E%84&zhida_source=entity) GPU（例如H100或H800）的高效的MLA推断（Inference）解码内核，旨在加速MLA机制的计算，特别适用于DeepSeek 系列模型（如 DeepSeek-V2、V3 和 R1）。

![](https://pic1.zhimg.com/v2-08c49d93419f3ecda3fece44bede1756_1440w.jpg)

DeepSeek V3/R1介绍

其中MLA是DeekSeek研发的多头潜注意力（ [Multi-head Latent Attention](https://zhida.zhihu.com/search?content_id=254228681&content_type=Article&match_order=1&q=Multi-head+Latent+Attention&zhida_source=entity) ）机制。通过低秩矩阵压缩KV Cache（键值缓存），减少内存占用，同时提升模型性能。

![](https://pic1.zhimg.com/v2-5e77df2629c8a00ead5a7511dc3d46c4_1440w.jpg)

FlashMLA借鉴 [FlashAttention](https://zhida.zhihu.com/search?content_id=254228681&content_type=Article&match_order=1&q=FlashAttention&zhida_source=entity) 分块Tiling 和显存优化的思想。通过以算代存减少对于显存带宽的要求，提升计算性能。FlashMLA的构建基于 [Cutlass](https://zhida.zhihu.com/search?content_id=254228681&content_type=Article&match_order=1&q=Cutlass&zhida_source=entity) 和CUDA体系。

FlashMLA 主要用于大模型推断/推理（Inference），特别是在需要处理长序列的场景中，如聊天机器人或代码生成工具。通过优化GPU利用率，解决大模型在推理阶段的显存瓶颈问题。

![](https://pic3.zhimg.com/v2-cae4dad2a7eee0f07bb465323b824272_1440w.jpg)

MLA

## 2 FlashMLA的关键技术与未来优化

FlashMLA是MLA技术和Flash Attention技术的结合，可以认为是Flash Attention的MLA版本。

**FlashMLA具有以下关键特征：**

1）Flash MLA支持变长序列和分页KV缓存。

2）基于 [BF16](https://zhida.zhihu.com/search?content_id=254228681&content_type=Article&match_order=1&q=BF16&zhida_source=entity) 格式（FP16也发布了）和至少12.3以上的CUDA。

3）支持Hopper架构的TMA优化。

4）可显著提升KV Cache性能和GPU计算性能。在 [H800 SXM5](https://zhida.zhihu.com/search?content_id=254228681&content_type=Article&match_order=1&q=H800+SXM5&zhida_source=entity) 上，可达 3000 GB/s的计算带宽（接近3.35TB/s的理论峰值）。

5）开源版本暂不支持反向传播计算。

6）使用MIT 许可证便于社区协作。

Hopper GPU架构可以看我们文章：

[zhuanlan.zhihu.com/p/48](https://zhuanlan.zhihu.com/p/487250706)

**FlashMLA未来可能的优化方向包括** ：

1）通过PTX编程进一步提高细粒度性能

2）探索FP8数据格式支持（需Hopper架构或更先进的TensorCore）

![](https://pic3.zhimg.com/v2-85f52a3dbfbb769f44a05269b29c84fe_1440w.jpg)

FlashMLA与其他相近计算方法对比

## 3 从Memory Bound到Flash Attention和MLA

### 3.1 Memory Bound与I0-Awareness

![](https://pica.zhimg.com/v2-7f89ad7880a540cec139be29063fb0bc_1440w.jpg)

传统计算芯片的分层存储架构（来源：互联网）

在传统的GPU和AI芯片中，存储架构分为不同的层次。一般来说内部的SRAM最快，外部的HBM或DRAM速度比SRAM慢很多。

**1）Die内存储**:主要用于缓存(Cache)及少量特殊存储单元(例如texture)，其特点是存储容量小但带宽大。SRAM就属于常见的Die（晶片）内存储，存储容量一般只有20-160MB，但是带宽可以达到甚至超过19TB/S。

**2）Die外存储** ：主要用于全局存储，即我们常说的显存，其特点是存储容量大但带宽小。HBM就属于常见的Die（晶片）外存储，存储容量一般是40GB以上，但带宽相比于SRAM小得多。

![](https://picx.zhimg.com/v2-d8d5f1ad50c4a981399a8e2e384420a7_1440w.jpg)

KV缓存

对于Transformer类的大模型来说，由于KV Cache巨大，很难直接放在Cache里，需要放在HBM或GDDR上，并在计算过程中频繁挪动KV数据。（另外有一些Transformer Free的结构就不需要反复挪动KV数据，还未成为主流技术）这时就会出现Memory Bound（存储限制）的情况，极大影响了KV Cache的吞吐带宽和大模型的计算速度。

在Flash Attention之前，也出现过一些加速Transformer计算的方法，着眼点是减少计算量，例如稀疏Attention做近似计算。但对于Attention来说，GPU计算瓶颈不在运算能力，而是在存储的读写速度上。Flash Attention吸取了这些加速方法的教训，改为通过降低对显存(HBM或GDDR)的访问次数来加快整体性能，这类方法又被称为I0-Awareness（IO优先或存储优先）

### 3.2 Flash Attention

FlashAttention 是一种高效的注意力机制优化技术，由斯坦福等大学的研究团队开发，最早于 2022 年提出，并在后续版本（如 FlashAttention-2、FlashAttention-3）中不断完善。FlashAttention旨在解决传统 Transformer 模型中多头注意力（Multi-head Attention, MHA）的计算和显存瓶颈，尤其是在处理长序列时。FlashAttention通过重新设计注意力计算方式，显著提升性能，同时保持与标准注意力机制相同的数学输出，使其成为近年来生成式AI和大模型领域的重要技术。FlashAttention拥有比PyTorch（当时的版本）标准注意力快2~4倍的运行速度，所需内存还减少了5~20倍。

![](https://pic2.zhimg.com/v2-536c2c58172c1474378561696da6d04b_1440w.jpg)

Flash Attention技术（来源：互联网）

**Flash Attention** 专注于标准多头注意力的高效实现，通过减少访问显存次数，优化并行度提升计算性能，但并不直接兼容MLA。

传统MHA 的计算复杂度为 O(n²)（n 为序列长度），并且需要存储大量的中间结果，这在长序列任务中会导致严重的显存压力和计算延迟。FlashAttention 的核心理念是避免显式计算和存储完整的注意力矩阵，而是通过 **分块计算（tiling）** 和 **融合操作** ，将注意力计算优化为接近O(n)的复杂度，同时大幅减少GPU内存访问。

**1）分块处理**: 将输入序列分割成小块（tiles），逐块计算注意力，避免一次性加载整个矩阵。

**2）显存优化**: 通过在线计算 softmax 和融合操作，减少中间结果的存储需求。

**3）硬件架构友好**: 充分利用GPU高速内存（如共享缓存）和并行计算能力。

### 3.3 MLA

DeepSeek使用的Multi-Head Latent Attention技术可大大节省KV缓存，从而显著降低了计算成本。

MLA的本质是对KV的有损压缩，提高存储信息密度的同时尽可能保留关键细节。该技术首次在DeepSeek-V2中引入，与分组查询和多查询注意力等方法相比，MLA是目前开源模型里显著减小KV 缓存大小的最佳方法。

MLA的方法是将KV矩阵转换为低秩形式：将原矩阵表示为两个较小矩阵（相当于潜向量）的乘积，在推断过程中，仅缓存潜向量，而不缓存完整的键KV。这规避了分组查询注意力和多查询注意力的查询的信息损失，从而在降低KV缓存的前提下获得更好的性能。

![](https://pic4.zhimg.com/v2-c316f4a4c408788953b1598df513fbcd_1440w.jpg)

矩阵的低秩近似（来源：互联网）

MLA随好，但明显没有针对现代加速框架的FlashAttention或PageAttention解决方案。这也使得DeepSeek R1在实际部署时需要单独优化KV吞吐性能。

## 4 FlashMLA微架构分析

### 4.1 FlashMLA核心技术

Flash MLA 的核心是高效的 MLA 解码内核，关键技术包括：

**1）低秩矩阵压缩** ：MLA 使用低秩矩阵，将KV缓存压缩为潜向量，减少内存占用。通过解压潜向量生成独特的KV头（KV Head）。

**2）针对GPU 优化** ：FlashMLA针对Hopper GPU 的Tensor Core进行youh优化，实现了可达3000 GB/s 的显存带宽和 580 TFLOPS 的计算性能（H800 SXM5 配置）。使用了SM90的关键特性GMMA、namedbarrier同步、cp.async。

**3）Row-wise/Block-wise优化** ：细粒度划分，在shared memory中原位处理计算，减少了额外的中间计算过程的显存占用，减少显存访问次数。

**4）Split-KV 分块处理** ：将KV拆分给多个SM（Stream Multiprocessor）处理（或者多次迭代），然后在局部把partial计算结果合并。

1. **变长序列支持** ：通过 tile\_scheduler\_metadata 和 num\_splits 参数，，FlashMLA 支持变长序列的并行处理，以缓解负载不均衡问题。

### 4.2 FlashMLA代码结构

FlashMLA 提供了Python 接口，如 get\_mla\_metadata 获取 MLA（Multi-Head Linear Attention）的meta数据；flash\_mla\_with\_kvcache，用于获取键值缓存（KV Cache）和执行注意力（FlashMLA）计算。

![](https://pic4.zhimg.com/v2-cd8d4456bcadb517582d8de3d402e3d1_1440w.jpg)

主要代码结构如下：（需要注意代码库还在不断更新，后面又添加了benchmark文件夹）

**1）flash\_mla/ 目录**

- **主要文件**: flash\_mla\_interface.py
- **作用**: Python 接口层，封装了底层 C++/CUDA实现，以便将 FlashMLA 集成到 PyTorch 工作流中。  
	这部分代码定义了flash\_mla\_with\_kvcache等函数，用于执行带 KV 缓存的MLA 前向计算。  
	参数包括查询向量（q）、键值缓存（kvcache）、块表（block\_table）、序列长度（cache\_seqlens）等。  
	**2）benchmark/ 目录**
- **主要文件**: bench\_flash\_mla.py
- **作用**: 用于对不同的多头注意力（Multi-Head Attention, MLA）实现进行基准测试和性能比较。

run\_torch\_mla：使用PyTorch实现的MLA基准测试。

run\_flash\_mla：使用flash\_mla库实现的MLA基准测试。

run\_flash\_infer：使用flashinfer库实现的MLA基准测试。

run\_flash\_mla\_triton：使用Triton实现的MLA基准测试

**2）setup.py**

- **作用**: 构建脚本，用于编译和安装 FlashMLA 模块。

**3）csrc/ 目录**

- **文件**:  
	flash\_api.cpp: C++ 接口，连接 Python 和 CUDA。  
	flash\_fwd\_mla\_bf16\_sm90.cu: 核心CUDA内核BF16支持，针对 Hopper 架构（SM90）优化。  
	flash\_fwd\_mla\_fp16\_sm90.cu：核心 CUDA 内核FP16支持。  
	flash\_mla.h, softmax.h, utils.h 等: 提供辅助函数和数据结构。
- **作用**: 实现了FlashMLA底层的CUDA实现和性能优化。

这部分代码使用 BF16（Brain Float 16）数据类型，以保障Attention计算精度。同时结合FlashAttention 2/3和Cutlass库，以实现高效注意力机制。

**flash\_mla.h：定义接口函数：**

- get\_mla\_metadata(num\_heads, head\_dim, num\_kv\_heads, kv\_head\_dim, block\_size, dtype)：获取 MLA meta数据。
- flash\_mla\_with\_kvcache(q, k, v, kvcache, seqlen, metadata, causal=True)：执行注意力计算。  
	**softmax.h：行softmax的计算（速度瓶颈）**  
	**named\_barrier.h：SM90 NamedBarrier枚举同步。**  
	**flash\_fwd\_mla\_metadata.cu：定义了一个用于获取MLA meta数据的内核函数和一个调用该内核函数的主函数。**
- get\_mla\_metadata\_func：用于调用内核函数。使用1个线程块，每个线程块包含32个线程。并检查内核函数的启动是否成功。

**Paged KV Cache 实现：**

**显存分块** ：以64为单位（block\_size = 64），通过block\_table维护逻辑块到物理显存的映射。

**流水线** ：分离数据加载与计算阶段，通过cp.async实现异步数据预取。

flash\_fwd\_splitkv\_mla\_kernel：用于并行计算Flash Attention的前向传播。

flash\_fwd\_splitkv\_mla\_combine\_kernel：用于合并多个分割的计算结果。

## 5 FlashMLA的价值与意义

FlashMLA 是 DeepSeek 团队在 AI 性能优化领域的重要成果，实现了在英伟达Hopper架构GPU的高效Inference。其价值在于：

1）通过开源鼓励开发者优化或适配其他硬件（如AMD GPU和其他AI芯片）。

2）鼓励开发者实现与现有加速框架（如 vLLM、SGLang等）的集成。

强烈建议OpenAI把域名送给DeepSeek。

## 延申阅读

**Day5陈巍：DeepSeek 开源Day（5）3FS&smallpond深入分析**

[zhuanlan.zhihu.com/p/26](https://zhuanlan.zhihu.com/p/26958884790)

**Day4陈巍：DeepSeek 开源Day（4）DualPipe&EPLB深入分析**

[zhuanlan.zhihu.com/p/26](https://zhuanlan.zhihu.com/p/26744800336)

**Day3陈巍：DeepSeek 开源Day（3）DualGEMM深入分析**

[zhuanlan.zhihu.com/p/26](https://zhuanlan.zhihu.com/p/26437292382)

**Day2 陈巍：DeepSeek Day 2 DeepEP技术深入分析**

[zhuanlan.zhihu.com/p/26](https://zhuanlan.zhihu.com/p/26204046487)

**Day1 陈巍：DeepSeek最新开源FlashMLA 技术深入分析**

[zhuanlan.zhihu.com/p/26](https://zhuanlan.zhihu.com/p/26031898869)

**陈巍：DeepSeek V3/R1的架构与训练技术2万字长文分析（下）**

[zhuanlan.zhihu.com/p/21](https://zhuanlan.zhihu.com/p/21755758234)

**陈巍：DeepSeek V3/R1的架构与训练技术2万字长文分析（上）**

[zhuanlan.zhihu.com/p/21](https://zhuanlan.zhihu.com/p/21208287743)

![](https://pic3.zhimg.com/v2-21b8b27b178c627444c549aee7063494_1440w.jpg)

参与DeepSeek与MoE讨论：

[gitlab.com/williamchent](https://link.zhihu.com/?target=https%3A//gitlab.com/williamchenth/grouplink/-/blob/main/MoEgroup.PNG)

参与deepseek一体机本地部署生态交流：

[gitlab.com/williamchent](https://link.zhihu.com/?target=https%3A//gitlab.com/williamchenth/grouplink/-/blob/main/deepseekallinoneGroup.PNG)

大模型（含deepseek）与AI芯片（含GPGPU）关键知识地图：

[zhuanlan.zhihu.com/p/25](https://zhuanlan.zhihu.com/p/25529708891)

deepseek-MoE资源汇总：

[github.com/chenweiphd/D](https://link.zhihu.com/?target=https%3A//github.com/chenweiphd/DeepSeek-MoE-ResourceMap)

还没有人送礼物，鼓励一下作者吧

编辑于 2025-07-02 15:23・广东