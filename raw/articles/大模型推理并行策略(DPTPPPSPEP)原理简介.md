---
title: "大模型推理并行策略(DP/TP/PP/SP/EP)原理简介"
source: "https://zhuanlan.zhihu.com/p/2003423046342554380"
author:
  - "[[kaiyuan​新知答主]]"
published:
created: 2026-04-29
description: "在大模型推理部署中，由于模型参数量常超出单GPU显存容量，或单GPU算力无法满足推理性能要求，通常需要借助并行策略来解决这类问题。又因计算流程存在差异，推理与训练的并行策略实现并不相同。本文旨在帮助读者快…"
tags:
  - "clippings"
---
[收录于 · LLM推理基础与框架](https://www.zhihu.com/column/c_1916901019268391457)

165 人赞同了该文章

目录

收起

1 DP策略

1.1 基本原理

1.2 代码演示

2 TP策略

2.1 基本原理

2.2 代码演示

3SP策略

3.1 基本原理

3.2 代码演示

3.3 SP与其它策略结合

4 PP策略

4.1 基本原理

4.2 代码演示

5 EP策略

5.1 基本原理

6 其它策略

6.1 CP策略

6.2 Ulysses并行

总结

在大模型推理部署中，由于模型参数量常超出单GPU显存容量，或单GPU算力无法满足推理性能要求，通常需要借助并行策略来解决这类问题。又因计算流程存在差异，推理与训练的并行策略实现并不相同。本文旨在帮助读者快速理解常见并行策略的基本原理。

推理主要应用的并行方式包括： [数据并行](https://zhida.zhihu.com/search?content_id=270099304&content_type=Article&match_order=1&q=%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3Nzc2MjU5MzAsInEiOiLmlbDmja7lubbooYwiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzAwOTkzMDQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.4BYxt9Wdr9JYM_mlNpSq5im9ZVHVwSOR3Iqqw97lOAU&zhida_source=entity) （DP）、 [序列并行](https://zhida.zhihu.com/search?content_id=270099304&content_type=Article&match_order=1&q=%E5%BA%8F%E5%88%97%E5%B9%B6%E8%A1%8C&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3Nzc2MjU5MzAsInEiOiLluo_liJflubbooYwiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzAwOTkzMDQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.wsTaZnFjZD15nvN3WuLwJDAJkuE8KJZFwAOSNWyrN-8&zhida_source=entity) (SP/CP)、 [张量并行](https://zhida.zhihu.com/search?content_id=270099304&content_type=Article&match_order=1&q=%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3Nzc2MjU5MzAsInEiOiLlvKDph4_lubbooYwiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzAwOTkzMDQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.Phsf24om6Ai6jRmpcZZWeQmeF4aoI_7s-3N1aDVgtyU&zhida_source=entity) （TP）、 [层并行](https://zhida.zhihu.com/search?content_id=270099304&content_type=Article&match_order=1&q=%E5%B1%82%E5%B9%B6%E8%A1%8C&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3Nzc2MjU5MzAsInEiOiLlsYLlubbooYwiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzAwOTkzMDQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.15bY1MOOBLTv1LlRcZMB4LNdnjQbY78rwbi4wWEIe0U&zhida_source=entity) （PP）。可根据输入激活值的切分维度对应不同的并行策略，一般切batch为DP、切序列为SP/CP、切隐藏层尺寸为TP。

![](https://pic3.zhimg.com/v2-fb8c441fed48d5ea14902b9b62475668_1440w.jpg)

## 1 DP策略

### 1.1 基本原理

DP(Data Parallel)数据并行，是解决数据并发量大时使用的策略，DP方法在不同的GPU运行LLM模型的多个副本，并在每个模型副本上独立处理用户请求组。其原理与开多个推理实例并发处理一样，不同的是，开DP是多个模型副本共用一个推理实例，由推理实例中的调度器负责分配请求给不同DP的模型副本。

![动图封面](https://pic3.zhimg.com/v2-b447569960a30348c83cc52dc5b4af78_b.jpg)

### 1.2 代码演示

通过多线程处理模拟多GPU，可以构造两个计算场景：

- **场景1** ：一个模型副本，我们用一个线程来运行这个模型，然后有4个数据任务，我们用一个线程池（4个线程）来同时发送数据给这个模型，但是模型处理是串行的，所以我们可以在模型内部加锁，使得同时只能有一个线程（即一个数据）被处理。
- **场景2** ：四个模型副本，每个模型副本在一个线程中，然后有4个数据，我们同样用4个线程来发送数据，但是每个数据发送给不同的模型副本，这样就能并行处理。

**代码位置** ： [InfraTech/llm\_infer/parallel\_strategies.ipynb Case1](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/InfraTech/blob/main/llm_infer/parallel_strategies.ipynb) [^1]

![](https://pic1.zhimg.com/v2-f4c9f00e7b025f32baf2d6f55416c4ce_1440w.jpg)

计算示例

注意，这里只是用线程模拟计算，不同的计算设备得到速度不一样。

## 2 TP策略

### 2.1 基本原理

Tensor Parallelism (TP)张量并行，将模型的每一层分割到不同GPU上执行，用户请求（输入数据）会在GPU间流转，每个GPU计算的部分结果最终重新组合为完整输出。TP的计算理论基础是矩阵的分块运算，该运算不会改变最终计算结果。

![](https://pic3.zhimg.com/v2-b1238e1dbb92dfa6350867bd92d1acc0_1440w.jpg)

矩阵列切(column split)

![](https://pica.zhimg.com/v2-3d1b60a6c44f5c9e9ef59c472a592836_1440w.jpg)

矩阵行切(row split)

TP在LLM推理中应用得比较广泛，其主要作用是降低单卡显存消耗以及计算量。

![动图封面](https://pic2.zhimg.com/v2-0ab61ebdd228f3ebe42f4e556b6c8b57_b.jpg)

### 2.2 代码演示

演示张量并行如何通过拆分大矩阵运算到多个计算单元的过程。选择大矩阵（如1024×1024）模拟真实计算场景。将输入矩阵按列分块（column-wise/column-split），计算分配：每个线程处理矩阵A与B的一个列块的乘积。最后，将所有线程的计算结果拼接成完整输出矩阵。对比机制:

- 基准测试：使用标准numpy矩阵乘法作为性能基准
- 并行实现：使用多线程模拟多设备并行计算
- 结果验证：确保并行计算与串行计算数值结果一致

性能对比：对比元计算与TP的速度差异，机器不同计算速度不一样，性能对比数据仅供参考。

**代码位置：** [InfraTech/llm\_infer/parallel\_strategies.ipynb Case2](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/InfraTech/blob/main/llm_infer/parallel_strategies.ipynb) [^1] ，运行之后可获得速度对比差异：

![](https://pic3.zhimg.com/v2-c9d1430d591f5624c7eafe2937988832_1440w.jpg)

## 3SP策略

### 3.1 基本原理

SP(Seqeunce Parallel)序列并行是将长序列切分成多个片段，分配到不同GPU设备上并行处理的模型并行策略。示意图如下：

![](https://picx.zhimg.com/v2-1beca1fadb3b0602ccab0db991a5221d_1440w.jpg)

### 3.2 代码演示

使用线程模拟多设备，并且使用简单的全连接层。对比：

1. 不切序列：整个序列数据通过一个完整的模型（多个层）进行计算。
2. 切序列（序列并行）：将序列分成多个部分，每个部分通过一个设备（用线程模拟）上的子模型计算，然后将结果合并。

步骤：

- 定义模型
- 生成输入数据
- 运行不切序列版本
- 运行切序列版本（序列并行）
- 比较结果和时间

代码位置： [InfraTech/llm\_infer/parallel\_strategies.ipynb Case3](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/InfraTech/blob/main/llm_infer/parallel_strategies.ipynb) [^1]

![](https://pic2.zhimg.com/v2-b6eaf8722c7ad64086fbb08bf3edbbd7_1440w.jpg)

### 3.3 SP与其它策略结合

Megatron中TP与SP结合的例子：

![](https://picx.zhimg.com/v2-b55ead91f48b071763e9ffb42409e291_1440w.jpg)

megatron并行示例

负载均衡中SP与DP结合案例：

![](https://picx.zhimg.com/v2-4fe80275a8466dcb3ad25fee01a37991_1440w.jpg)

## 4 PP策略

### 4.1 基本原理

PP(Pipeline Parallel)并行是将模型按层拆分到不同设备，数据以流水线方式在不同设备间顺序流动处理。PP最先是在训练中广泛使用(Megatron2 [^2])。PP前向与后向计算中会出现空泡，训练中需要考虑空泡的消除。

![](https://pic1.zhimg.com/v2-0661403a1b35fcfa0aa4049dc0d7d0d0_1440w.jpg)

在推理任务中，流水线并行（PP）虽然仅涉及前向传播，但其实际应用场景相对有限，通常仅在GPU显存确实无法容纳相应的模型权重时才会被采用。

![动图封面](https://pic2.zhimg.com/v2-c3ea4bf42e1dc56b1f03aa6ee39b1bb9_b.jpg)

### 4.2 代码演示

构建一个流水线并行的演示：假设模型有两层，我们将这两层分别放在两个线程（或设备）上。流水线并行中，数据被分成多个微批次（micro-batches），每个微批次依次通过模型的各个阶段（层）。在这个例子中，有两个阶段（两个线程），每个线程处理模型的一层。模拟：将一个批次的数据分成两个微批次，然后通过流水线的方式处理。

代码位置： [InfraTech/llm\_infer/parallel\_strategies.ipynb Case4](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/InfraTech/blob/main/llm_infer/parallel_strategies.ipynb) [^1]

![](https://pic3.zhimg.com/v2-ec2efb1a56c23666a67c8bf9bef2d55a_1440w.jpg)

## 5 EP策略

### 5.1 基本原理

EP（Expert Parallel）是MoE模型中的并行策略，将不同专家网络分配到不同GPU上。每个GPU只存储部分专家参数，输入数据根据路由机制被分发到对应专家GPU计算，最后汇总结果。这显著扩展了模型总参数量，同时控制单个GPU内存占用，适用于超大稀疏模型训练。

![动图封面](https://pic2.zhimg.com/v2-784b53b00922f07b5dd4c2edb62d5a5d_b.jpg)

当前EP与DP的结合常见场景，Attention使用DP、FFN使用EP。

![](https://picx.zhimg.com/v2-ea4e573ad28ec6b0ff6fda1055444e6d_1440w.jpg)

EP切分会带来负载不均的问题，可通过EPLB解决，参考《 [MoE并行负载均衡](https://zhuanlan.zhihu.com/p/29963005584) 》 [^3] 。

## 6 其它策略

### 6.1 CP策略

CP(Context Parallel) [上下文并行](https://zhida.zhihu.com/search?content_id=270099304&content_type=Article&match_order=1&q=%E4%B8%8A%E4%B8%8B%E6%96%87%E5%B9%B6%E8%A1%8C&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3Nzc2MjU5MzAsInEiOiLkuIrkuIvmloflubbooYwiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzAwOTkzMDQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.ebQbEMuTyiK0K1Nyjl4QuepbAZj9upzzCREP-uw6e5Y&zhida_source=entity) 与序列并行SP均是针对序列维度进行划分的并行策略，且两者最初均在训练并行中被提出。其发展脉络如下：首先出现的是SP策略，它主要解决了模型前向与反向传播中（除Attention计算外）由序列切分带来的内存与计算开销。随后，为了进一步解决Attention模块自身的序列并行问题，Megatron框架中引入了CP策略。二者原理相似，但针对的计算阶段有所不同。具体参考《 [Context Parallelism的原理与代码浅析](https://zhuanlan.zhihu.com/p/698447429) [^4] 》

![](https://pic4.zhimg.com/v2-dd9db7b6656a1246b9c83613eb601ad3_1440w.jpg)

CP计算流程

### 6.2 Ulysses并行

Ulysses的全称是DeepSpeed‑Ulysses，其核心逻辑：开启序列并行后，在多头Attention运算之前，多个GPU设备之间会进行数据交换，使单个GPU能够拥有完整的序列；Attention 计算完成后，再通过集合通信将序列还原为原本被切分的形状。

![](https://pica.zhimg.com/v2-5e376da84e61b159181c39abe87c15a6_1440w.jpg)

在《 [推理Ulysses并行优化与DeepSeekV3/V3.2实践](https://zhuanlan.zhihu.com/p/1995776941110878482) [^5] 》中对Ulysses原理有详细解释，此处不做赘述。

## 总结

在大模型推理场景中，主流推理框架均已支持多种并行策略。每种策略各有其优缺点，旨在解决不同层面的性能与资源瓶颈。

| 并行策略 | 大模型推理中的优缺点 |
| --- | --- |
| 数据并行 (DP) | 优点：处理多请求的基础且常用策略，实现简单。   缺点：内存冗余，每个设备需保存完整模型副本。PD分离架构下，其角色常转化为请求级调度。 |
| 张量并行 (TP) | 优点：最常用于解决单层参数过大问题，能将大层（如FFN）拆分到多卡。   缺点：设备间通信密集，延迟敏感，对设备间互联带宽要求极高。 |
| 流水线并行 (PP) | 优点：可拆分极深模型（层数多）。   缺点：推理中不常用。因天然存在串行依赖，会引入“气泡”，极大降低推理效率和增加延迟。 |
| 序列并行 (SP) | 优点：较常用于处理长序列，可拆分激活内存和注意力计算，是TP的有效补充。   缺点：同样引入额外通信，实现复杂度较高。其思想与CP（上下文并行）类似。 |
| [专家并行](https://zhida.zhihu.com/search?content_id=270099304&content_type=Article&match_order=1&q=%E4%B8%93%E5%AE%B6%E5%B9%B6%E8%A1%8C&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3Nzc2MjU5MzAsInEiOiLkuJPlrrblubbooYwiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzAwOTkzMDQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.6qvWIQhGRex8S0SpZvmul-cbs-2gm6XJ6ojBMkvWJ2A&zhida_source=entity) (EP) | 优点：MoE模型专用的高效扩容方式，仅激活部分参数，扩展性好。   缺点：仅限于MoE架构，需要复杂的负载均衡与路由。 |
| 上下文并行 (CP) | 优点：一种更细粒度的SP，专注于优化KV Cache在长上下文中的内存分布。   缺点：通常不作为独立策略，而是被融入SP的优化中。 |

实际选用时，需结合具体场景综合考虑，例如模型参数量、PD/AF分离需求、硬件拓扑特点等因素 [^6] [^7] 。关于推理并行策略优化推荐进一步阅读：

---

**更多推理知识： [LLM推理知识指南](https://zhuanlan.zhihu.com/p/1954137524881580796)**

> 想深耕AI Infra领域？欢迎访问 **[InfraTech](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/InfraTech)** 库！内容涵盖大模型基础、PyTorch/vLLM/SGLang框架入门、性能加速等核心方向，配套 **[50+知识干货](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/InfraTech)** 及适合初学者的 **[notebook](https://link.zhihu.com/?target=https%3A//github.com/CalvinXKY/InfraTech)** 练习。

**文中不足之处**

[@kaiyuan](https://www.zhihu.com/people/da4e6b50eb50d6f120b604f6cf15b33e)

## 参考

编辑于 2026-03-24 12:03・中国香港[人工智能](https://www.zhihu.com/topic/19551275)[大模型](https://www.zhihu.com/topic/25402720)[深度学习（Deep Learning）](https://www.zhihu.com/topic/19813032)

[^1]: ^ <sup><a href="#ref_1_0">a</a></sup> <sup><a href="#ref_1_1">b</a></sup> <sup><a href="#ref_1_2">c</a></sup> <sup><a href="#ref_1_3">d</a></sup> [https://github.com/CalvinXKY/InfraTech/blob/main/llm\_infer/parallel\_strategies.ipynb](https://github.com/CalvinXKY/InfraTech/blob/main/llm_infer/parallel_strategies.ipynb)

[^2]: [https://arxiv.org/abs/2104.04473](https://arxiv.org/abs/2104.04473)

[^3]: [https://zhuanlan.zhihu.com/p/29963005584](https://zhuanlan.zhihu.com/p/29963005584)

[^4]: [https://zhuanlan.zhihu.com/p/698447429](https://zhuanlan.zhihu.com/p/698447429)

[^5]: [https://zhuanlan.zhihu.com/p/1995776941110878482](https://zhuanlan.zhihu.com/p/1995776941110878482)

[^6]: [https://developer.nvidia.cn/blog/demystifying-ai-inference-deployments-for-trillion-parameter-large-language-models/](https://developer.nvidia.cn/blog/demystifying-ai-inference-deployments-for-trillion-parameter-large-language-models/)

[^7]: [https://arxiv.org/pdf/2101.03961](https://arxiv.org/pdf/2101.03961)