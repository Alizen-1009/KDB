---
title: "Kimi新作《Attention Residuals》：对Transformer中残差结构的调整"
source: "https://zhuanlan.zhihu.com/p/2017346333862831014"
author:
  - "[[Loster分享AI算法，多模态大模型，AIGC，agent内容～]]"
published:
created: 2026-04-22
description: "论文地址： https://arxiv.org/pdf/2603.15031 基于 Kimi 团队（Moonshot AI）于 2026 年 3 月发布的最新论文 《Attention Residuals》（arXiv:2603.15031），以下是对该论文的主要创新点、核心技术和实现细节的深…"
tags:
  - "clippings"
---
论文地址： [arxiv.org/pdf/2603.1503](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2603.15031)

基于 Kimi 团队（Moonshot AI）于 2026 年 3 月发布的最新论文 **[《Attention Residuals》](https://zhida.zhihu.com/search?content_id=271623594&content_type=Article&match_order=1&q=%E3%80%8AAttention+Residuals%E3%80%8B&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzcwMTQxMzAsInEiOiLjgIpBdHRlbnRpb24gUmVzaWR1YWxz44CLIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjcxNjIzNTk0LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.rczlobbhqJ82oS5LMENTdbDL-T4i42k4J1r8XRw71es&zhida_source=entity)** （arXiv:2603.15031），以下是对该论文的主要创新点、核心技术和实现细节的深度分析：

### 论文背景与痛点

在现代大语言模型（LLMs）的架构中，带有 [PreNorm](https://zhida.zhihu.com/search?content_id=271623594&content_type=Article&match_order=1&q=PreNorm&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzcwMTQxMzAsInEiOiJQcmVOb3JtIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjcxNjIzNTk0LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.0g9yrPudEFrh1BEzx6Sssj7JsHQfMeyAQdxgYbTurdY&zhida_source=entity) 的标准残差连接（ [Residual connections](https://zhida.zhihu.com/search?content_id=271623594&content_type=Article&match_order=1&q=Residual+connections&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzcwMTQxMzAsInEiOiJSZXNpZHVhbCBjb25uZWN0aW9ucyIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI3MTYyMzU5NCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.ZkQNCBF9WWGtKWoDWgbdJhCJ9pCTxbM0CVVyx8RFY2o&zhida_source=entity) ）是标配配置。然而，传统的残差网络以 **固定的单位权重（1.0）** 将所有层的输出进行均匀累加。这种均匀的聚合会导致两个严重的结构性问题：

1. **隐状态幅度失控（Uncontrolled hidden-state growth）** ：特征值的幅度随着层数的加深而不受控制地增长。
2. **表征稀释（PreNorm Dilution）** ：由于前面所有层的输出不断累加，任何单一特定层对最终输出的相对贡献会被逐渐稀释。

---

### 一、 主要创新点

1. **提出 Attention Residuals (AttnRes) 替代固定残差** 论文提出了一种全新的残差机制—— **注意力残差（AttnRes）** 。它放弃了传统残差对历史层输出的“无脑等权相加”，转而引入在网络深度方向（Depth-wise）上的 Softmax 注意力机制。该机制允许每一层根据输入内容自适应地、有选择性地聚合前面所有层的特征表示。
2. **提出 [Block AttnRes](https://zhida.zhihu.com/search?content_id=271623594&content_type=Article&match_order=1&q=Block+AttnRes&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzcwMTQxMzAsInEiOiJCbG9jayBBdHRuUmVzIiwiemhpZGFfc291cmNlIjoiZW50aXR5IiwiY29udGVudF9pZCI6MjcxNjIzNTk0LCJjb250ZW50X3R5cGUiOiJBcnRpY2xlIiwibWF0Y2hfb3JkZXIiOjEsInpkX3Rva2VuIjpudWxsfQ.XIktwKcUzfQGuk3cLXGfBJb13WXwcLb8yQTkcmbBsZw&zhida_source=entity) 解决显存与通信扩展性危机** 如果严格让每一层去 Attention 之前所有的层，会导致显存和通信开销随着层数 $L$ 的增加呈 $O \left(\right. L d \left.\right)$ 增长（ $d$ 为隐藏层维度），这在大规模分布式训练中是不可接受的。为此，论文提出了 **分块注意力残差（Block AttnRes）** ，将网络层划分为多个 Block。在块内保持标准残差累加，而只在“块级别（Block-level）”特征间计算 Attention。这一创新将开销大幅降至 $O \left(\right. N d \left.\right)$ （ $N$ 为块数），从而能够在几乎不增加算力负担的前提下保留 Full AttnRes 的绝大部分收益。
![](https://pic3.zhimg.com/v2-e9942e73caa3fc7d35b1cf0441b89b52_1440w.jpg)

---

### 二、 核心技术

1. **[深度方向的伪查询注意力](https://zhida.zhihu.com/search?content_id=271623594&content_type=Article&match_order=1&q=%E6%B7%B1%E5%BA%A6%E6%96%B9%E5%90%91%E7%9A%84%E4%BC%AA%E6%9F%A5%E8%AF%A2%E6%B3%A8%E6%84%8F%E5%8A%9B&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzcwMTQxMzAsInEiOiLmt7HluqbmlrnlkJHnmoTkvKrmn6Xor6Lms6jmhI_lipsiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzE2MjM1OTQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9.7T9aOApeRRVuniCzQc54F371PoWkQpCivLN0OefZFFE&zhida_source=entity) （Depth-Wise Attention with Pseudo-Query）** 在具体的技术设计上，AttnRes 并没有使用极其复杂的注意力计算公式，而是为每一层 $l$ 引入了一个 **单一的可学习伪查询向量（pseudo-query $w_{l} \in \mathbb{R}^{d}$ ）** 。
- 该查询向量 $w_{l}$ 会与前面层的输出计算点积注意力，生成注意力权重 $\alpha_{i \rightarrow l}$ 。
- 尽管 $w_{l}$ 是网络参数，但由于前面层的输出特征包含了当前输入的上下文信息，因此最终生成的权重 $\alpha_{i \rightarrow l}$ 是 **依赖于输入且内容感知（content-aware）** 的。

2**.混合结构的 Block Partitioning（分块划分）** 对于拥有数十上百层的 LLM，研究团队发现将层数平均划分为约 $8$ 个 Block（ $N \approx 8$ ）即可达到理想效果。

- **块内（Intra-block）** ：执行标准的残差累加运算，生成一个统一的 Block Representation。
	- **块间（Inter-block）** ：将当前层之前的历史 Block 提取出来，使用当前层的伪查询向量去对历史 Block Representation 进行注意力加权求和，以此作为新的残差输入。

3**.系统级工程优化：缓存与两阶段计算** 针对分布式训练环境（特别是流水线并行），Block AttnRes 集成了两个系统级优化：

- **基于缓存的流水线通信（Cache-based pipeline communication）** ：用于跨节点高效传递历史 block 的表征。
![](https://picx.zhimg.com/v2-8b06817a01d5bb68d1348bcce211837d_1440w.jpg)

- **两阶段计算策略（Two-phase computation strategy）** ：使得注意力残差几乎可以作为标准残差模块的 **“即插即用（drop-in replacement）”** 方案替代品，实现最小的工程开销（minimal overhead）。

---

### 三、 模型实现与实验表现

该技术在极高的计算规模下进行了充分的验证和工程落地：

1. **提升模型 Scaling Laws（扩展定律）上限** 实验证明，使用 Block AttnRes 训练的模型在各个计算预算下的验证集 Loss 均始终低于 Baseline。 **Block AttnRes 模型达到的效果，相当于增加了 1.25 倍计算量的传统 Baseline 模型** （换言之，提供了约 25% 的白嫖算力等效收益）。
2. **改善训练动态（Training Dynamics）** 引入 AttnRes 后，成功缓解了 PreNorm 的稀释问题。随着网络深度的加深，输出的幅度得到了有效限制（不再无界增长），并且各层的梯度范数（gradient norms）分布变得更加均匀，极大有利于超大规模极深网络的收敛。
3. **集成至 [Kimi Linear 架构](https://zhida.zhihu.com/search?content_id=271623594&content_type=Article&match_order=1&q=Kimi+Linear+%E6%9E%B6%E6%9E%84&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzcwMTQxMzAsInEiOiJLaW1pIExpbmVhciDmnrbmnoQiLCJ6aGlkYV9zb3VyY2UiOiJlbnRpdHkiLCJjb250ZW50X2lkIjoyNzE2MjM1OTQsImNvbnRlbnRfdHlwZSI6IkFydGljbGUiLCJtYXRjaF9vcmRlciI6MSwiemRfdG9rZW4iOm51bGx9._A3IgfrB4EAU6TxEJZD3cSfR-_-lfaPD6F7dvWfYqN8&zhida_source=entity) （48B MoE 模型实践）**
- **训练配置** ：研究团队将该技术部署在总参数 48B、激活参数 3B 的 Kimi Linear 架构中，并在 1.4T 预训练 Token 上进行了训练验证。
- **下游任务全面跃升** ：与 Baseline 相比，在多数基准测试中都有稳定提升。其中最显著的增长体现在 **多步复杂推理** 和 **代码生成** 上——例如在高难度推理榜单 GPQA-Diamond 上猛增了 7.5 个百分点（从 36.9 提升至 44.4），在 HumanEval 代码测试上提升了 3.1 个点（59.1 提升至 62.2）。

---

**总结** ： 《Attention Residuals》 是一篇典型的“微小架构改动带来巨大系统级收益”的论文。Kimi 团队通过跨层 Attention 终结了 [Transformer](https://zhida.zhihu.com/search?content_id=271623594&content_type=Article&match_order=1&q=Transformer&zd_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ6aGlkYV9zZXJ2ZXIiLCJleHAiOjE3NzcwMTQxMzAsInEiOiJUcmFuc2Zvcm1lciIsInpoaWRhX3NvdXJjZSI6ImVudGl0eSIsImNvbnRlbnRfaWQiOjI3MTYyMzU5NCwiY29udGVudF90eXBlIjoiQXJ0aWNsZSIsIm1hdGNoX29yZGVyIjoxLCJ6ZF90b2tlbiI6bnVsbH0.HBoaDMLpY4B0wvAMgVkuO31rgByRFbj-x9Kw4TduhbE&zhida_source=entity) 统治多年的固定残差累加范式，并通过精巧的 Block 分块与流水线系统优化扫清了该机制落地的计算壁垒，成为了下一代大语言模型极具潜力的基础架构标准。

编辑于 2026-03-19 08:27・广东[kimi模型](https://www.zhihu.com/topic/1862547279633536077)[AI大模型](https://www.zhihu.com/topic/27420012)[模型架构](https://www.zhihu.com/topic/512451952)