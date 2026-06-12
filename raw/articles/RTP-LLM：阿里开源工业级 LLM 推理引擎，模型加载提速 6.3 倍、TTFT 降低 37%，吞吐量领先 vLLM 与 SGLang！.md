---
title: "RTP-LLM：阿里开源工业级 LLM 推理引擎，模型加载提速 6.3 倍、TTFT 降低 37%，吞吐量领先 vLLM 与 SGLang！"
source: "https://mp.weixin.qq.com/s/vhN7ct0sUr0NIFAAgnIILw"
author:
  - "[[阿里，北大，浙大]]"
published:
created: 2026-06-09
description: "大模型推理性能瓶颈正成为 AI 规模化落地的最大障碍。阿里巴巴最新开源的 RTP-LLM 引擎通过全栈式系统设计，在模型加载、KV 缓存管理、推测解码等多个维度实现突破性优化，性能全面超越 vLLM 和 SGLang，为工业级大模型部署提供了全新解决方案。"
tags:
  - "clippings"
---
阿里，北大，浙大 *2026年5月31日 00:00*

关键词：RTP-LLM、 **大模型推理** 、 ***Prefill-Decode 分离*** 、 **KV 缓存管理** 、推测解码

,19分钟

> "大语言模型已经彻底改变了 AI 应用，但大规模部署它们带来了重大挑战。"这是 RTP-LLM 论文开篇的第一句话，也是当前整个 AI 产业面临的共同困境。

![图片](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchH9EjzapB2oVHJPxVsWguFCbdibN353UHdXKxBgGsoibibYpdk79QXpO4iaLZTauNA58Vamica6oibsH6VzufqegliaPMM1kkUyRRNQr70/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

随着模型规模从数十亿参数增长到数千亿参数，传统推理系统在计算效率、内存管理、可扩展性和运维复杂度等方面的局限性日益凸显。

RTP-LLM 作为阿里巴巴大模型预测团队经过多年生产环境打磨的成果， **已经成功部署在淘宝、天猫、菜鸟等多个核心业务线，服务超过 1 亿用户。**

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHic7Q0OMU3icPPEfFmRSX7LpArlOgHz6aicjcaESPgyUjzhP6arBXibiabdYKmMkOiaA1WVx37YYic2FoKibv0lF4mNInicXnCxNY9bnPB8/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1)

- **RTP-LLM: High-Performance Alibaba LLM Inference Engine**
- 论文：https://arxiv.org/pdf/2605.29639
- 项目代码：https://github.com/alibaba/rtp-llm
- 文档：https://rtp-llm.ai/
- 1.4 万字，阅读 60 分钟，播客 19 分钟

,19分钟

相关推荐

- [速度：大模型推理的下一个 Scaling Law，深度解析 TileRT 高性能推理引擎及 GLM-5.1 生产级实践](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901538&idx=1&sn=f3055a8ef80e376c63335de3ccce42ad&scene=21#wechat_redirect)
- [一端写就，全端运行！端侧推理引擎 OmniInfer：让所有设备实现“触手可及”的大模型端侧推理体验](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899937&idx=1&sn=47fdb2f53cc9521e9db4527253cae0d1&scene=21#wechat_redirect)
- [老卡跑原生FP8模型！硬件成本降50%+！国产大模型推理引擎 Chitu（赤兔）全解析与实践指南](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895244&idx=1&sn=57c31bb4e6c4d1126388508a43b53107&scene=21#wechat_redirect)

> 论文指出， **生产环境中的大模型部署面临四大根本性挑战： *动态顺序工作负载下 GPU 利用率低下、无约束 KV 缓存增长导致的内存耗尽、面对架构异构性的系统刚性、以及阻碍快速迭代的运维脆弱性。***

![图 1：RTP-LLM 系统架构：该架构以 “中心化调度 + 分布式执行” 为核心，拆解 LLM 推理全链路为 7 大核心组件，精准匹配工业级部署需求。Master 作为全局调度中枢，统筹 Prefill/Decode 节点、多级缓存与 DP 控制器，解决传统框架单节点局限问题。PD 解耦部署模式是核心创新，将计算密集的 Prefill 与内存带宽密集的 Decode 物理分离，可独立扩缩容。同时配套 Carbon 服务实现故障自动恢复，兼顾高性能与运维稳定性，适配电商、客服等高并发场景。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHibhGer1zHJzkMWdBUlJicibywoj7ZzZHibWFcgAmcmqttayPhIsQZTVZHJ6MEzuzA1CUyvnOkgicEIWhPbpDE8DWQmhDicx9zuJDNMc/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)

图 1：RTP-LLM 系统架构：该架构以 “中心化调度 + 分布式执行” 为核心，拆解 LLM 推理全链路为 7 大核心组件，精准匹配工业级部署需求。Master 作为全局调度中枢，统筹 Prefill/Decode 节点、多级缓存与 DP 控制器，解决传统框架单节点局限问题。PD 解耦部署模式是核心创新，将计算密集的 Prefill 与内存带宽密集的 Decode 物理分离，可独立扩缩容。同时配套 Carbon 服务实现故障自动恢复，兼顾高性能与运维稳定性，适配电商、客服等高并发场景。

RTP-LLM 通过集成式设计理念，从模型加载、流量调度、缓存管理到执行优化进行了全栈式重构。

![表 1：性能评估关键指标。
该表定义 LLM 推理性能评估的核心指标，涵盖首 token 时间、吞吐量、GPU 内存占用、缓存命中率、批量延迟、困惑度 6 项关键维度。TTFT 衡量请求响应速度，吞吐量反映系统处理能力，GPU 内存占用关联模型部署成本，缓存命中率体现 KV 缓存复用效率，批量延迟表征批量处理效率，PPL 用于量化模型输出质量。多维度指标覆盖延迟、吞吐、资源占用与精度，为不同框架的公平对比提供统一标准，适配生产环境评估需求。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHiccLACYSPcjfG8AOtjRqsQVbyaIicamBP2oRmickjLicLyX77yf9MjxLSxtFs8R6eHoYNhibd38EaTsrKrC1WogjYvrl6wAvic79Klk/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=3)

表 1：性能评估关键指标。 该表定义 LLM 推理性能评估的核心指标，涵盖首 token 时间、吞吐量、GPU 内存占用、缓存命中率、批量延迟、困惑度 6 项关键维度。TTFT 衡量请求响应速度，吞吐量反映系统处理能力，GPU 内存占用关联模型部署成本，缓存命中率体现 KV 缓存复用效率，批量延迟表征批量处理效率，PPL 用于量化模型输出质量。多维度指标覆盖延迟、吞吐、资源占用与精度，为不同框架的公平对比提供统一标准，适配生产环境评估需求。

实验结果表明，RTP-LLM **在多个关键指标上全面超越当前主流的 vLLM 和 SGLang** ：模型加载速度提升 4.7 倍至 6.3 倍，生产流量调度中 TTFT P95 延迟降低 35%至 37%，缓存重用率提高 215%，推测解码吞吐量提升 1.12 倍至 2.48 倍，多模态推理吞吐量提升 1.86 倍至 2.52 倍。

![表 7：不同张量并行（TP）配置下，Qwen3-235B-A22B 模型的加载时间对比，时间单位为秒（s）。该表对比超大规模 235B 模型加载性能，RTP-LLM 在 TP=4 时加载耗时 37.1s，TP=8 时 33.0s，较 SGLang、vLLM 快 4.7-6.3 倍。基线框架 TP 提升后加载时间反而增加，源于低效 I/O 与通信设计。RTP-LLM 通过共享权重并行读取、I/O 与广播重叠，将模型加载耗时压缩至分钟级，支撑 600B 级模型的快速迭代与分钟级部署，适配大规模生产环境的模型更新需求。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchH8PvqjwVictE788omgicNXonQ3KOrzCwQaPhFE3RrJKPwUffhJpLfzw7KiajfjibRhcM364gPD5yZE49WCEW37ibH0oVLibt2twkmMNk/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4)

表 7：不同张量并行（TP）配置下，Qwen3-235B-A22B 模型的加载时间对比，时间单位为秒（s）。该表对比超大规模 235B 模型加载性能，RTP-LLM 在 TP=4 时加载耗时 37.1s，TP=8 时 33.0s，较 SGLang、vLLM 快 4.7-6.3 倍。基线框架 TP 提升后加载时间反而增加，源于低效 I/O 与通信设计。RTP-LLM 通过共享权重并行读取、I/O 与广播重叠，将模型加载耗时压缩至分钟级，支撑 600B 级模型的快速迭代与分钟级部署，适配大规模生产环境的模型更新需求。

![图 4：不同张量并行配置下，中等规模模型（8B-32B 参数）的加载时间对比：中等模型测试验证了 RTP-LLM 加载优化的普适性，TP 配置越高提速越明显。vLLM 与 SGLang 随 TP 增加加载时间反增，因多进程下重复读、通信开销叠加；而 RTP-LLM 通过单进程读 + 分布式广播，将 I/O 负载分散到多 GPU，高 TP 下优势放大。该结果证明文件序驱动 + 并行 I/O 设计，可突破传统分布式加载的扩展性瓶颈，适配 8B-32B 主流商用模型快速迭代需求。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHicib4fKanrPcIicTibw3Q27qjvcDZ4ibUrctMADn17yxAGD3kHnuUhfAiaL4r1SJKB1DUOAIAEBVDR9fia0lS8BVH8YpmDXPQf2a9z80/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)

图 4：不同张量并行配置下，中等规模模型（8B-32B 参数）的加载时间对比：中等模型测试验证了 RTP-LLM 加载优化的普适性，TP 配置越高提速越明显。vLLM 与 SGLang 随 TP 增加加载时间反增，因多进程下重复读、通信开销叠加；而 RTP-LLM 通过单进程读 + 分布式广播，将 I/O 负载分散到多 GPU，高 TP 下优势放大。该结果证明文件序驱动 + 并行 I/O 设计，可突破传统分布式加载的扩展性瓶颈，适配 8B-32B 主流商用模型快速迭代需求。

这些数据不仅代表了技术上的突破， ***更意味着在相同硬件成本下能够服务更多用户，或者在相同服务质量下大幅降低基础设施投入。***

![图片](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHicRCDT5zYxLhicjCAdXia5vuvqRvcLlnxmd1LACstsEpCylDeb5hkoCJia7Lv2mm3s6mM5llBDJfzMxYRRrKiaoaES3oXalYpsibxHk/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=6)

## 本文目录

- 一、RTP-LLM 系统架构设计
- 1.1 核心系统组件
	- 1.2 执行流程
	- 1.3 部署模式
- 二、高效模型加载技术
- 2.1 文件顺序驱动加载
	- 2.2 混合分布式读取
	- 2.3 共享内存重用
	- 2.4 I/O-通信重叠
	- 2.5 性能评估
- 三、流量调度与 KV 缓存管理
- 3.1 Prefill 阶段负载均衡
	- 3.2 Decode 阶段负载均衡
	- 3.3 KV 缓存管理架构
	- 3.4 前缀缓存匹配
	- 3.5 采样前缀哈希
	- 3.6 缓存匹配与调度集成
	- 3.7 性能评估
- 四、Prefill-Decode 分离架构
- 4.1 分离架构的优势
	- 4.2 跨节点 KV 缓存传输
	- 4.3 性能评估
- 五、模块化推测解码框架
- 5.1 框架架构
	- 5.2 支持的算法
	- 5.3 Prompt Lookup 推测采样
	- 5.4 性能评估
- 六、量化与多模态支持
- 6.1 量化技术
	- 6.2 多模态模型支持
	- 6.3 性能评估
- 七、相关工作
- 7.1 内存管理技术
	- 7.2 推理调度技术
	- 7.3 推测解码技术
	- 7.4 多模态推理技术
	- 7.5 与 RTP-LLM 的比较
- 总结与展望
![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchH8Md5RgA2Wkm4Gbzia4t5Trib2RdntSzof09DsAxMd9e3szWf1la2n3LFDatzibARyDtd6gXBXSkkagibRfBaVCJ2o1icGEUm8USVibY/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=7)

## 一、RTP-LLM 系统架构设计

> RTP-LLM 采用了分层模块化的系统架构设计， **将推理过程分解为多个独立组件，每个组件都针对特定功能进行了深度优化** 。这种设计 ***不仅提高了系统的可维护性和可扩展性，还允许不同组件独立演进和部署。***

系统架构的核心思想是"解耦与协同"，通过将计算特性差异巨大的阶段分离到不同的硬件资源上，实现资源的最优配置。同时，通过统一的控制平面和数据平面，确保各个组件之间能够高效协同工作。

![图 1：RTP-LLM 系统架构：该架构以 “中心化调度 + 分布式执行” 为核心，拆解 LLM 推理全链路为 7 大核心组件，精准匹配工业级部署需求。Master 作为全局调度中枢，统筹 Prefill/Decode 节点、多级缓存与 DP 控制器，解决传统框架单节点局限问题。PD 解耦部署模式是核心创新，将计算密集的 Prefill 与内存带宽密集的 Decode 物理分离，可独立扩缩容。同时配套 Carbon 服务实现故障自动恢复，兼顾高性能与运维稳定性，适配电商、客服等高并发场景。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHibdRaKLHmhw7fqlZpmMibxj6icGnSqEpwjkbby882BjNoeAicFibtRLB90mul6djpfC9EuPLnhZ2tJsRIKiavaUmXWib5zBh6ciakB0JY/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=8)

图 1：RTP-LLM 系统架构：该架构以 “中心化调度 + 分布式执行” 为核心，拆解 LLM 推理全链路为 7 大核心组件，精准匹配工业级部署需求。Master 作为全局调度中枢，统筹 Prefill/Decode 节点、多级缓存与 DP 控制器，解决传统框架单节点局限问题。PD 解耦部署模式是核心创新，将计算密集的 Prefill 与内存带宽密集的 Decode 物理分离，可独立扩缩容。同时配套 Carbon 服务实现故障自动恢复，兼顾高性能与运维稳定性，适配电商、客服等高并发场景。

### 1.1 核心系统组件

> RTP-LLM 的系统架构由七个核心组件组成：

- 前端应用：作为用户请求的入口，负责请求预处理，包括分词和元数据提取，然后将请求转发给主节点进行调度。
- 名称服务：执行心跳检测和服务发现，确定哪些集群处于运行状态，但不负责负载均衡。
- 主节点：承担流量调度和全局协调的关键角色，维护系统状态的全局视图，包括工作节点可用性、KV 缓存分布和当前负载情况。
- Prefill 节点：处理计算密集的 Prefill 阶段，并行处理整个输入提示并生成初始 KV 缓存状态。
- Decode 节点：管理内存密集的 Decode 阶段，利用缓存的注意力状态自回归地生成 token 。
- 多级缓存：实现了跨越 GPU 内存、本地 CPU 内存、远程 CPU 内存和分布式存储的分层存储系统，提高 KV 缓存的复用效率。
- DP 控制器：负责在单个部署上下文中协调和管理一批请求的执行，管理本地资源分配，包括 GPU 内存管理和批处理执行。

### 1.2 执行流程

> RTP-LLM 的完整执行流程在论文中通过算法 1 进行了详细描述。

![算法 1：RTP-LLM 分层架构执行流程。该算法定义 RTP-LLM 核心执行逻辑，以哈希前缀匹配、四级缓存加载、动态批处理为核心。第一步生成 token 哈希键，第二步全局前缀匹配定位缓存，第三步 Master 生成批任务。四级缓存从 GPU 到分布式存储逐级加载，最大化缓存命中率，减少 I/O 开销。流程中心化调度 + 分布式执行，兼顾全局最优与节点效率，是 RTP-LLM 工业级调度的核心逻辑，适配复杂动态负载。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchH9huMtCHN3Y9Wnhe83vevKjVObwV9l2vIShrHUU4K7v1VljysQO2QIpVb3q9libMj3uEIIZMDgfP3YY8Ma9VW5nWaf6dB7d2hcw/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=9)

算法 1：RTP-LLM 分层架构执行流程。该算法定义 RTP-LLM 核心执行逻辑，以哈希前缀匹配、四级缓存加载、动态批处理为核心。第一步生成 token 哈希键，第二步全局前缀匹配定位缓存，第三步 Master 生成批任务。四级缓存从 GPU 到分布式存储逐级加载，最大化缓存命中率，减少 I/O 开销。流程中心化调度 + 分布式执行，兼顾全局最优与节点效率，是 RTP-LLM 工业级调度的核心逻辑，适配复杂动态负载。

整个过程可以分为五个主要步骤：

1. 前缀哈希生成：主节点从用户请求的 token 序列中生成哈希键，用于后续的前缀匹配。
2. 全局缓存匹配：主节点使用生成的哈希键在全局缓存中进行前缀匹配，识别已有的 KV 缓存块。
3. 负载分析与调度：主节点根据请求详情、匹配结果和当前集群状态进行负载分析，制定执行批处理并分发到指定的推理节点。
4. 分层内存访问：推理节点按照从快到慢的顺序尝试加载所需的 KV 缓存块，依次检查 GPU 内存、本地 CPU 内存、远程 CPU 内存和分布式存储。
5. 推理执行与缓存更新：推理节点对批处理执行模型推理，完成后返回使用的 KV 缓存块并更新 LRU 指标。

### 1.3 部署模式

> RTP-LLM 支持两种不同的部署策略：

- PD-Fusion：Prefill 和 Decode 阶段在同一个统一节点上共存和执行。这种模式适用于资源有限或者负载相对均衡的场景。
- PD-Disaggregation：将 Prefill 和 Decode 阶段物理分离到专用节点上。这是论文中重点介绍的拓扑结构，也是阿里巴巴生产环境中主要采用的部署模式。

在 PD-Disaggregation 部署中，每个推理服务节点都配备了一个专用的 Carbon 服务，负责在发生故障时自动恢复和重启推理服务，确保系统的高可用性。

**RTP-LLM 的系统架构设计体现了"面向生产环境"的核心理念** ，通过 Prefill-Decode 分离、多级缓存和分层调度，解决了传统推理系统在大规模部署时面临的资源利用率低、延迟不稳定和可扩展性差等问题。

## 二、高效模型加载技术

> **大模型加载在分布式环境** 中面临着严峻的性能挑战， **特别是当处理跨多个张量并行进程的数百亿参数模型时** 。在阿里巴巴的生产环境中， ***所有模型检查点都存储在内部 FUSE 云存储系统上，这使得文件访问效率严重依赖于 I/O 模式。***

传统的模型结构驱动加载方法存在两个关键性能问题：进程间的冗余文件读取和非顺序文件访问模式，这严重降低了 FUSE 预取效率和文件级缓存利用率。RTP-LLM 通过将加载范式从模型结构驱动重构为文件顺序驱动，彻底解决了这些问题。

### 2.1 文件顺序驱动加载

传统的加载方法是遍历权重模块并加载其组成的张量，这导致每个张量并行进程都需要读取所有模型文件以提取其分配的张量部分。而 **RTP-LLM 采用的文件顺序驱动加载方法则是顺序遍历模型文件，在处理下一个文件之前加载当前文件中的所有张量。**

这种方法确保了顺序文件访问模式，最大化了 FUSE 预取的有效性，同时保持了与社区引擎的兼容性。

### 2.2 混合分布式读取

为了消除冗余读取，RTP-LLM **集成了 IBM 的 fastsafetensors 库与 RTP fastsafetensors 实现** 。这种混合方法将每个模型文件分配给单个进程进行读取，然后 **利用 PyTorch 分布式广播在所有进程之间高效共享张量。**

这种设计消除了每个进程都需要读取每个文件的需求，同时保留了基于 FUSE 的 I/O 优化的优势。

### 2.3 共享内存重用

原始的 fastsafetensors 库为每个文件读取操作分配和注册固定内存，每 2GB 分配需要 600ms 的开销。RTP-LLM 通过 **将加载接口封装在一个类中，在多个文件读取之间重用单个共享内存缓冲区** ，彻底消除了这种冗余的分配成本。

### 2.4 I/O-通信重叠

> RTP-LLM 通过并行化文件读取操作与张量广播，优化了通信和 I/O 的重叠。

![图 2：模型加载优化：图中聚焦文件序驱动 I/O、共享内存复用、I/O 与通信并行三大核心优化，直击 FUSE 云存储下大模型加载痛点。传统结构驱动加载存在重复读、随机 I/O 低效问题，文件序遍历保证顺序访问，最大化 FUSE 预取效率。共享内存复用消除重复内存分配开销，I/O 与广播并行重叠耗时环节，从 I/O 模式、内存管理、通信协同三方面系统性解决千亿级模型加载慢的难题。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHicVIDVZ3t3VVMwaZklVP6ndp84D1veicZS5LznhS7s201R6ibuFSGbkLk9jcrov4lUBeNmNq0xNOTXgxlB2B4ibdib5qldfoVqgRw8/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=10)

图 2：模型加载优化：图中聚焦文件序驱动 I/O、共享内存复用、I/O 与通信并行三大核心优化，直击 FUSE 云存储下大模型加载痛点。传统结构驱动加载存在重复读、随机 I/O 低效问题，文件序遍历保证顺序访问，最大化 FUSE 预取效率。共享内存复用消除重复内存分配开销，I/O 与广播并行重叠耗时环节，从 I/O 模式、内存管理、通信协同三方面系统性解决千亿级模型加载慢的难题。

与顺序读取文件然后分发张量不同，RTP-LLM 允许文件 I/O 操作与张量广播同时进行，最大化了分布式系统中的资源利用率。

### 2.5 性能评估

> 论文对 RTP-LLM 的模型加载性能进行了全面评估，对比了五个不同规模的模型（8B-235B 参数）在不同张量并行配置下的表现。

![图 4：不同张量并行配置下，中等规模模型（8B-32B 参数）的加载时间对比：中等模型测试验证了 RTP-LLM 加载优化的普适性，TP 配置越高提速越明显。vLLM 与 SGLang 随 TP 增加加载时间反增，因多进程下重复读、通信开销叠加；而 RTP-LLM 通过单进程读 + 分布式广播，将 I/O 负载分散到多 GPU，高 TP 下优势放大。该结果证明文件序驱动 + 并行 I/O 设计，可突破传统分布式加载的扩展性瓶颈，适配 8B-32B 主流商用模型快速迭代需求。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHib43sT20enWZySakEyUvqZKaT3vt7JeqMyfCWJibXz8TQolIBUh4bFxscicV0rjsCXg9jy5NXO8iahgoses0Wpab6Vp4iaKGzwZc2A/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=11)

图 4：不同张量并行配置下，中等规模模型（8B-32B 参数）的加载时间对比：中等模型测试验证了 RTP-LLM 加载优化的普适性，TP 配置越高提速越明显。vLLM 与 SGLang 随 TP 增加加载时间反增，因多进程下重复读、通信开销叠加；而 RTP-LLM 通过单进程读 + 分布式广播，将 I/O 负载分散到多 GPU，高 TP 下优势放大。该结果证明文件序驱动 + 并行 I/O 设计，可突破传统分布式加载的扩展性瓶颈，适配 8B-32B 主流商用模型快速迭代需求。

对于大规模的 Qwen3-235B-A22B 模型，RTP-LLM 的性能优势更加明显：

![表 7：不同张量并行（TP）配置下，Qwen3-235B-A22B 模型的加载时间对比，时间单位为秒（s）。该表对比超大规模 235B 模型加载性能，RTP-LLM 在 TP=4 时加载耗时 37.1s，TP=8 时 33.0s，较 SGLang、vLLM 快 4.7-6.3 倍。基线框架 TP 提升后加载时间反而增加，源于低效 I/O 与通信设计。RTP-LLM 通过共享权重并行读取、I/O 与广播重叠，将模型加载耗时压缩至分钟级，支撑 600B 级模型的快速迭代与分钟级部署，适配大规模生产环境的模型更新需求。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHicERYY3hZYZprgmcDXMcbwtzqgyh3Ca6fW0KjQB4ibLzJxcczWCp4kJofFIPQt8gn6GbYWmQqIW3rLxpWkLYtiabOb4CMItutuicw/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=12)

表 7：不同张量并行（TP）配置下，Qwen3-235B-A22B 模型的加载时间对比，时间单位为秒（s）。该表对比超大规模 235B 模型加载性能，RTP-LLM 在 TP=4 时加载耗时 37.1s，TP=8 时 33.0s，较 SGLang、vLLM 快 4.7-6.3 倍。基线框架 TP 提升后加载时间反而增加，源于低效 I/O 与通信设计。RTP-LLM 通过共享权重并行读取、I/O 与广播重叠，将模型加载耗时压缩至分钟级，支撑 600B 级模型的快速迭代与分钟级部署，适配大规模生产环境的模型更新需求。

RTP-LLM 的模型加载优化将 600B+参数模型的部署时间从小时级降低到分钟级，这对于需要频繁更新模型的企业级应用来说具有革命性的意义，极大地提高了模型迭代的速度和灵活性。

## 三、流量调度与 KV 缓存管理

> 流量调度是大模型推理系统的核心组件之一，直接决定了系统的吞吐量、延迟和资源利用率。RTP-LLM 针对 Prefill 和 Decode 阶段不同的计算特性，采用了不同的负载均衡机制，并结合先进的 KV 缓存管理技术，实现了高效的请求调度和资源分配。

传统的流量调度策略通常将 Prefill 和 Decode 视为相同的工作负载，采用统一的调度算法，这导致了资源的不合理分配和性能的下降。RTP-LLM 通过将两个阶段分离并采用针对性的调度策略，显著提高了系统的整体性能。

### 3.1 Prefill 阶段负载均衡

> 对于 Prefill 请求，前端应用首先对输入序列进行分词并生成块哈希标识符。每个请求被划分为多个块（例如每块 64 个 token ），每个块的哈希键基于其 token ID 计算。前端应用将请求的块哈希 ID、序列长度和可选的聊天 ID 发送给主节点。

主节点将具有相似序列长度的请求分组为批处理，以最小化填充开销。窗口大小 根据 DP 组大小和队列深度动态调整。主节点查询 DP 控制器的实时负载状态，包括运行中/等待中的请求、GPU 内存和 KV 缓存占用率。

当所有 DP 控制器都繁忙时，主节点采用预测调度，通过估计完成时间来决定将请求调度到哪个 DP 控制器：

其中， 表示请求 r 在 DP 控制器 上开始执行的时间， 是基于序列长度和批处理组成预测的 Prefill 时间。主节点将请求调度到预计最先完成的 DP 控制器，以减少队列等待时间。

### 3.2 Decode 阶段负载均衡

> Decode 请求优先考虑 KV 缓存亲和性：当带有聊天 ID 的请求到达时，主节点检查该聊天会话是否先前已分配给某个工作节点。如果存在匹配且该工作节点有足够的容量，主节点直接路由到该工作节点，利用本地 KV 缓存局部性。

对于缓存管理，主节点实现了准入控制、驱逐优先级和背压信号，以防止缓存抖动。

### 3.3 KV 缓存管理架构

> RTP-LLM 采用基于哈希的方法在工作节点之间进行高效的前缀匹配。本地 KV 缓存管理器维护一个统一的哈希映射，聚合来自所有工作节点的缓存键，将哈希键映射到块标识符和工作节点元数据。

与为每个工作节点维护单独的哈希映射需要 次查找不同，RTP-LLM 将所有工作节点的缓存键合并到单个哈希映射中，使前缀匹配的复杂度降低到 ，其中 B 是块的数量，W 是工作节点的数量。

主节点以高频率（20ms）查询工作节点状态以进行调度决策，而缓存键同步以较低频率（50ms）进行。工作节点维护缓存版本号，当请求缓存键时，管理器包含最后已知的版本。如果未更改，工作节点返回轻量级确认；如果已更改，工作节点返回增量更新以最小化数据传输。

### 3.4 前缀缓存匹配

> 当前端应用发送带有块哈希 ID 的请求时，主节点查询本地 KV 缓存管理器进行前缀匹配。匹配过程在论文中通过算法 2 进行了详细描述。

![算法 2：前缀缓存匹配。该算法实现基于统一哈希映射的高效前缀匹配，核心是单次哈希表遍历完成全局匹配，复杂度从 O (B×W) 降至 O (B)。算法逐块匹配请求前缀哈希，累计匹配长度并记录各工作节点最大匹配长度，匹配中断则终止流程。通过合并所有节点缓存键至单一哈希表，减少跨节点查询开销，同时 50ms 低频同步缓存版本，降低数据传输量。高效匹配为 KV 缓存复用提供基础，显著减少预填充计算量，提升系统整体吞吐量。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchH9eFIgqia83Xq1kiazykSG7GLceZquWU0P01ZOjiaRtXAF93XIx3EPOcQ6S12x63pxTiaDlKy5c2zMjs82NnnVUDFcHsLw5htUt6KU/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=13)

算法 2：前缀缓存匹配。该算法实现基于统一哈希映射的高效前缀匹配，核心是单次哈希表遍历完成全局匹配，复杂度从 O (B×W) 降至 O (B)。算法逐块匹配请求前缀哈希，累计匹配长度并记录各工作节点最大匹配长度，匹配中断则终止流程。通过合并所有节点缓存键至单一哈希表，减少跨节点查询开销，同时 50ms 低频同步缓存版本，降低数据传输量。高效匹配为 KV 缓存复用提供基础，显著减少预填充计算量，提升系统整体吞吐量。

### 3.5 采样前缀哈希

对于工作节点上的前缀匹配，RTP-LLM 使用采样前缀哈希来平衡匹配粒度和存储开销。当缓存块包含的 token 少于阈值（例如 208 个 token ）时，仅对该长度进行哈希。对于更大的块，以固定间隔创建多个哈希条目。

具体来说，对于包含 个 token 的块，在以下位置创建哈希条目：208, 212, 216, 220, 224, 228,..., 直到 n。这种以起始阈值 208 和步长 4 为参数的采样策略，允许在多个粒度上进行前缀匹配，同时控制元数据开销。

### 3.6 缓存匹配与调度集成

当主节点收到调度请求时，它并行查询本地和远程 KV 缓存管理器：

- 本地缓存查询：查询本地 KV 缓存管理器的统一哈希映射以获取工作节点级缓存匹配，返回每个工作节点的最大匹配长度。
- 远程缓存查询：查询远程 KV 缓存管理器服务器以获取 3FS 缓存匹配，返回来自持久存储的最大匹配长度。

主节点结合这两个结果计算每个候选工作节点 w 的缓存重用分数：

其中 、 和 是根据工作负载特性调整的权重因子。该分数与工作节点负载信息结合，做出最终的调度决策。

### 3.7 性能评估

> 论文使用阿里巴巴内部的真实部署环境评估了 RTP-LLM 的流量调度性能，包括内部机器人问答服务和淘宝商家客服咨询。

![表 2：两种真实生产负载下的流量调度性能对比。TS：流量调度，延迟单位为毫秒（ms）。
该表基于阿里内部机器人问答、淘宝商家客服两类真实生产场景，验证 RTP-LLM 流量调度（TS）的效果。开启 TS 后，机器人问答场景 TTFT P95 从 83.3ms 降至 52.3ms，客服场景从 350ms 降至 226ms，延迟降幅超 35%。推理延迟也同步下降，核心原因是统一哈希映射与缓存亲和路由优化 KV 缓存复用，减少预填充阶段计算开销，支撑高并发、动态请求的稳定低延迟服务。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHibtRZVlKeH2eVMWLMmmbqGZvklQFQD1N9ibVb67j2bQobQqRIia9ghgKwAwftu36MON6nHAicibibf2RZvHPdgdB5yoGzrUXDKuxJd0/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=14)

表 2：两种真实生产负载下的流量调度性能对比。TS：流量调度，延迟单位为毫秒（ms）。 该表基于阿里内部机器人问答、淘宝商家客服两类真实生产场景，验证 RTP-LLM 流量调度（TS）的效果。开启 TS 后，机器人问答场景 TTFT P95 从 83.3ms 降至 52.3ms，客服场景从 350ms 降至 226ms，延迟降幅超 35%。推理延迟也同步下降，核心原因是统一哈希映射与缓存亲和路由优化 KV 缓存复用，减少预填充阶段计算开销，支撑高并发、动态请求的稳定低延迟服务。

![表 3：不同策略下的 KV 缓存复用长度对比，缓存复用长度单位为 token。
该表量化流量调度策略对 KV 缓存复用效率的提升，内部机器人问答场景开启 TS 后，缓存复用长度从 26.6token 增至 83.8token，提升 215%；淘宝客服场景从 833token 增至 840token。缓存复用提升源于 RTP-LLM 的统一哈希映射前缀匹配，能高效识别请求间共享前缀，复用已有 KV 缓存块。复用率提升直接减少预填充机器数量，从 80 台降至 20 台，显著降低生产部署成本。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchH98ufPwkQ03fBt1BviaAVFhkcmtpvWp5hBotj4eibX8yaCGLSfnCnIKFbAcGyqicoAibf6H7A52S4gyic3hqOnWpoOYuYYibjP8HaOYc/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=15)

表 3：不同策略下的 KV 缓存复用长度对比，缓存复用长度单位为 token。 该表量化流量调度策略对 KV 缓存复用效率的提升，内部机器人问答场景开启 TS 后，缓存复用长度从 26.6token 增至 83.8token，提升 215%；淘宝客服场景从 833token 增至 840token。缓存复用提升源于 RTP-LLM 的统一哈希映射前缀匹配，能高效识别请求间共享前缀，复用已有 KV 缓存块。复用率提升直接减少预填充机器数量，从 80 台降至 20 台，显著降低生产部署成本。

RTP-LLM 的流量调度与 KV 缓存管理技术通过统一哈希映射、缓存亲和性路由和分层缓存架构，实现了 215%的缓存重用率提升和 35%-37%的 TTFT 延迟降低，这在大规模生产环境中意味着巨大的成本节约和用户体验提升。

## 四、Prefill-Decode 分离架构

> Prefill-Decode 分离是 RTP-LLM 最具创新性的架构设计之一，它利用了 LLM 推理过程中两个阶段截然不同的计算特性，将它们物理分离到专用的计算资源上，实现了独立的扩展和优化。

LLM 推理过程可以分解为两个具有不同计算特性的阶段：

- Prefill 阶段：并行处理整个输入提示，生成并存储所有提示 token 的 KV 缓存条目，并产生第一个输出 token 。这个阶段涉及所有输入 token 的并行计算，因此是计算密集型的。
- Decode 阶段：自回归地生成后续 token ，利用当前 token 和历史 KV 缓存。每个解码迭代处理单个 token 并相应地更新 KV 缓存，因此是内存带宽密集型的。

传统的推理系统将这两个阶段在同一个节点上执行，导致资源的不合理分配：Prefill 阶段需要大量的计算资源但内存带宽需求较低，而 Decode 阶段需要大量的内存带宽但计算资源需求较低。这种不匹配导致 GPU 资源无法得到充分利用。

### 4.1 分离架构的优势

> Prefill-Decode 分离架构通过将这两个阶段物理分离到专用节点上，带来了以下几个关键优势：

1. 资源最优配置：Prefill 节点可以配置为高计算能力的硬件，而 Decode 节点可以配置为高内存带宽的硬件，实现资源的最优匹配。
2. 独立扩展：可以根据工作负载的特点独立扩展 Prefill 和 Decode 节点的数量。例如，在长上下文场景下，可以增加更多的 Prefill 节点；在高并发短响应场景下，可以增加更多的 Decode 节点。
3. 批处理优化：Prefill 节点可以处理更大的批处理大小，最大化计算资源的利用率；Decode 节点可以处理更高的并发，最大化内存带宽的利用率。
4. 故障隔离：Prefill 和 Decode 节点的故障相互独立，提高了系统的整体可靠性。

### 4.2 跨节点 KV 缓存传输

在分离架构中，一个关键的挑战是如何高效地将 Prefill 节点生成的 KV 缓存传输到 Decode 节点。RTP-LLM 使用 NCCL IBRC（InfiniBand 可靠连接）进行高性能、低延迟的数据传输，确保缓存状态在分布式部署中的高效通信。

### 4.3 性能评估

> 论文使用 Qwen3-Coder-480B-FP8 模型评估了 Prefill-Decode 分离架构的性能，这是一个大规模的 MoE 模型，部署在阿里巴巴的在线业务环境中。

部署配置采用了 5 个节点的分布式设置，每个节点配备 8 个 GPU。其中 4 个节点专用于 Prefill 处理，1 个节点处理 Decode 操作。这种不对称分配反映了典型的工作负载特性，即 Prefill 由于输入序列的并行处理需要更多的计算资源，而 Decode 由于每个请求的计算强度较低而受益于更高的并发性。

每个节点配置了张量并行（TP=8）、专家并行（EP=8）和数据并行（DP=1）。专家并行配置使用 DeepEP 进行专家层之间的高效 All2All 通信，优化了 MoE 模型中的通信开销。

Prefill 节点配置了 64 的批处理大小，允许高效地并行处理输入序列。Decode 节点支持 128 的更高并发性，利用解码操作的内存绑定特性同时处理多个请求。

![表 4：启用 KV 缓存 FP8 的 Qwen3-Coder-480B-A35B-Instruct-FP8 模型，不同框架的性能对比。
该表对比超大规模 MoE 模型在 RTP-LLM、SGLang、vLLM 上的表现，RTP-LLM 缓存命中率达 45.09%，是 SGLang 的 1.57 倍、vLLM 的 2.36 倍；TTFT 仅 1338.38ms，较基线框架快 4.7 倍以上。吞吐量与基线框架持平，源于 PD 解耦架构将预填充、解码分离，预填充节点聚焦高吞吐，解码节点优化低延迟，结合高效缓存调度，适配超大规模 MoE 模型的生产部署需求。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchH9EQVYAW4uqpc2a7ylYxy2Of2dqOo05IoRJvFv8oPMRwIyXY5ibJxbTJMQnYU8RIhg2iabDcC92diatic6EBy9zJS8CQicFzztLnsnw/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=16)

表 4：启用 KV 缓存 FP8 的 Qwen3-Coder-480B-A35B-Instruct-FP8 模型，不同框架的性能对比。 该表对比超大规模 MoE 模型在 RTP-LLM、SGLang、vLLM 上的表现，RTP-LLM 缓存命中率达 45.09%，是 SGLang 的 1.57 倍、vLLM 的 2.36 倍；TTFT 仅 1338.38ms，较基线框架快 4.7 倍以上。吞吐量与基线框架持平，源于 PD 解耦架构将预填充、解码分离，预填充节点聚焦高吞吐，解码节点优化低延迟，结合高效缓存调度，适配超大规模 MoE 模型的生产部署需求。

Prefill-Decode 分离架构通过将计算特性差异巨大的两个阶段分离到专用资源上，实现了 4.72 倍至 5.33 倍的 TTFT 加速，这对于需要快速响应的在线交互式应用来说具有决定性的意义。

## 五、模块化推测解码框架

> 推测解码是解决 LLM 推理中顺序生成瓶颈的关键技术，它通过引入并行 token 验证机制，将顺序解码转换为并行验证，显著提高了 GPU 利用率。RTP-LLM 实现了一个全面的推测采样框架，支持多种推测采样算法，同时保持了模块化和可扩展性。

传统的自回归生成存在固有的顺序瓶颈，每个生成的 token 都依赖于所有先前的 token ，这限制了 GPU 的利用率。推测解码通过使用一个较小的提议模型生成多个候选 token ，然后使用目标模型并行验证这些 token ，从而突破了这个瓶颈。

### 5.1 框架架构

> RTP-LLM 的推测采样框架采用 C++实现，将推测采样分解为四个模块化组件：

- ProposeExecutor：管理不同算法（朴素推测采样、Prompt Lookup、Eagle、MTP）的 token 提议生成。
- ScoreExecutor：处理目标模型的并行 token 评分。
- SpeculativeSampler：实现验证算法，根据概率分布确定接受哪些候选 token 。
- SpeculativeUpdater：将接受的 token 更新到原始流中。

执行流程如下：

1. ProposeExecutor 使用配置的提议算法生成 k 个候选 token 。
2. ScoreExecutor 通过目标模型执行并行前向传播，同时对所有 k 个候选 token 进行评分。
3. SpeculativeSampler 应用验证算法，根据概率分布确定接受哪些候选 token 。
4. SpeculativeUpdater 将接受的 token 集成到原始生成流中，相应地推进生成状态。

每个组件都维护清晰的输入/输出接口和无状态操作，确保松耦合并促进算法实验。

### 5.2 支持的算法

> RTP-LLM 的框架支持多种推测采样方法：

- 朴素推测采样：直接使用较小的 GPT 模型作为提议模型。
- MTP（多 token 预测）：在一次前向传播中预测多个下一个 token 以进行并行验证（例如 DeepSeek-V3）。
- Eagle：用于未来隐藏状态预测的新型自回归头训练。
- Prompt Lookup：针对历史提示的 n-gram 匹配以进行 token 提议。

### 5.3 Prompt Lookup 推测采样

> Prompt Lookup 是一种专门的推测采样形式，特别适用于提取式场景，其中生成的内容可以直接从输入提示中复制。该算法通过使用最近生成的 token 对输入提示进行 n-gram token 匹配，提取后续的 k 个 token 作为候选提议，并通过评分模型进行验证。

![算法 3：N 元语法 token 匹配与推测采样。
该算法实现提示词查找式推测采样，适配文本生成、代码编辑等场景。算法先基于历史 token 与输入提示词做 N 元语法匹配，提取 k 个候选 token，再经评分模型并行验证，最终通过验证算法筛选有效 token。针对代码编辑场景，优化光标定位、初始匹配跳过、位置更新等逻辑，适配代码连续复制特性。模块化设计支持与 Medusa、Eagle 等推测解码算法灵活切换，提升长文本、代码生成等场景的推理速度，适配高并发生产需求。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchH9OUl8JEFVxCicibYdiciboDrVSqBfECzhaQLNlAicKiciatEk5TFmyayD0RKhYRiaNWKYjjib7atftf9mPm9iagmVlJu8dagr1OibWXB2TDA/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=17)

算法 3：N 元语法 token 匹配与推测采样。 该算法实现提示词查找式推测采样，适配文本生成、代码编辑等场景。算法先基于历史 token 与输入提示词做 N 元语法匹配，提取 k 个候选 token，再经评分模型并行验证，最终通过验证算法筛选有效 token。针对代码编辑场景，优化光标定位、初始匹配跳过、位置更新等逻辑，适配代码连续复制特性。模块化设计支持与 Medusa、Eagle 等推测解码算法灵活切换，提升长文本、代码生成等场景的推理速度，适配高并发生产需求。

### 5.4 性能评估

> 论文使用 DeepSeek-V3-0324 模型评估了 RTP-LLM 的推测解码性能，其中模型的最后一层用作推测解码的草稿模型。

![表 5：DeepSeekV3-0324 模型推测解码下，不同框架的吞吐量（token/s）对比。
该表验证 RTP-LLM 推测解码的性能优势，吞吐量达 187.53token/s，较 vLLM 提升 1.12 倍，较 SGLang 提升 2.48 倍。优势源于 RTP-LLM 采用直接 C++ 算子调用机制，规避 vLLM 等开源框架 Python 转 C++ 的调用开销，减少推测解码流水线的算子调用延迟。同时模块化推测解码框架支持多算法灵活切换，适配代码生成、长文本生成等不同生产场景，提升推理效率。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHib5GjHDfZMyPoLmjq9Y1cpEibXxuATcibtodf5rYEWTBNxoV9adqK4HoxTEHZzo4Iib1ccAiaedLm2O3337NK8U9JibzgId1KEviaVt8/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=18)

表 5：DeepSeekV3-0324 模型推测解码下，不同框架的吞吐量（token/s）对比。 该表验证 RTP-LLM 推测解码的性能优势，吞吐量达 187.53token/s，较 vLLM 提升 1.12 倍，较 SGLang 提升 2.48 倍。优势源于 RTP-LLM 采用直接 C++ 算子调用机制，规避 vLLM 等开源框架 Python 转 C++ 的调用开销，减少推测解码流水线的算子调用延迟。同时模块化推测解码框架支持多算法灵活切换，适配代码生成、长文本生成等不同生产场景，提升推理效率。

论文还报告了一个真实的生产部署场景：使用 Qwen3-235B-A22B MoE 模型的在线商家数据代理服务，启用了多 token 预测（MTP）。

![表 6：真实部署（235B MoE + MTP）：解码端单卡吞吐量（TPS）与平均单 token 生成时间（TPOT，ms）。
该表基于 235B MoE 模型 + MTP 推测解码的真实商家数据代理服务，对比四种解码配置的性能。1TP8DP 配置在 128-512 并发下 TPOT 最优，适配在线低延迟场景；2TP4DP 配置吞吐量最高，适合离线高吞吐任务；4TP×4 低并发下延迟最优，高并发稳定性下降。不同配置适配差异化生产需求，MTP 保持约 1.9 token / 步的有效采样率，KV 缓存利用率超 90%，验证方案在真实高并发场景的有效性。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHibEIu3eC68LFU3ZE4xf0hpMo2aV5bkFTH8ZWfbTlovO9JSIzDp8y1em9XSVBmTjQk8OKJ8XaW6B4b7osVmzXXRYqDQl2ZnCJkQ/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=19)

表 6：真实部署（235B MoE + MTP）：解码端单卡吞吐量（TPS）与平均单 token 生成时间（TPOT，ms）。 该表基于 235B MoE 模型 + MTP 推测解码的真实商家数据代理服务，对比四种解码配置的性能。1TP8DP 配置在 128-512 并发下 TPOT 最优，适配在线低延迟场景；2TP4DP 配置吞吐量最高，适合离线高吞吐任务；4TP×4 低并发下延迟最优，高并发稳定性下降。不同配置适配差异化生产需求，MTP 保持约 1.9 token / 步的有效采样率，KV 缓存利用率超 90%，验证方案在真实高并发场景的有效性。

RTP-LLM 的模块化推测解码框架通过支持多种算法和 C++级别的优化， **实现了 1.12 倍至 2.48 倍的吞吐量提升，同时保持了输出质量不变** ，这对于高吞吐量的离线处理和低延迟的在线服务都具有重要价值。

## 六、量化与多模态支持

模型量化是 LLM 推理不可或缺的优化技术，它解决了内存容量和带宽饱和的关键挑战。多模态模型支持则是当前 AI 发展的重要趋势，要求推理系统能够高效处理图像、视频和文本等多种模态的输入。RTP-LLM 在这两个方面都进行了深入的优化。

### 6.1 量化技术

将模型权重和中间状态的精度从 FP16/BF16 降低到更低的位格式（如 INT8/INT4），直接提高了吞吐量并使更大的模型能够部署在商品硬件上。

#### 6.1.1 仅权重量化

> 由于模型权重构成了模型内存占用的大部分， **RTP-LLM 主要采用仅权重量化方法** 。这些技术将权重转换为较低精度（通常为 INT4/INT8），同时保持激活在较高精度以进行计算，以最小化质量损失。

RTP-LLM 支持多种先进的量化方法：

- GPTQ（通用流水线 token 量化）：作为一种高效的一次性训练后量化方法，GPTQ 仍然是基础。
- AQT（激活感知量化）框架：现代方法如 AWQ（激活感知权重量化）和 HQQ（半二次量化）被优先用于推动到极低的位宽（如 INT3、INT2），通过利用激活分布或先进的非线性优化技术，确保 LLM 推理能力的最小退化。
- FP8：新兴的行业标准 FP8 也被集成，提供高性能和最小的精度退化，特别是当与支持该格式的硬件加速器结合使用时。

#### 6.1.2 KV 缓存量化

> KV 缓存存储中间注意力状态，随着上下文长度动态增长，很快成为内存带宽和容量的瓶颈， **特别是对于支持 128K+ token 上下文的模型。为了缓解这种内存压力，RTP-LLM 将标准量化技术专门应用于 KV 缓存。**

- 动态量化：Key 和 Value 张量在生成过程中从 FP16/BF16 量化为较低精度（通常为 INT8、INT4 或 FP8）。它直接使用每张量或每块动态缩放来确定量化因子，优先考虑硬件效率和速度。
- 内存占用减少：量化 KV 缓存有效地将其大小减少了一半（对于 INT8/FP8）或更多。这直接减轻了与 Decode 阶段相关的内存带宽压力，对于在内存绑定条件下最大化引擎的有效批处理大小和并发请求容量至关重要。

### 6.2 多模态模型支持

> RTP-LLM 主要关注接受图像作为输入的模型，特别支持 LLaVA 和 Qwen-VL 等突出的多模态架构。

#### 6.2.1 支持的多模态架构

- LLaVA 架构：LLaVA 模型集成遵循 HuggingFace 格式规范。配置文件包含 mm\_vision\_tower 关键字，指定 Vision Transformer（ViT）组件的路径。通常，此实现使用 OpenAI 的预训练 CLIP 模型进行视觉特征提取。调用接口与 HuggingFace 格式保持一致，用户使用 `<image>` 标签在提示中指定图像插入位置。
- Qwen-VL 架构：Qwen-VL 的实现与 LLaVA 在架构方法上略有不同。虽然 Qwen-VL 的 ViT 组件也使用 CLIP，但其参数与大语言模型（LLM）部分集成，导致 ViT 参数直接从模型检查点读取，而不是从单独的配置文件读取。调用接口类似地遵循 HuggingFace 格式约定，使用 `<img>{img_url}</img>` 标签在提示中标记图像。

#### 6.2.2 EPD 分离部署

> 对于生产使用，RTP-LLM 支持基于服务器的多模态模型服务。RTP-LLM 采用了解耦的独立部署策略，将大语言模型（LLM）和多模态模型分开部署。

![图 3：EPD 解耦架构。
该图展示多模态模型的视觉 Transformer（ViT）与大语言模型（LLM）解耦部署架构。架构将视觉编码与语言生成分离，视觉数据先输入独立 ViT 模块生成嵌入，再与文本输入拼接送入 LLM 推理。解耦设计让 ViT 与 LLM 占用独立计算流，高并发请求下实现计算重叠，避免资源争抢。同时降低单卡 GPU 内存占用，适配多图像输入的生产场景，提升多模态推理的吞吐量与延迟表现。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHibV4IQg1bbzWYibkFHNeuQJTc2WP0uZKanVegoSFtZYkiaIXU2ibibZpyxprLiciarpHf7bPRmCHwoyRfnRHBtVaVBzC6LDGh9DnfvKA/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=20)

图 3：EPD 解耦架构。 该图展示多模态模型的视觉 Transformer（ViT）与大语言模型（LLM）解耦部署架构。架构将视觉编码与语言生成分离，视觉数据先输入独立 ViT 模块生成嵌入，再与文本输入拼接送入 LLM 推理。解耦设计让 ViT 与 LLM 占用独立计算流，高并发请求下实现计算重叠，避免资源争抢。同时降低单卡 GPU 内存占用，适配多图像输入的生产场景，提升多模态推理的吞吐量与延迟表现。

### 6.3 性能评估

#### 6.3.1 量化推理性能

> 论文评估了 RTP-LLM 在 Qwen3-32B 模型上的量化性能，比较了不同量化配置（AWQ(FP8)、FP8 KV 缓存、基线）下的表现。

![图 5：Qwen3-32B 模型在不同量化配置下的批处理延迟与精度损失对比：量化性能对比聚焦延迟 - 精度平衡，FP8 KV 缓存量化优势显著，批延迟降 35%-40% 且精度损失极小（PPL 仅差 0.01）。AWQ 量化侧重权重量化，延迟降幅弱于 KV 缓存量化；基线无量化性能最优但显存占用高。这说明 RTP-LLM 自适应量化优先优化 KV 缓存 —— 推理阶段显存核心开销，契合 LLM 内存带宽瓶颈特性，实现效率与精度最优平衡。](https://mmbiz.qpic.cn/sz_mmbiz_png/GxIgp4icchHibR4qYa1FX8Hu03fTM1UbkVYuAgmYZ6xIwuz3uXuRrcQM5POjgLf4sg52zsGwEIdLo2h4dicLibPrnEZ8AlR2UG5XpudaDf5Ha7o/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=21)

图 5：Qwen3-32B 模型在不同量化配置下的批处理延迟与精度损失对比：量化性能对比聚焦延迟 - 精度平衡，FP8 KV 缓存量化优势显著，批延迟降 35%-40% 且精度损失极小（PPL 仅差 0.01）。AWQ 量化侧重权重量化，延迟降幅弱于 KV 缓存量化；基线无量化性能最优但显存占用高。这说明 RTP-LLM 自适应量化优先优化 KV 缓存 —— 推理阶段显存核心开销，契合 LLM 内存带宽瓶颈特性，实现效率与精度最优平衡。

![图 6：Qwen3-32B 模型在不同量化配置下的首 token 延迟与吞吐量对比：量化配置下 TTFT 降幅达 1.9-3.0 倍，吞吐量全面领先，核心源于优化量化内核与内存访问模式。FP8 KV 缓存量化在保持高吞吐的同时，大幅降低首 token 延迟，适配实时交互场景；基线无量化虽精度最优，但延迟高、吞吐受限。该结果验证 RTP-LLM 量化设计并非单纯降精度，而是针对推理阶段内存瓶颈的精准优化，兼顾实时性与吞吐。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHibpNoFiaibLtB9WffN25mHGCOdcMQ92cV8MVhvByx2iaIyDrpuDoFkV8qXaQDnPkhq6ibVHUT0iaZgiaxv7lLrBYVsadCUcYkCcE063Q/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=22)

图 6：Qwen3-32B 模型在不同量化配置下的首 token 延迟与吞吐量对比：量化配置下 TTFT 降幅达 1.9-3.0 倍，吞吐量全面领先，核心源于优化量化内核与内存访问模式。FP8 KV 缓存量化在保持高吞吐的同时，大幅降低首 token 延迟，适配实时交互场景；基线无量化虽精度最优，但延迟高、吞吐受限。该结果验证 RTP-LLM 量化设计并非单纯降精度，而是针对推理阶段内存瓶颈的精准优化，兼顾实时性与吞吐。

#### 6.3.2 多模态推理性能

> 论文使用公共 GQA 基准评估了 RTP-LLM 的 ViT 分离（EPD）性能，比较了 Qwen/Qwen2.5-VL-7B-Instruct 模型在不同框架下的表现。

![图 7：不同框架下，Qwen/Qwen2.5-VL-7B-Instruct 模型在 GQA 数据集上的性能与 GPU 显存占用对比：EPD 解耦让 RTP-LLM 吞吐量达 6288 token/s，是 vLLM 的 2.52 倍，显存分布高度优化。GPU0 仅 9279MB 用于 ViT 编码，GPU1 集中资源做文本生成，避免传统框架显存均匀分配的浪费。ViT 与 LLM 并行执行实现计算重叠，减少等待延迟，同时适配多图像输入的视觉问答场景，证明解耦设计是多模态推理的高效路径。](https://mmbiz.qpic.cn/mmbiz_png/GxIgp4icchHicXxKV3GfkU9FMgvcVoRcqHTuibKRs9yn8yWSozB3SX6sr2c6XUF5YKoeZo0UbSianhNQicrAgzzaMGloOs9rnpMMQibqcso01k34w/640?wx_fmt=png&from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=23)

图 7：不同框架下，Qwen/Qwen2.5-VL-7B-Instruct 模型在 GQA 数据集上的性能与 GPU 显存占用对比：EPD 解耦让 RTP-LLM 吞吐量达 6288 token/s，是 vLLM 的 2.52 倍，显存分布高度优化。GPU0 仅 9279MB 用于 ViT 编码，GPU1 集中资源做文本生成，避免传统框架显存均匀分配的浪费。ViT 与 LLM 并行执行实现计算重叠，减少等待延迟，同时适配多图像输入的视觉问答场景，证明解耦设计是多模态推理的高效路径。

RTP-LLM 的量化技术实现了 35%-40%的批处理延迟减少和 1.9 倍至 3.0 倍的 TTFT 改善，而多模态 EPD 分离架构实现了 1.86 倍至 2.52 倍的吞吐量提升和 2.12 倍至 2.36 倍的 TTFT 降低，这些优化使 RTP-LLM 能够高效支持从密集模型到多模态模型的各种架构。

## 七、相关工作

近年来，LLM 推理系统领域取得了显著的进展，涌现出了许多优秀的解决方案。这些工作从不同角度解决了 LLM 推理面临的挑战，但大多数都集中在优化孤立的组件，而忽视了生产规模部署所需的系统级交互。

### 7.1 内存管理技术

> 内存管理是 LLM 推理系统中最关键的挑战之一。传统的内存分配策略设计用于固定大小的分配，被证明不足以满足 LLM 推理的动态和可变大小的内存需求。

- vLLM 引入了 PagedAttention，这是一种革命性的方法，将 KV 缓存视为分页虚拟内存系统的集合。通过将 KV 缓存划分为可以在不同请求之间分配、释放和共享的固定大小页面，PagedAttention 显著提高了内存效率和利用率。
- SGLang 提出了 RadixAttention，基于前缀树结构进一步提高了缓存命中率，特别适用于多轮对话场景。RadixAttention 通过共享公共前缀的 KV 缓存，减少了重复计算，提高了系统吞吐量。

然而，这些系统的缓存管理主要局限于单个节点的 GPU 内存，缺乏对多级缓存和跨节点缓存共享的支持。RTP-LLM **通过实现跨越 GPU 内存、本地 CPU 内存、远程 CPU 内存和分布式存储的四级缓存架构，以及基于统一哈希映射的全局前缀匹配** ，显著扩展了缓存的范围和效率。

### 7.2 推理调度技术

> 连续批处理是 LLM 推理调度的一项重要创新，它允许动态请求调度而不影响生成质量。

- Orca 系统首次引入了连续批处理的概念，通过在每个解码步骤替换已完成的请求，提高了 GPU 利用率。

然而，传统的连续批处理将 Prefill 和 Decode 视为相同的工作负载，采用统一的调度算法，这导致了资源的不合理分配。近年来，Prefill-Decode 分离架构逐渐受到关注。

- Splitwise 提出了使用阶段拆分的高效生成式 LLM 推理，将 Prefill 和 Decode 阶段分离到不同的计算资源上。
- DistServe 进一步发展了这一思想，通过分离 Prefill 和解码来优化大语言模型服务的有效吞吐量。

RTP-LLM 在这些工作的基础上，实现了 **更完善的 Prefill-Decode 分离架构，包括动态流量调度、智能负载均衡和高效的跨节点 KV 缓存传输** 。此外，RTP-LLM 还 **结合了多级缓存管理和推测解码技术** ，形成了一个全面的解决方案。

### 7.3 推测解码技术

> 推测解码通过引入并行 token 验证机制，解决了 LLM 推理中的顺序生成瓶颈。原始的推测解码方法使用一个较小的草稿模型生成候选 token ，然后使用目标模型并行验证这些 token 。

近年来，出现了许多改进的推测解码算法。

- Medusa 通过在目标模型上添加多个预测头，避免了对单独草稿模型的需求。
- Eagle 提出了一种新的自回归头训练方法，用于未来隐藏状态预测。
- Prompt Lookup 通过 n-gram 匹配从输入提示中提取候选 token ，特别适用于提取式场景。

然而，大多数现有的推理系统 **只支持特定的推测解码算法，缺乏模块化和可扩展性** 。RTP-LLM 实现了一个全面的推测采样框架，支持多种算法，并通过 C++级别的优化消除了 Python 到 C++的调用开销，显著提高了性能。

### 7.4 多模态推理技术

> 随着多模态模型的兴起，如何高效处理图像和文本的联合推理成为了一个重要的研究方向。传统的多模态推理系统通常将 ViT 和 LLM 耦合在同一个节点上，导致资源争用和性能下降。

**EPD 分离架构提出了将 ViT 和 LLM 独立部署的思想，允许它们使用单独的流进行推理** ，避免资源争用。RTP-LLM 在这一思想的基础上，实现了更完善的多模态支持，包括 ***对 LLaVA 和 Qwen-VL 等主流架构的支持，以及优化的视觉特征提取和文本生成流水线。***

### 7.5 与 RTP-LLM 的比较

> 与现有的推理系统相比，RTP-LLM 具有以下几个关键优势：

1. 全栈式系统设计：RTP-LLM 从模型加载、流量调度、缓存管理到执行优化进行了全栈式重构，而不是仅仅优化孤立的组件。
2. 生产级可靠性：RTP-LLM 经过了阿里巴巴大规模生产环境的验证，服务超过 1 亿用户，具备企业级的运维特性，如容错、滚动更新和按请求性能隔离。
3. 全面的模型支持：RTP-LLM 支持从 8B 到 600B+参数的各种模型架构，包括密集模型、MoE 模型和多模态模型。
4. 卓越的性能：在多个关键指标上全面超越 vLLM 和 SGLang，特别是在模型加载速度、TTFT 延迟和多模态推理性能方面。

RTP-LLM 在吸收现有技术优点的基础上， **通过系统级的集成设计和生产环境的实战打磨，解决了现有系统在大规模部署时面临的诸多挑战** ，为工业级大模型推理提供了一个全面、高效、可靠的解决方案。

## 总结与展望

> RTP-LLM 作为阿里面向工业级 LLM 部署的全栈推理引擎，通过 Prefill-Decode 解耦、分层 KV 缓存、模块化投机解码、自适应量化、解耦多模态五大核心技术，系统性解决了 GPU 利用率低、KV 缓存溢出、异构适配差、迭代缓慢四大痛点。 **真实生产验证显示，其性能全面领先 vLLM、SGLang，已支撑阿里核心电商、客服等场景，服务超亿级用户。**

未来，RTP-LLM 将聚焦三大方向：

1. **稀疏注意力优化** ：集成 DSA 等稀疏注意力机制，进一步提升超长上下文推理效率。
2. **异构硬件扩展** ：深化 AMD、ARM、国产芯片适配，构建全平台推理生态。
3. **社区生态共建** ：依托开源社区，迭代优化算法，拓展模型支持范围。

RTP-LLM 的开源，为工业级 LLM 推理提供了可复用的全栈范式，推动大模型技术从实验室走向大规模产业落地，为 AI Infra 领域提供了重要参考。

相关推荐

- [速度：大模型推理的下一个 Scaling Law，深度解析 TileRT 高性能推理引擎及 GLM-5.1 生产级实践](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447901538&idx=1&sn=f3055a8ef80e376c63335de3ccce42ad&scene=21#wechat_redirect)
- [一端写就，全端运行！端侧推理引擎 OmniInfer：让所有设备实现“触手可及”的大模型端侧推理体验](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447899937&idx=1&sn=47fdb2f53cc9521e9db4527253cae0d1&scene=21#wechat_redirect)
- [老卡跑原生FP8模型！硬件成本降50%+！国产大模型推理引擎 Chitu（赤兔）全解析与实践指南](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447895244&idx=1&sn=57c31bb4e6c4d1126388508a43b53107&scene=21#wechat_redirect)

交流加群请在 NeuralTalk 公众号后台回复：加群

GPU · 目录

继续滑动看下一个

NeuralTalk

向上滑动看下一个