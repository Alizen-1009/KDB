---
title: "KV Cache 研习社 · 深潜 | LMCache、Mooncake、FlexKV——三条路线谁在改写 KV 的命运？"
source: "https://mp.weixin.qq.com/s/utA9JDybd7nYVkoj2T-2JA"
author:
  - "[[KV Cache 研习社]]"
published:
created: 2026-11-09
description: "三个开源项目用三种架构哲学回答同一个问题：KV Cache 不该用完即扔。LMCache 做引擎之上的独立 KV 层（TTFT 降 13.6×），Mooncake 做集群级 PD 分离 + RDMA 分布式池（Kimi 生产吞吐 +525…"
tags:
  - "clippings"
---
KV Cache 研习社 Q仔学AI *2026年9月14日 08:18*

KV Cache 研习社 · 原理深潜 · KV Cache 存储工程

> **KV Cache 存储工程** | 三条路线横评 | KV 资产化：从用完即扔到可复用数据

伺候一个 128K 上下文的用户，KV Cache 就要吃掉 63 GB 显存——刚好塞满一张 H100。多一个并发用户？再买一张卡。多 100 个？买一个机柜。而机柜里 80% 的 CPU 内存和 SSD 在空转。

这就是 2025 年推理集群的荒诞：GPU 一半算力在等数据搬运，一半显存被 KV 塞满，旁边的 CPU 内存和 SSD 却完全在空转。三个开源项目——LMCache、Mooncake、FlexKV——分别从不同角度给出了答案，它们的共同信念只有一条：KV Cache 不该用完就扔。

## Keynote

- **同一个痛点**
	：KV 被锁在单个引擎进程里，请求结束即丢弃，同一个 system prompt 被几万个请求重复 Prefill
- **三种架构哲学**
	：LMCache 做"层"（引擎之上的独立 KV 中间层），Mooncake 做"池"（集群级 PD 分离 + RDMA 分布式 KV 池），FlexKV 做"管道"（四级存储逐级卸载流水线）
- **生产数据说话**
	：LMCache TTFT 降 **13.6×** （Qwen3-235B 实测），Mooncake 请求容量 + **525%** （FAST 2025 最佳论文），FlexKV 合入 **NVIDIA Dynamo** 成为官方卸载方案
- **不是三选一，是三层叠加**
	：它们解决的是不同维度的问题——引擎内碎片、节点间隔离、存储层级断裂——可以组合使用

## 一、KV 为什么被当成废弃物？

在传统推理架构里，KV 和请求同生共死——请求结束，KV 全部丢弃。这带来了三个互相叠加的问题。

第一个是重复计算。同一个 system prompt 被 1000 个请求共享，但每个请求都要各算一遍 KV，算完就扔，下次再来还是各算一遍。第二个是碎片化：8 个数据并行（DP）副本各自维护独立的 KV 缓冲区，400 GB CPU 内存被切成 8 份互不相通——A 副本算过的 KV，B 副本用不了。第三个是搬不动：KV 被锁在单台机器的 GPU 显存里，跨节点不能共享，换引擎不能通用。

这三个问题的根源可以归结为一个概念： **命运共享（fate-sharing）** ——KV 的生死和引擎进程绑在一起。三个项目分别从这三个维度给出了解法。

## 二、LMCache：把 KV 从进程里拆出来

### 核心思路：独立的 KV 守护进程

LMCache 的做法是把 KV Cache 管理从推理引擎里拆出来，变成一个独立的守护进程（daemon）。引擎进程崩了，KV 还在；切换引擎，KV 照用。

> **图 1：LMCache In-Process vs MP 模式架构对比** 来源：LMCache 官方博客（2026-04） ![LMCache 架构对比](https://mmbiz.qpic.cn/mmbiz_png/kr8WB9wsIWfC97icNah0eqSoB2pEW1K895oHoH96PoneJN8XV2ra2fE6qpn1EUcBpfgoZfhB8SHTunZf2nB7n8mKlibSEl45qCXPbZiaMbn7J0/640?from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

### 技术细节：MP 模式怎么打破命运共享

2026 年 4 月发布的 Multi-Process（MP）模式是 LMCache 和"引擎内置卸载"拉开差距的关键一步。传统方案里，vLLM 的 8 个数据并行（DP）副本各自维护独立的 KV 缓冲区，即使总共分配了 400 GB CPU 内存用于卸载（每副本 50 GB），KV 也是碎片化分布的——A 副本算过的 KV，B 副本用不了。

MP 模式的核心变化是把 400 GB 内存统一成一个池。所有 DP 副本通过 LMCacheMPConnector 动态注册到同一个 LMCache Server，共享同一块 host memory，同一个 system prompt 的 KV 只算一次，8 个副本都能复用。

\# 启动 LMCache MP Server，分配 400GB host memory lmcache server --l1-size-gb 400 --eviction-policy LRU # vLLM 接入，所有 DP rank 共享同一个 KV 池 vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \\ --data-parallel-size 8 --enable-expert-parallel \\ --kv-transfer-config '{"kv\_connector":"LMCacheMPConnector","kv\_role":"kv\_both"}'

存储后端是可插拔的：L1 是 CPU 内存（延迟最低）、L2 是本地 SSD（持久化）、远端是 Redis/Mooncake/S3（跨节点）。KV 从 GPU HBM 卸载到低成本介质，下次请求命中时直接取回，不用重算 Prefill。

### 实测数据：13.6 倍 TTFT 改善从哪来

在 Qwen3-235B-A22B-Instruct-2507-FP8 + 8×H100 的基准测试中（vLLM 0.18.1 + LMCache 0.4.3-dev，多轮对话 QPS 2.0，持续 120 秒），MP 模式的 TTFT 均值从 3.98 秒降到 0.29 秒，p99 从 13.55 秒降到 1.30 秒，解码吞吐均值从 9.8 tok/s 提升到 37.5 tok/s。

> **图 2：LMCache MP 模式 vs 引擎内置卸载——TTFT 对比** 来源：LMCache 官方博客（2026-04） ![TTFT 对比](https://mmbiz.qpic.cn/sz_mmbiz_png/kr8WB9wsIWffTna1jic5ibc8IFEL1aa77QTX18whiaicLjo2eRJ49DN0XcqjanSFMGQUSAUue6lHDb0N9uHzzQibicElJiaszMTS7KI5EHOHMFlrAc/640?from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1)

13.6 倍的 TTFT 改善，来源不是更快的 GPU，而是 **不再重算** ——MP 模式把命中率从碎片化的低水位拉到统一池的高水位，GPU 算力被释放给 Decode 而非重复 Prefill。LMCache 不替代引擎，而是在引擎之上加了一层 KV 管理，就像 CDN 不替代 Web 服务器，而是在前面加了一层缓存。

## 三、Mooncake：把 KV 池化到整个集群

### 核心思路：PD 分离 + RDMA 传输引擎

Mooncake 的做法比 LMCache 更激进——不只是把 KV 从引擎里拆出来，而是把 Prefill 和 Decode 分到不同的机器上，再用一个高速传输引擎让 KV 在它们之间流动。

> **图 3：Mooncake 架构——PD 分离 + 分布式 KV 池** 来源：自画（基于 FAST 2025 论文 Figure 2） ![Mooncake 架构](https://mmbiz.qpic.cn/mmbiz_png/kr8WB9wsIWeua8zXCaK7paNt7kHltsSyUiaCGgKLcKYLUDRL7m06qibDSJbOtbQwibAu6fic0M1CgANhD7mD1OvhFrQiaibUxcib7r6BXERB0GFiaek/640?from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)

PD 分离之所以值得做，是因为 Prefill 和 Decode 的资源需求完全相反：Prefill 是计算密集型，一次性读完所有输入 token，吃 FLOPS；Decode 是内存密集型，每步只生成 1 个 token，但需要持续读 KV，吃 HBM 带宽。两者挤在同一张 GPU 上，Prefill 的大批量计算会打断 Decode 的逐字生成，导致延迟抖动——就像让一个厨师同时负责切菜和摆盘，切菜的大动作会震歪正在摆的盘。分开之后，Prefill 节点堆算力，Decode 节点堆带宽，各取所长。

### 技术细节：Transfer Engine 的四个关键设计

Transfer Engine 是 Mooncake 的心脏，一个基于 RDMA 的高速传输引擎，让 KV 在 Prefill 节点、Decode 节点和分布式存储之间以接近内存带宽的速度流动。

> **图 4：Transfer Engine 零拷贝 vs 传统四步拷贝** 来源：自画 ![Transfer Engine 零拷贝对比](https://mmbiz.qpic.cn/mmbiz_png/kr8WB9wsIWfTXRRTyEOd2mzllFspOZA5oU3GOF5MVAsoXeLuMSAUOg9xzwmSDKCYiaMtyYFc2L7AEf1VyRJQ4JX1z3VDA5iavDzqoSILIibOgo/640?from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=3)

第一个设计是零拷贝：数据从源节点的 VRAM/DRAM 直接传到目标节点的 VRAM/DRAM，不经过 CPU 内存中转。传统方案需要 GPU→CPU→网络→CPU→GPU 四次拷贝，Transfer Engine 只需一次 RDMA 直传。第二个设计是多 NIC 并行加条带化（Striping），单个节点的多张网卡同时传输，大 KV 对象被切分成多个条带并行传输到多个目标节点。Kimi 生产环境每节点 8 张 100Gbps 网卡，聚合带宽 800Gbps。第三个设计是对象级存储，KV Cache 被封装为对象而非裸字节流，支持 Get/Put/List/Del 操作，Master Node 集中管理对象到缓冲区的映射，支持前缀匹配——新请求进来时先在 Master Node 查最长匹配前缀，决定从哪个存储节点取 KV。第四个设计是预测性早拒绝，当集群过载时，调度器不是让所有请求都排队等死，而是基于预测提前拒绝那些注定无法满足 SLO 的请求，避免浪费算力在注定超时的请求上。这是 Mooncake 和其他方案最大的架构差异，它从设计之初就假设"请求永远多于算力"。

### 生产实测：FAST 2025 最佳论文的数据

Mooncake 不是论文里的原型系统，而是 Kimi 的生产平台。FAST 2025 最佳论文的实测数据显示，真实负载下请求容量提升 75%（对比传统同机架构），模拟极端场景下吞吐提升最高 525%（同时满足 SLO），KV 池利用了整个集群中闲置的 CPU 内存和 SSD——这些资源在传统架构里利用率不到 20%。

2025 年 7 月，Mooncake 支撑了 Kimi K2（1T 参数 MoE 模型）在 128×H200 上的部署：PD 分离加大规模专家并行，Prefill 吞吐 224K tok/s，Decode 吞吐 288K tok/s。这是当时公开报道的最大规模 PD 分离部署，128 张 H200 被拆成 Prefill 集群和 Decode 集群，中间用 Transfer Engine 的 RDMA 网络连起来。

### 生态扩张：正在成为 KV 传输的 RDMA 标准

Mooncake 的 Transfer Engine 已经变成了推理生态的传输底座：

> **图 5：Mooncake Transfer Engine 生态扩张时间线** 来源：自画 ![Mooncake 生态时间线](https://mmbiz.qpic.cn/sz_mmbiz_png/kr8WB9wsIWdQRKZVLaqic7Sbx02URGP48zdibibpH6IHxh50EWAw6KnR96nnSDVPZz9TaVHgVZZS717f77qJia8kSZnqwrhDgCpCDfp5khAdvII/640?from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4)

一个值得注意的趋势是，Mooncake 正在成为 KV 传输领域的 NCCL——就像 NCCL 定义了 GPU 间的梯度同步协议，Mooncake 正在定义 KV 的跨节点传输协议。

---

LMCache 解决了"KV 别锁在进程里"，Mooncake 解决了"KV 别锁在单机里"。但还有一个问题没解决：KV 别锁在单一介质里。这就是 FlexKV 的战场。

## 四、FlexKV：四级存储，逐级降本

### 核心思路：GPU→CPU→SSD→远端的卸载流水线

LMCache 和 Mooncake 都在解决"KV 放哪里"的问题，但现实是存储介质之间存在巨大的成本-性能鸿沟：

> **图 6：四级存储介质——容量与带宽的鸿沟** 来源：自画 ![四级存储介质对比](https://mmbiz.qpic.cn/mmbiz_png/kr8WB9wsIWcqZUR9BLOHQFcyJUgYxfPuJ9628Gx6JBsY0mEKJuicRYceHfy4Lp34Dib1ibpACJzCGXJS5FZ7G41AgiamflI5xWiaEY7OsSfZm8Jg/640?from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)
> 
> **图 7：FlexKV 四级卸载架构** 来源：FlexKV GitHub 官方仓库 ![FlexKV 架构](https://mmbiz.qpic.cn/mmbiz_png/kr8WB9wsIWcyWRqvHISqfbNTINTfhwoibyjccuHdib17wr9oML1ITUnUicWEtMM721PUe9JCdzoGtHiatTMZror4Jqv8yX4vfutECLJ4n41cooE/640?from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=6)

FlexKV 的设计目标是同时利用四级存储，让热 KV 留在 HBM、温 KV 沉到 DRAM、冷 KV 落 SSD、归档 KV 进远端，按温度自动分层。

### 技术细节：控制平面与数据平面分离

FlexKV 由三个核心模块组成，分工清晰：

> **图 8：FlexKV 三模块分工——存在哪 → 往哪搬 → 怎么搬** 来源：自画 ![FlexKV 三模块分工](https://mmbiz.qpic.cn/mmbiz_png/kr8WB9wsIWf6r83aDYxrXZGSJDJBoDbhSo1f8t8aia48meicvEhar1LficVXlbWVmr9ic1PX5iaC05KhL4MlIZDB4qFzqS4vP47N7ZUtLvFHhmk8/640?from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=7)

### 异步 API：让搬运与计算重叠

FlexKV 的 get/put 都是异步的，这是它能"免费"提供四级存储的关键。get 异步意味着匹配和数据传输可以与前面的计算重叠（预取），put 异步意味着 GPU→CPU 的数据拷贝可以与后续计算重叠，CPU→SSD→远端的搬运完全由 TransferEngine 异步处理，对主进程透明。只要调度器能提前知道下一个请求需要什么 KV，就能在它到达之前把 KV 从低级存储搬到高级存储——这是流水线化的预取与淘汰，而不是等 Attention 算到一半再去取。

### 分布式 KV 复用：跨节点共享

2026 年 1 月集成 Mooncake Transfer Engine 后，FlexKV 打通了本地四级卸载加跨节点 RDMA 复用的完整链路。分布式 RadixTree 让每个节点维护全局索引的本地快照，避免中心化瓶颈；租约机制保证跨节点数据传输期间的数据有效性；GMS（Global Meta Store）通常用 Redis，各节点定期上传本地索引。

### 生产验证

FlexKV 在 2026 年 3 月合入 NVIDIA Dynamo（PR #5858），同月合入 vLLM mainline（PR #34328），7 月合入 SGLang mainline（PR #29701）——三大推理框架全部原生支持，不需要打补丁。2026 年 7 月，FlexKV 已经支持 DeepSeek-V4 的异构 KV 组（C4/C128/indexer）、FullKV + SWA 双缓存、压缩状态 sidecar、逐层恢复，这是目前对最新模型架构支持最完整的方案。

腾讯云生产环境验证数据显示，可用 KV 容量从 HBM 的 80 GB 扩展到 TB 级，对比纯 HBM 方案并发用户数提升 3-10×，对比纯 SSD 卸载方案延迟降低 67%（热 KV 留在 DRAM 层）。如果 LMCache 是 CDN、Mooncake 是跨城物流，FlexKV 就是仓库里的四级货架——热的放门口、冷的放深处、按温度自动轮转。

## 五、三条路的全景对比

> **图 9：KV Cache 三条路线全景对比** 来源：自画 ![三项目全景对比](https://mmbiz.qpic.cn/sz_mmbiz_png/kr8WB9wsIWfFdnvqSO3Pj1ibvLejoayE5Dy2fcSYeMOSHxJ10C1I9Rfye91jeiad7MfvQyibOd3NEEkwlvrP3TUatFWVMhWD7ZlEMSj8rvm24Q/640?from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=8)

### 怎么选

**多引擎混部** （vLLM + SGLang 共存）→ **LMCache** ，唯一真正引擎无关的方案。 **长上下文 + 高并发** （Kimi 类场景）→ **Mooncake** ，PD 分离 + 集群 KV 池，吞吐最优。 **单机显存不够但不想买更多 GPU** → **FlexKV** ，四级卸载，利用闲置 CPU/SSD。 **极致性能 + 全套 NVIDIA 生态** → **FlexKV + Mooncake** ，Dynamo 原生 + RDMA 传输。 **最新模型架构** （DeepSeek-V4 等）→ **FlexKV** ，异构 KV 组支持最完整。 **没有 RDMA 网络** → **LMCache** 或 **FlexKV** ，纯 TCP 即可运行。

### 组合使用

三者不是互斥的，2026 年的生产趋势是叠加使用：LMCache 做引擎内 KV 管理、Mooncake Transfer Engine 做跨节点传输（2025 年 5 月已联合发布）；FlexKV 做单机四级卸载、Mooncake TE 做跨节点 RDMA 复用（2026 年 1 月集成）；三者联合则是 LMCache 做引擎抽象层、FlexKV 做本地多级存储、Mooncake 做集群级 KV 池，构成全栈 KV 资产管理。

## 六、两条路线的合流：平台侧 + 算法侧

> **图 10：平台侧 + 算法侧合流路线图** 来源：自画 ![平台侧算法侧合流](https://mmbiz.qpic.cn/sz_mmbiz_png/kr8WB9wsIWcTJJFBnwcgeqMzpsTzldic5icqP28ZlRLdpclEqJhvTUfyZMalRIVtoF7D9LCQ1ZibY3uibDkqcIzQDySibiaq5EnmQpd6MWGUrLAK0/640?from=appmsg&watermark=1&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=9)

三个项目看似解决不同的问题，但背后是同一个范式转移：KV Cache 不再是推理过程的临时状态，而是一种可持久化、可检索、可共享的数据资产。LMCache 把它变成独立于引擎的微服务，Mooncake 把它变成跨节点流动的集群资源，FlexKV 把它变成按温度自动分层的存储对象——这是平台侧的演进。

与此同时，算法侧也在同步演进。MLA 和 GQA 砍掉 KV 的维度与头数，KV 量化从 FP16 压到 FP8、INT4 让体积减半再减半，线性注意力直接消灭 KV 随上下文增长这一项。算法侧的目标是让 KV 本身更小、更可控。

两条路线正在合流。FlexKV 已经支持 DeepSeek-V4 的异构 KV 组（C4/C128/indexer），不再只是 block 级存取；LMCache 推出 KV Editing SDK，允许修改已缓存的 KV 而非重算；Mooncake 的 Transfer Engine 甚至被用于模型权重传输，在 SGLang RL 场景下把 1T 参数模型的权重更新从 53 秒压到 7.2 秒。

下一步方向是从"粗粒度存取"到"细粒度编辑"。当前三个项目的操作粒度还是 block 级（整个请求的 KV 块），但算法侧的需求越来越细——token 级编辑（如 RAG 场景修改检索片段）、head 级管理（不同 attention head 的 KV 有不同生命周期）、维度级区分（MLA 的低秩向量与全秩向量分开管理）。当平台侧能支持这些精细操作，算法侧的压缩与编辑才能真正落地，这是三个项目的下一个战场。

**下一篇，我们聚焦 KV 感知调度** ——当 KV 变成了可共享的资产，调度器该怎么利用这个信息？llm-d 的实测数据是：精确前缀路由 vs 随机调度，TTFT 差距 **170 倍** 。

## 参考文献

1. **LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference**
	(Cheng et al., arXiv 2510.09665, 2025)
2. **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving**
	(Qin et al., FAST 2025 Best Paper)
3. **FlexKV: 面向高性能分布式推理的 KVCache Manager**
	(Tencent TACO + NVIDIA, 2026)
4. **LMCache's New Architecture Boosts MoE Inference Performance by 10×**
	(LMCache Blog, 2026-04)
5. **Deploying Kimi K2 with PD Disaggregation and Large-Scale Expert Parallelism**
	(LMSYS Blog, 2025-07)
6. **Dynamo: NVIDIA's Production LLM Inference Framework**
	(NVIDIA Documentation, 2026)

## 延伸阅读

- 上一篇：一文读懂 LLM 推理引擎：vLLM 六大模块与 KV Cache 六层栈
- 下一篇预告：KV 感知调度——llm-d 如何用前缀路由把 TTFT 降 170 倍
- Wiki 参考：concept-kv-cache-offloading、concept-three-tier-scheduling

KV Cache 研习社

从显存到固态，拆解 LLM 推理的每一层存储