---
title: "vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署"
source: "https://mp.weixin.qq.com/s/UpbZf12Ap-mNJfKe2GsLLw"
author:
  - "[[AFD Plugin Team]]"
published:
created: 2026-07-25
description: "vLLM AFD Plugin 是一个实验性外部插件，把 Attention-FFN 分离（AFD）带入 MoE 推理——将 Attention 与 FFN 拆成两个独立部署的服务，让二者按各自的负载特性独立扩展，同时完整保留 vLLM 的请求生命周期与 OpenAI 兼容接口。"
tags:
  - "clippings"
---
AFD Plugin Team vLLM *2026年7月24日 17:13*

> vLLM AFD Plugin 是一个实验性外部插件，把 Attention-FFN 分离（AFD）带入 MoE 推理——将 Attention 与 FFN 拆成两个独立部署的服务，让二者按各自的负载特性独立扩展，同时完整保留 vLLM 的请求生命周期与 OpenAI 兼容接口。插件同时支持 NVIDIA GPU 与昇腾 NPU（Ascend），提供同步 / 异步连接器、DeepSeek V2/V3 系列模型封装，以及 eager、graph、dual-batch 等执行路径。适合关注 MoE 推理部署、异构硬件、以及 Attention/FFN 独立扩展的推理工程团队参考。

我们很高兴推出 **vLLM AFD Plugin** <sup>[1]</sup> ——一个实验性外部插件，把 **Attention-FFN 分离（Attention-FFN Disaggregation，AFD）** 带入 vLLM。

vLLM AFD Plugin 面向 Mixture-of-Experts（MoE）模型引入 AFD：把 Attention 与 FFN 拆成两个独立部署的服务。插件完整保留 vLLM 现有的请求生命周期与 OpenAI 兼容服务接口，同时允许 Attention 与 FFN 两条路径各自独立扩展。

该项目目前支持 NVIDIA GPU 与昇腾 NPU，提供同步与异步连接器、DeepSeek V2/V3 系列模型封装，以及在明确验证范围内的 eager、graph 与 dual-batch 执行路径。

> **注意**  
> 该项目仍处于实验阶段，需要在不同硬件后端上做更多大规模测试。

## 为什么要做 Attention-FFN 分离？

MoE 推理在每一个 transformer 层里，都把两类差异极大的计算揉在一起。Attention 是有状态的，与请求调度、KV cache 紧密耦合；而 FFN（专家路径）则由路由到的专家计算与 all-to-all 通信主导。当两条路径共享同一套 worker 拓扑时，服务系统只能为需求截然不同的两类负载做出同一套扩展与执行选择。

要把这种拆分做到可用，需要解决以下几个系统设计上的挑战：

1. 1\. **Attention 与 FFN 的扩展需求不同。** Attention 的容量取决于请求状态、序列长度和 KV cache 压力；专家容量则取决于 token 路由和专家负载。服务系统应当允许两条路径使用不同的 rank 拓扑、独立扩展，而不是强制它们共用一套布局。
2. 2\. **Attention 与 FFN 的运行时职责不同。** Attention 需要调度、KV cache 协调和采样；FFN 执行只需要激活值、路由元数据，以及一条把专家输出送回来的通道。把服务拆开后，FFN 侧就可以作为一个轻量的、由连接器驱动的守护进程运行。
3. 3\. **通信是后端相关的。** CUDA 与昇腾暴露的是不同的集合通信库、graph 运行时和优化过的 MoE 算子。一套通用的连接器契约让面向模型的流程保持稳定，同时允许每个后端自行掌控其数据通路。
4. 4\. **通信与计算可以重叠。** 异步 dispatch 与 MoE ubatching（微批）能让相互独立的阶段重叠执行，而不是把所有专家计算都串行地堵在 Attention 路径后面。

综合起来，这些挑战定义了 AFD 的核心设计目标：保持 vLLM 面向请求的 Attention 路径不变，同时把 FFN 执行移到一个狭窄的连接器接口之后，让它能够独立地扩展、通信和执行。

## 架构内部

![图片](https://mmbiz.qpic.cn/mmbiz_png/icV2Iiao3PgouDopxibVMPHKM3l9soQJS7NyicibLNnK8VBiaLhRkUJGCIxB7JicEkCpPuawiaYQPJIG7AGtjza2lf1dOUcW0WoHRYSYvV8RG8XwfDI/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=0)

vLLM AFD Plugin 运行时架构

插件通过 vLLM 的 `vllm.general_plugins` 入口点和标准的 `--additional-config` 通道接入，不需要改动 vLLM 源码树。

运行时由三个主要部分组成：

- • **Attention 服务。** Attention worker 保留 vLLM 的调度器、KV cache、批处理、模型生命周期和采样路径。一个由插件提供的 model runner 会把 AFD 元数据装入 forward context，并把 data-parallel、ubatch、层、graph 等状态发布给 FFN 侧。
- • **FFN 服务。** FFN worker 没有请求流量、没有调度器、也没有 KV cache。一个后台循环接收元数据和激活值，在插件提供的模型封装上调用 `compute_ffn_output()` ，再把结果发回 Attention。请求始终发往 Attention 的 API server。
- • **连接器层。** 在每个切分层，连接器把 Attention 的隐藏状态连同 FFN 服务所需的执行元数据一起传输过去，再把算出的 FFN 输出返回。一个后端中立的连接器接口定义了这次交换，同时允许每个后端实现自己的通信与运行时优化。

这个接入面被刻意设计得很小。vLLM 继续掌管其现有抽象适配的服务控制平面，而插件负责提供 AFD worker、model runner、连接器、元数据、模型切分点，以及一小组按版本限定的兼容性补丁的具体实现。

### 连接器与后端支持

| 连接器 | 后端 | 执行方式 | 推荐阶段 | Graph 支持 |
| --- | --- | --- | --- | --- |
| `P2pNcclAFDConnector` | GPU | 同步 P2P | Decode | `FULL_DECODE_ONLY`  CUDA graph |
| `CAMP2pAFDConnector` | NPU | 同步 CAMP2P/HCCL | Decode | `FULL_DECODE_ONLY`  ACL graph |
| `CAMAsyncAFDConnector` | NPU | 异步 CAM | Prefill | 暂不支持 |

各连接器共享同一套高层交换流程——Attention 输出送往 FFN，FFN 输出送回 Attention。后端包各自独立，从而让 CUDA graph 行为、ACL graph 行为、NCCL 通信和昇腾自定义算子之间互不渗透。

### 已支持特性

- • **原生 vLLM 服务接口。** 现有 vLLM 用户依然用 `vllm serve` 启动，把请求发往 OpenAI 兼容端点，并通过 `--additional-config` 配置运行时。
- • **GPU 与 NPU 两套实现。** GPU worker 扩展 vLLM v1 类，NPU worker 直接扩展 vLLM-Ascend 类。共享行为落在配置、拓扑、元数据和连接器契约里，而不是跨设备的继承关系。
- • **面向 decode 吞吐的同步 AFD。** `P2pNcclAFDConnector` 与 `CAMP2pAFDConnector` 同步地交换 Attention 激活值与 FFN 输出，让两种角色在以吞吐为导向的 decode 部署中独立扩展。它们当前的 graph 路径分别在 CUDA 和 ACL 上采用 `FULL_DECODE_ONLY` 语义。
- • **面向 prefill 的异步 AFD。** `CAMAsyncAFDConnector` 使用 CAM 异步 dispatch 与 combine 算子，把 prefill 阶段的 Attention rank 与专家 worker 解耦。配合 AFD 管理的 MoE ubatching，它让相互独立的 Attention 与 FFN 阶段重叠执行，减少流水线停顿。该路径目前面向 PD 分离部署中的 prefill 阶段，暂不支持 graph 执行。
- • **MoE 模型接入。** 插件为 DeepSeek V2/V3 系列架构（含 DeepSeek V3.2）与 GLM MoE DSA 注册了封装。封装把 Attention 与 FFN 计算分离暴露出来，同时复用上游的层实现。
- • **Graph 与 ubatching 路径。** 同步的 GPU 与 NPU 连接器支持 decode-only 的 graph capture。Dual Batch Overlap（DBO）支持恰好两个 ubatch，CAM async 则为其 prefill 路径提供由 AFD 管理的 MoE ubatching。

## 性能一瞥

### 使用 CAMP2pAFDConnector 的同步 AFD decode 吞吐

同步 decode recipe（见 afd-plugin PR #67 <sup>[2]</sup> ）在昇腾 910C 上，针对 DeepSeek-V3.2 W8A8 模型，将常规 EP64 部署与基于 `CAMP2pAFDConnector` 的 AFD 部署做了对比。该基准测量的是饱和状态下的 decode 吞吐，而非在线服务延迟。

| 部署 | 物理拓扑 | 总 die 数 |
| --- | --- | --- |
| EP64 | DP64, EP64, TP1 | 64 |
| 48A16F | 48 个 Attention rank, 16 个 FFN rank | 64 |
| 64A16F | 64 个 Attention rank, 16 个 FFN rank | 80 |

> **注意**  
> 这些是受控的性能结果，不是精度结果，也不是生产服务结果。受限于机器可用性，物理上的 48A16F 与 64A16F 部署分别模拟了逻辑上的 192A64F 与 256A64F 规模。实验用一个确定性的强制均衡循环替换了自然的路由专家 ID，这会改变模型输出。 `AFDDecodeBenchConnector` 负责提供 decode-only 的 KV 状态，并为 AFD 启用了 DBO。

吞吐按部署的总 die 数归一化：

```
tokens/s/die = 聚合输出 token 吞吐 / 部署的总 die 数
```

两组负载都使用定长输入，输出在 512 到 1,536 token 之间均匀分布。

#### 16K 定长输入

![DeepSeek-V3.2 16K decode throughput per die](https://mmbiz.qpic.cn/sz_mmbiz_png/icV2Iiao3PgotXy4kdheKibZCzeN1vlIcsruWRomQ1ybaNgaBVpL0u9KpEPSziaeWIJDpljEoLA4HsKWGsvib9KYElqoMPqUZCmxq8VxSUrejoG4/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=1)

DeepSeek-V3.2 16K decode throughput per die

EP64 达到 **232.6 tokens/s/die** ，48A16F 达到 **220.3 tokens/s/die** ，64A16F 达到 **258.9 tokens/s/die** 。相对 EP64，AFD 的结果为 48A16F **\-5.3%** 、64A16F **+11.3%** 。

#### 32K 定长输入

![DeepSeek-V3.2 32K decode throughput per die](https://mmbiz.qpic.cn/mmbiz_png/icV2Iiao3Pgou2MW2v4mT5m74gOs45NVCqJcmM85ckYmcDtQvLqQgQV2VTCoWD6BSWCAcI7BhlEop4xibnuO4rQ8SEQ2s8f9njqgXwhIJrsVQU/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=2)

DeepSeek-V3.2 32K decode throughput per die

EP64 达到 **168.2 tokens/s/die** ，48A16F 达到 **151.4 tokens/s/die** ，64A16F 达到 **183.3 tokens/s/die** 。相对 EP64，AFD 的结果为 48A16F **\-10.0%** 、64A16F **+9.0%** 。

在两种输入长度下，48A16F 都低于 EP64 基线，而 64A16F 交出了最高的归一化吞吐： **16K 下 +11.3%** 、 **32K 下 +9.0%** 。这个结果说明 Attention 与 FFN 的配比很重要——单靠分离本身并不能保证吞吐提升。

受限于机器可用性，我们没有评估更高 Attention-to-FFN 配比的部署。已观察到的趋势表明，在所测试的配比下，FFN rank 仍有计算余量，而非计算受限。因此，提高 Attention rank 的比例，可能会带来进一步的吞吐提升。

### 使用 CAMAsyncAFDConnector 的异步 AFD prefill 性能

仓库中包含一个早期的 CAM async 实验：在两个昇腾 910C 节点上，使用一个裁剪到 10 层的 DeepSeek V3.2 W8A8 模型。对比采用强制专家均衡，将 `DP4PCP8 TP1` 基线与一套 AFD 布局做比较——后者由 Attention `DP3PCP8 TP1` 加 FFN `EP8` 组成。

![Median TTFT comparison for the CAM async experiment](https://mmbiz.qpic.cn/mmbiz_png/icV2Iiao3PgouvUBCYmyPLjg6JGvJl2XnP8zeUXFQaEUPPKvoFDeXV4MSwnmCyvbr2ddDkWzpzic8a7LImWMxQR5ML31dCmibt5iawFs8VVice76U/640?wx_fmt=png&from=appmsg&watermark=1#imgIndex=3)

Median TTFT comparison for the CAM async experiment

在所测的各个请求速率下，AFD 配置降低了中位 / P50 的 TTFT。在每秒 12 个请求时，中位 TTFT 从 **15.1 秒降到 8.0 秒** ，降幅约 **47%** 。在每秒 10 和 12 个请求时，实测差距都约为 7.2 秒。

**注意** ：这些数字是对 CAM async 执行路径的一次聚焦验证，而不是对完整 DeepSeek V3.2 或任意 AFD 拓扑的通用性能结论。性能收益也可能随负载而变化。

## 快速开始

当前实现需要 Python 3.10–3.13，目标 vLLM 版本为 `0.19.1` 。

### 安装

安装步骤详见我们的 README <sup>[3]</sup> 。

### 部署 recipe

部署命令取决于后端、连接器、模型和 rank 拓扑。为避免在此重复堆配置，请直接使用维护中的 AFD Plugin recipe <sup>[4]</sup> ：

- • **GPU 同步 AFD：** DeepSeek V2 Lite P2P NCCL recipe <sup>[5]</sup> 覆盖了面向 decode 的同机部署与 PD 分离部署、eager 与 CUDA graph 执行，以及多种 DP/TP 布局。
- • **NPU 异步 prefill AFD：** DeepSeek V3.2 CAM async recipe <sup>[6]</sup> 记录了所需环境、拓扑、AFD 配置、基准设置和当前限制。

最新的连接器支持矩阵、配置字段和完整启动命令，请参阅仓库 README 与 recipe 目录。

## 当前边界与路线图

该项目有意把当前边界摆在明面上：精确锁定 vLLM 版本、仅支持 model runner v1、两种角色都加载完整权重、graph 模式仅限 decode、DBO 恰好两个 ubatch，以及端到端测试受硬件门槛限制。

下一阶段的开发将聚焦于：

- • **更广的 vLLM 兼容与上游对齐：** 跟进更新的 vLLM 版本，评估 model runner v2，把兼容性补丁保持在最小，并在通用抽象成熟后贡献回上游。
- • **更灵活的执行：** 扩展 graph 模式、ubatch 数量、异步阶段，以及经过验证的 rank 拓扑。
- • **生产级验证：** 在完整模型与真实负载上，发布可复现的精度、延迟、吞吐、稳定性和多节点结果。
- • **更多模型与连接器覆盖：** 通过现有的模型封装与连接器接口接入更多 MoE 架构和后端传输，并为每个新支持的模型和连接器配套部署 recipe。
- • **多模态与 vLLM-Omni 集成：** 探索 AFD 如何与 vLLM-Omni <sup>[7]</sup> 及异构多模态流水线集成，包括在自回归（AR）、Diffusion Transformer（DiT）等可从 Attention/FFN 独立扩展中受益的阶段中的应用。
- • **异构硬件与低延迟服务：** 探索把 Attention 与 FFN 角色部署在不同类型的加速器与互联上，并配套连接器、调度、放置、计算-通信重叠等优化，以降低 TTFT 与 inter-token latency。

## 加入社区

vLLM AFD Plugin 尚处早期阶段，来自模型、服务与硬件社区的反馈将塑造它的方向。

- • **代码与文档：** afd-plugin 仓库 <sup>[1]</sup>
- • **运行时设计文档：** GPU 与昇腾的 Attention/FFN 设计文档 <sup>[8]</sup>
- • **Issue 与功能请求：** GitHub Issues <sup>[9]</sup>

让我们一起，为 MoE 推理构建一个更可组合、更懂硬件的未来。

## 致谢

感谢以下团队与贡献者对 vLLM AFD Plugin 的设计、开发、适配、测试与社区协作所提供的支持。

**Ascend and vLLM Team：** Chenzhou Jiang、Yujuan Cao、Tiangu Tang、Yeju Zhou、Yuxin Wang、Yu Chu、XiaoKun Pu、Ren Guo、Xin Ye、Qingyuan Liu、Xiaojun Yang、Shushu Chen、Anjie Hou、Ziheng Zhou、Hongsheng Liu、Roger Wang、Kaichao You

**StepFun：** Song Yuan

**Ant Group：** Shoujian Zheng、Wengang Chen、Haojiang Zheng

**FastAFD：** Junda Chen、Yichao Fu、Yuxuan Zhang

**参考链接**

1. 1\. vLLM AFD Plugin — `github.com/vllm-project/afd-plugin`
2. 2\. afd-plugin PR #67 — `github.com/vllm-project/afd-plugin/pull/67`
3. 3\. README 安装说明 — `github.com/vllm-project/afd-plugin#install`
4. 4\. AFD Plugin recipe 目录 — `github.com/vllm-project/afd-plugin/tree/main/recipe`
5. 5\. DeepSeek V2 Lite P2P NCCL recipe — `github.com/vllm-project/afd-plugin/tree/main/recipe/gpu/p2p_nccl/deepseek_v2_lite`
6. 6\. DeepSeek V3.2 CAM async recipe — `github.com/vllm-project/afd-plugin/blob/main/recipe/npu/cam_async/DeepSeek-V3.2.md`
7. 7\. vLLM-Omni — `github.com/vllm-project/vllm-omni`
8. 8\. 运行时设计文档 — `github.com/vllm-project/afd-plugin/tree/main/docs`
9. 9\. GitHub Issues — `github.com/vllm-project/afd-plugin/issues`

> vLLM 官方博客
> 
> vllm.ai/blog/2026-07-23-vllm-afd-plugin

技术 Blog · 目录

阅读原文