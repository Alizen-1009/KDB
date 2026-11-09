---
title: "SGLang 低延迟模式：Kernel 与运行时"
source: "https://mp.weixin.qq.com/s/8pV1fFyfbwd2dxdLAgVndg"
author:
  - "[[SGLang]]"
published:
created: 2026-11-09
description: "SGLang 推出低延迟模式：用插件框架接入 NVIDIA 与 AMD 的专用 kernel。AgentX 回放中，GB300 单会话 P90 单用户速度提升 1.38×，MI355X 在 4 会话下提升 1.56×。"
tags:
  - "clippings"
---
SGLang GiantPandaLLM *2026年10月20日 10:37*

> 原文：SGLang Low-Latency Mode: Kernels and Runtime（https://www.lmsys.org/blog/2026-10-18-sglang-low-latency-mode）
> 
> 作者：SGLang、NVIDIA 与 AMD 团队 · 2026 年 10 月 18 日 · 本文为中文翻译

低延迟推理有大量可以针对具体模型、硬件和负载做定制的优化空间。这类优化往往需要大量 kernel 融合，不同实现之间还会出现重复代码，很难长期维护。为了支持这种定制化，我们推出了 **SGLang Low-Latency Mode（低延迟模式）** ：它通过一个插件框架把专用 kernel 接入 SGLang，在 NVIDIA 和 AMD GPU 上都带来了显著的延迟下降。在 4 张 NVIDIA GB300 上做 AgentX trace 回放，并发 1 个会话时，P90 单用户 token 生成速度从 **671.2 提升到 928.4 tokens/s（1.38×）** ；在 4 张 AMD MI355X 上，并发 4 个会话时，从 **331.6 提升到 516.0 tokens/s（1.56×）** 。每组结果都是在同一硬件上，把低延迟模式和 SGLang 基线做对比。

![AgentX 在 NVIDIA GB300 与 AMD MI355X 上 1、4、8 个并发会话下的性能](https://mmbiz.qpic.cn/mmbiz_jpg/gvzXpmzO2KZuRo1jrsgwK8rvYHuPpdP0wQp0LkFdgeO3z9cjqVnTibALeEzgoHPXJeQO2dSduiblSNP7GkHr8rwY4lY6zCHaxgGN5Ghs2ZaKg/640?wx_fmt=webp&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=0)

AgentX 在 NVIDIA GB300 与 AMD MI355X 上 1、4、8 个并发会话下的性能

*AgentX · DeepSeek-V4.1-Flash + DSpark · NVIDIA GB300 与 AMD MI355X。*

**目录**

1. 面向低延迟推理的 kernel
- NVIDIA 低延迟 kernel
	- AMD 低延迟 kernel
3. 插件系统：设计与集成
4. 未来计划

## 面向低延迟推理的 kernel

### NVIDIA 低延迟 kernel

在 NVIDIA GPU 上，我们的做法是让相互依赖的 kernel 与通信重叠执行，同时针对小 batch 专门定制 attention、MoE 和草稿采样。这些 kernel 放在 sgl-nvidia-mega（https://github.com/sgl-project/sgl-nvidia-mega） 里，它在 NVIDIA GB300 上同时加速 DeepSeek-V4.1-Flash + DSpark 的目标模型验证和草稿执行。

#### 时间都花在哪里

只有 1 个请求时，DSpark 每一步要验证 6 行 token，每个 kernel 分到的工作很少，而依赖关系和跨 GPU 通信又会叠加延迟。在预热后的 CUDA graph profile 中，插件只把验证 graph 的 kernel 数从 1,205 个减少到 1,189 个，但把 decoder 层延迟的中位数从 96.8 µs 降到了 68.4 µs。收益主要来自减少等待和让工作重叠，其中 attention 区域下降最多。下表先拆解这些变化，后面再逐一讲主要的优化。

| 单个 decoder 层的区域（1 个请求，中位数） | SGLang | 使用插件 |
| --- | --- | --- |
| Attention：从上一层 MoE 输出到输出投影 | 33.4 µs | 18.3 µs |
| Attention 输出投影与张量并行规约 | 10.9 µs | 9.4 µs |
| MoE：从路由到规约后的输出 | 44.4 µs | 36.5 µs |
| 整层：从一次路由到下一次路由 | 96.8 µs | 68.4 µs |

*每组在 4 张 GPU 上统计 780–800 个实例的中位数。各区域的中位数相加不等于整层的中位数。Profiling 本身会影响计时，这些数据用于解释机制，而不是汇报 serving 性能。*

#### 针对低延迟的优化

**1\. 输入还没准备好，就先启动下一个 kernel。** Programmatic dependent launch（PDL）允许一个 kernel 在前一个 kernel 还在运行时就开始执行：它可以先准备 barrier、设置 tensor memory、加载权重，再去等待自己需要的输入。SGLang 本来就在用 PDL，插件做的是把依赖的释放点提前，让更多准备工作和正在进行的计算重叠。比如，MoE 规约一开始轮询其他 GPU 的结果，就立刻释放下一层的第一个投影。

Attention 路径还使用了专为小 batch 定制的投影 GEMM，并在加载 query 时顺便完成 RoPE，省掉一个单独的 kernel。在依赖允许的地方，attention 会在等待 query 之前就开始收集 KV 数据。在只有 1 个请求时，插件还会在每一步开始时，把相互独立的 Engram 查表和投影放到一条侧流上执行；它们的 all-reduce 仍留在主流上，以保证各 GPU 之间的通信顺序一致。

**2\. 在 GEMM 的 epilogue 里直接发出投影结果。** Attention 输出投影之后，每张 GPU 只持有一部分结果，必须跨 GPU 规约。SGLang 的标准路径用两个 kernel：GEMM 先写出结果，再由一个融合 kernel 把它读回来、发给其他 GPU，并完成规约、mHC 更新和归一化。

插件仍然保持两个 kernel，但把“发送”挪进了 GEMM 的 epilogue：每个 CTA 算完自己的 tile，就直接从寄存器通过 NVLink multicast store 写到其他 GPU，此时别的 tile 还在计算。消费端通过 PDL 提前启动，等到这些结果发布后再做规约。这样通信开始得更早，也不需要为了发送而把部分结果再读一遍；消费端本身的计算不变。

![SGLang 与 NVIDIA 插件中 DeepSeek-V4.1-Flash 在 GB300 上的 attention 输出交接](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gvzXpmzO2KZkSEDj5yPRxMKDmDGUricblAztGOaicjJibIHMYAJPcSmhuZFlkFicPvSq0eAXTtRIFaHkiapJoaqmeTIzDF3NFlOzV5KB7X9KDYsY/640?wx_fmt=webp&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=1)

SGLang 与 NVIDIA 插件中 DeepSeek-V4.1-Flash 在 GB300 上的 attention 输出交接

*插件把跨 GPU 发送挪进了 GEMM 的 epilogue。两条路径都是两个 kernel；方框大小不代表耗时。*

**3\. 路由还没结束，就开始专家 GEMM。** 并发很低时，路由在 MoE 路径里会占相当大的比重。最多只有 8 行时，每个专家被选中的 token 都能放进一个 8 行的 tile。插件用 bitmap 和 population count 来摆放这些 tile，不需要扫描或排序，同时保持 SGLang 原本的 top-k 选择不变。它会提前启动第一个专家 GEMM，让准备工作和路由重叠，并跳过 padding 权重里全零的 tile。标准 kernel 本来也用 8 行的 tile，收益来自更简单的路由、更早的执行和更少的无效计算。在 profile 中，MoE 窗口的中位数从 44.4 µs 降到 36.5 µs，第二个 GEMM 和规约几乎没有变化。

![1 个请求时，使用与不使用 NVIDIA 插件的 MoE 实测时间线](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gvzXpmzO2KZW85kHzYb7s4flhVWKAgUj5EwGsq7XGTrFEa5hk4l0JxVQGialYaTPc8sdljib56rdJM16xH6OEZdlY1MmicFichsPhiajVGLrOiaOo/640?wx_fmt=webp&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=2)

1 个请求时，使用与不使用 NVIDIA 插件的 MoE 实测时间线

*同一张 GPU、同一层、同一次回放中配对的一组 trace。GEMM1 的条形包含它等待路由的时间，所以更早启动可能让这一条变长，但整体 MoE 窗口变短。*

**4\. 把顺序的草稿采样留在一个 kernel 里。** DSpark 的 Markov head 是逐个提出候选 token 的。插件把全部 5 次提议合并成每张 GPU 一次 kernel 启动，把 head 的权重留在寄存器里，同时跨 GPU 协调 token 的选择。这样，profile 中的提议阶段从 85.1 µs 降到 34.2 µs，并且不改变 SGLang 的投机解码 worker 和接受逻辑。

#### Serving 结果

在 8K 输入 / 1K 输出的负载下，NVIDIA 插件在 4 张 GB300 上、1–8 个并发请求时，把单用户 decode 速度提升了 12–38%。两组运行的配置完全相同，模拟的平均接受长度都是 3.51，只切换了插件开关。

![DeepSeek-V4.1-Flash 在 4 张 GB300 上，使用与不使用 NVIDIA 插件时的单用户 decode 速度与每 GPU 总输出吞吐](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gvzXpmzO2Kb9rcByQ2ibfkBOI4FxSVAiamQgp04Fqql5L01E8tCve2PFSE94yichX59ibH90hBOOO2UIwU6WFtAgiaRy3XiautDUbHmJaJcRPWRics/640?wx_fmt=webp&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=3)

DeepSeek-V4.1-Flash 在 4 张 GB300 上，使用与不使用 NVIDIA 插件时的单用户 decode 速度与每 GPU 总输出吞吐

*DeepSeek-V4.1-Flash + DSpark · 4 × GB300 · 8K 输入 / 1K 输出。*

### AMD 低延迟 kernel

在 AMD GPU 上，我们的做法是把一个 decoder 层的大部分计算和通信合并进少数几个 persistent kernel。这些 kernel 放在 sgl-amd-mono（https://github.com/sgl-project/sgl-amd-mono） 里，它在 AMD Instinct MI355X 和 MI350X 上同时加速 DeepSeek-V4.1-Flash + DSpark 的目标模型验证和草稿执行。

并发很低时，每个 decoder 层要跑几十个小 kernel 和两次张量并行 all-reduce，主要有三类开销：

- **启动开销：** HIP graph 省掉了 CPU 端的启动工作，但每个小 kernel 在 GPU 上仍要花 2–4 µs。
- **边界空闲：** 每个 kernel 必须执行完下一个才能开始，于是一个 kernel 收尾、下一个爬坡时，计算单元在空转，权重加载也会停顿。
- **通信停顿：** 每层要等两次跨 GPU 的 all-reduce，期间没有任何计算在跑。
![batch size 为 1 时一个 DeepSeek-V4.1 层的示意时间线](https://mmbiz.qpic.cn/mmbiz_jpg/gvzXpmzO2KYemCbYFrDoMopiaVjAPo0UgPQl0PVbThUz59PDTmmXLXSzibrYP81uWOHib2Vkbic9oVqRaxrEU8zvEbicPkh03ZdO7BHr8VS7B9mY/640?wx_fmt=webp&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=4)

batch size 为 1 时一个 DeepSeek-V4.1 层的示意时间线

*标准路径要跑几十个小 kernel，每个启动边界都有空闲，两次 all-reduce 会打断计算。使用 AMD 插件后，这一层只需要两次 FlyDSL 启动（K1 和 K2）：不依赖路由的权重提前加载，部分和在计算继续进行的同时发往其他 GPU。方框大小不代表耗时。*

#### 每层两次 persistent 启动

Persistent FlyDSL kernel 把每层的大部分工作合并成两次启动。 **K1** 负责 mHC 更新、query 和 KV 投影，以及写 KV cache； **K2** 负责 attention、输出投影、剩下的 mHC 更新、MoE 以及两次 all-reduce。索引层会在两者之间加入压缩、 **K2i** 打分和 top-k。启动边界保证 K1 写入的 KV 在 K2 读取之前对整张 GPU 可见。

#### 一次启动内部的阶段重叠

每个阶段产出的值都带有当前 decode 步的标签一起存储，下一个阶段（无论在同一张 GPU 还是另一张 GPU 上）只要看到标签匹配，就可以开始处理这个值，不需要单独的 flag，也不需要在两步之间重置。不依赖路由的权重会在输入到达之前就加载：K1 的投影权重在 norm 完成之前加载；K2 里 mHC 混合权重和共享专家的权重，会在它们要相乘的数据到来之前加载。路由专家的权重则等路由器选定专家后再加载。

#### 把 all-reduce 搬进 kernel

每算完一个输出投影的 tile，K2 就把它推送到每张 GPU 的显存里；每张 GPU 按 rank 顺序对这些片段求和，而此时其他 workgroup 还在继续生产 tile（push 与 reduce 阶段：https://github.com/sgl-project/sgl-amd-mono/blob/e952b221aafc05ef5adb84d2caf942f7283c1f0b/sgl\_amd\_mono/deepseek\_v41/gfx950/stages/seam.py）。MoE 的输出也用同样的方式规约。每一次跨 GPU 等待都设置了低于 SGLang watchdog 超时的截止时间；一旦超时，所有 GPU 上的这次启动都会停止，而不是让服务卡死。

AMD 的实现兼容 decode graph，直接接入 SGLang 的 graph runner：kernel 的编译和缓冲区的准备在初始化阶段完成，SGLang 随后为每个支持的 shape 捕获专用执行路径，并通过现有的 serving 流程回放。

#### Serving 结果

在 AgentX trace 回放中，AMD 插件在 4 张 MI355X 上、1、4、8 个并发会话时，把 P90 单用户 token 生成速度提升了 45–56%，结果见文章开头的总览图。两组运行使用相同的 AgentX 配置，都开启 DSpark 投机解码和 TP4。

## 插件系统：设计与集成

专用 kernel 可能需要大量代码：每一种硬件、模型架构和并行配置的组合都可能需要单独实现，不同负载还会带来进一步的定制。组合越多，需要评审、测试和维护的代码就越多。如果把每一种实现都合进 SGLang，核心代码库会越来越难维护，评审者和维护者的负担也会不断加重。

![外部 kernel 库向 SGLang 注册模型规格，由 SGLang 校验配置、构建插件并分发支持的步骤](https://mmbiz.qpic.cn/sz_mmbiz_jpg/gvzXpmzO2KYBL10cjPLELKD5XfzxS68D3jx57ASgPsIicfMVmxWCkrjbh0hoIoTYGicUbACjS2N2vT0EA4gELqMoq9IBvGeCK5Nt44mK4IW5I/640?wx_fmt=webp&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1#imgIndex=5)

外部 kernel 库向 SGLang 注册模型规格，由 SGLang 校验配置、构建插件并分发支持的步骤

*外部 kernel 库通过注册、校验、构建和运行时分发接入 SGLang。图中展示的是共同的设计；AMD 和 NVIDIA 的接口仍在演进中。*

插件系统通过“模型规格”把这些外部库接入 SGLang：规格中声明实现的依赖和支持的配置。kernel 团队可以独立开发自己的实现，同时复用 SGLang 的调度和运行时资源。

集成过程把启动时的准备和运行时的执行分开：

- **守卫（Guard）：** 专用 kernel 会对硬件、精度、并行方式和 serving 功能做出假设。SGLang 在加载权重之前检查声明的依赖，让不兼容的配置尽早失败；在构建插件之前，还会对加载好的模型做额外检查。这些检查能区分两种情况：服务配置本身不受支持，还是受支持的服务遇到了插件覆盖范围之外的某一步。
- **构建（Build）：** 权重准备和执行资源在初始化阶段完成，供后续请求复用。集成会照顾到不同实现的需求：AMD 可以在权重转换之前保留 checkpoint 张量，NVIDIA 则拿到一个 binding，在运行时状态可用时通过它访问。SGLang 会在确定 KV cache 大小之前构建插件，把插件的显存分配计算在内，然后在模型的前向路径中装上分发逻辑。
- **运行时（Runtime）：** 专用执行和通用执行可以共存，因为一个插件可能只覆盖某些步骤类型和 token 数。装好的分发器根据插件的支持条件，选择插件的前向还是 SGLang 原本的前向。各张量并行 rank 必须对覆盖范围达成一致，避免进入不兼容的集合通信。这个选择嵌入在 ModelRunner 现有的执行流程中：graph 捕获时，为该 shape 选定的路径会被录下来，serving 时直接回放 graph，不需要重复做 Python 层的分发。

这种设计让 kernel 团队可以交付聚焦的优化，而不必把每一种专用实现都加进 SGLang 的核心代码库。它把集成工作集中在明确的接口上，并允许每个库逐步扩大覆盖范围：专用 kernel 负责它们优化的场景，而在受支持的配置内，剩下的步骤由 SGLang 的通用前向处理。

## 未来计划

### 支持更多模型

GLM-5.3、K3、MiniMax M3 等模型在 NVIDIA 和 AMD GPU 上的 Megakernel / Monokernel 开发正在进行中，我们计划在这些实现就绪后接入插件框架。这些低延迟 kernel 展现出了进一步提升性能的潜力。

### CI 与性能验证

外部 kernel 实现不在 SGLang 常规的代码评审流程之内，因此在插件接口处建立完善的 CI 覆盖至关重要，用来验证这些实现满足 SGLang 的正确性和运行时要求。我们计划测试兼容性检查、权重绑定、前向输出和状态更新（包括 KV cache 写入）。随着 SGLang 和外部库的演进，端到端测试还会覆盖 graph 执行、回退行为和性能回归。

### 持续演进共享的插件接口

插件框架虽然已经就位，但仍处于早期阶段。随着接入的 kernel 越来越多，并从生产使用中积累经验，我们会在长期迭代中持续改进它的接口和设计。

sglang · 目录

阅读原文