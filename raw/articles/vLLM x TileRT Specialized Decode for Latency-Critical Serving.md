---
title: "vLLM x TileRT: Specialized Decode for Latency-Critical Serving"
source: "https://vllm.ai/blog/2026-07-14-vllm-tilert-pd"
author:
  - "[[TileRT team]]"
published: 2026-07-14
created: 2026-07-26
description: "vLLM prefill paired with TileRT decode through vLLM V1's connector interface: a specialized, latency-optimized decode engine that coexists with native vLLM deco"
tags:
  - "clippings"
---
Disaggregated serving, which separates the compute-bound prefill phase from the memory-bandwidth-bound decode phase, has become an increasingly standard pattern for serving large language models at scale, and vLLM supports it through a first-class connector interface.  
将计算密集型的预填充阶段与内存带宽密集型的解码阶段分离的解耦式服务，已成为大规模服务大型语言模型的一种日益标准的模式，而 vLLM 通过一流的连接器接口支持这种模式。

That architectural shift carries a benefit that is easy to overlook: once prefill and decode are separated, **the decode side becomes pluggable**. Different serving regimes reward different engine designs. The prefill pool, the scheduler, the caching layer, and the serving API stay exactly where they are, while the decode pool becomes a deliberate choice.  
这种架构转变带来了一个容易被忽视的好处：一旦预填充和解码分离， **解码部分就变成了可插拔的** 。不同的服务机制会奖励不同的引擎设计。预填充池、调度器、缓存层和服务 API 保持不变，而解码池则成为一个需要精心选择的设计。

Today, we are introducing exactly such a choice: **vLLM prefill paired with TileRT decode**, integrated through vLLM V1's public connector interface and shipping with TileRT 0.1.5. For latency-critical workloads, this pairing delivers **TileRT's native per-user decode speed**, while everything else about the deployment remains stock vLLM.  
今天，我们推出了这样一种选择： **vLLM 预填充与 TileRT 解码相结合** ，通过 vLLM V1 的公共连接器接口集成，并随 TileRT 0.1.5 一起发布。对于延迟敏感型工作负载，这种组合可提供 **TileRT 的原生单用户解码速度** ，而部署的其他所有方面都与标准的 vLLM 保持一致。

## Why a second decode option?为什么需要第二个解码选项？

vLLM's native decode is, and remains, the right default: it is built for high-throughput batched serving across a huge range of models and hardware. But there is a growing class of workloads, e.g., agentic loops, interactive coding assistants, real-time voice, where the metric that matters is not aggregate throughput but how fast tokens reach each individual user. These workloads are latency-bound, and they call for a decode engine designed from the ground up for exactly that regime. Native decode and TileRT target different points on the same throughput–latency frontier, which is why they compose.  
vLLM 的原生解码一直是且仍然是最佳默认选择：它专为跨多种模型和硬件的高吞吐量批量服务而设计。但越来越多的工作负载，例如智能体循环、交互式编码助手和实时语音，其关键指标并非总吞吐量，而是令牌到达每个用户的速度。这些工作负载受延迟限制，因此需要一个专为这种环境而设计的解码引擎。原生解码和 TileRT 针对的是同一吞吐量-延迟边界上的不同点，这也是它们可以组合使用的原因。

TileRT is such an engine: a new inference runtime built around the single goal of pushing per-user decode speed toward the limits of the hardware. We have written elsewhere about why we believe [speed is becoming its own scaling dimension](https://www.tilert.ai/blog/speed-as-the-next-scaling-law.html).  
TileRT 就是这样一款引擎：它是一个全新的推理运行时环境，其唯一目标是将每个用户的解码速度推向硬件极限。我们曾在其他文章中阐述过，为什么我们认为 [速度本身正在成为一个独立的扩展维度](https://www.tilert.ai/blog/speed-as-the-next-scaling-law.html) 。

This post is not about the engine, though. It is about a more practical question: can you adopt a specialized decode engine **without giving up the ecosystem you depend on**, e.g., OpenAI-compatible APIs, scheduling, prefix caching, tool calling, and the operational maturity of vLLM?  
不过，这篇文章并非关于引擎本身，而是关于一个更实际的问题：能否 **在不放弃所依赖的生态系统** （例如，与 OpenAI 兼容的 API、调度、前缀缓存、工具调用以及 vLLM 的运行成熟度）的情况下，采用专门的解码引擎？

This integration is designed to make that trade-off as small as possible:  
这种整合旨在尽可能减少这种权衡取舍：

- **Prefill is vLLM.** Scheduling, chunked prefill, prefix caching — untouched.  
	**预填充功能是 vLLM。** 调度、分块预填充、前缀缓存——保持不变。
- **The serving surface is vLLM.** Same APIs, same request format, same tooling.  
	**服务界面采用 vLLM。API** 、请求格式和工具都相同。
- **Only decode changes, and only for the traffic you send there.** The TileRT-paired stack runs alongside your existing vLLM deployment; each workload picks its endpoint.  
	**仅解码更改，且仅针对您发送到该位置的流量。TileRT** 配对堆栈与您现有的 vLLM 部署并行运行；每个工作负载都会选择其端点。

## Architecture: coexistence by design建筑：设计中的共存

The core design principle is **zero changes to vLLM**: no fork, no patches, no wrapped internal workers. The integration lives entirely behind vLLM V1's public extensibility surface: a `KVConnectorBase_V1` implementation, composed under `MultiConnector` and loaded through the standard `kv_connector_module_path` mechanism. This matters beyond engineering aesthetics: adding a TileRT decode pool cannot destabilize a vLLM deployment you already run, and upgrading vLLM does not mean re-porting a fork.  
核心设计原则是 **vLLM 零改动** ：不创建分支，不打补丁，也不封装内部工作进程。该集成完全基于 vLLM V1 的公共扩展接口：一个 `KVConnectorBase_V1` 实现，在 `MultiConnector` 下构建，并通过标准的 `kv_connector_module_path` 机制加载。这不仅仅关乎工程美观：添加 TileRT 解码池不会破坏您已运行的 vLLM 部署，升级 vLLM 也无需重新提交分支。

![Coexistence by design: latency-critical traffic is marked by the TileRT PD router and claimed by the TileRT connector, while general traffic flows through the native disaggregation path, both served by a single stock vLLM prefill pool composed under MultiConnector.](https://vllm.ai/blog-assets/figures/2026-07-14-vllm-tilert-pd/pd_arch.png)

Coexistence by design: latency-critical traffic is marked by the TileRT PD router and claimed by the TileRT connector, while general traffic flows through the native disaggregation path, both served by a single stock vLLM prefill pool composed under MultiConnector. 设计上的共存：对延迟要求严格的流量由 TileRT PD 路由器标记，并由 TileRT 连接器认领，而一般流量则通过原生解耦路径流动，两者都由 MultiConnector 下组成的单个标准 vLLM 预填充池提供服务。

**Routing.** A lightweight router fronts the TileRT pool. For each request it sets `max_tokens=1` (vLLM performs the prefill and emits the first token) and attaches the target decode node in the standard pass-through field: `kv_transfer_params = {"tilert_host": ..., "tilert_ctrl_port": ...}`. Traffic for the native pool flows through the usual disaggregation proxy, unmodified.

**Claim filtering.** The TileRT connector claims only requests carrying the mark and is a strict no-op for everything else, so the two decode pools can share one prefill instance (even a single forward batch): adopting TileRT for some traffic changes nothing for the rest.

**A pure producer.** The connector acts as a `kv_producer` only, it never touches scheduling or sampling; it extracts and ships state after prefill. In every other respect the prefill instance is a stock vLLM server.

## How the handoff works

For cross-engine disaggregation to be practical, three things have to be true: the transfer must be fast, it must not slow the prefill node down, and the decode engine must pick up exactly where prefill left off.

**Data plane.** After prefill, the request's attention state (compressed KV, the sparse-attention index caches, and a small amount of metadata) moves to the decode node as RDMA one-sided writes into pre-registered GPU buffers, with either Mooncake or NIXL as the transfer engine. No intermediate serialization, no staging through host memory. The handoff protocol itself is independent of the underlying transfer engine, whose job is only to move bytes.

**Fully overlapped with prefill.** State extraction happens inside the forward window: the request's state is copied to a staging buffer before its cache blocks can be recycled, and a background sender performs the actual network transfer. A request bound for TileRT never blocks the next prefill iteration, including for native-pool requests sharing the same batch.

**Injection into a live engine.** On arrival, the state is converted to TileRT's native layout and injected directly into a running engine; decoding begins immediately, with multi-token speculative decoding active from the first step.

## Evaluation

![GLM-5.1-FP8 token generation speed on 8× NVIDIA B200 with TileRT v0.1.5. Output length 1K, input length 1K–192K. Bars compare TileRT without MTP, with MTP at average acceptance length 3.2, and the peak under best-case MTP acceptance 4.0.](https://vllm.ai/blog-assets/figures/2026-07-14-vllm-tilert-pd/glm5_tilert_mtp.png)

GLM-5.1-FP8 token generation speed on 8× NVIDIA B200 with TileRT v0.1.5. Output length 1K, input length 1K–192K. Bars compare TileRT without MTP, with MTP at average acceptance length 3.2, and the peak under best-case MTP acceptance 4.0.

## Choosing your decode pool

Route to **TileRT decode** when per-user token speed is the binding constraint, e.g., interactive agents, real-time assistants, latency-SLO inference, and the model is one TileRT supports.

Stay on **native vLLM decode** for maximum aggregate throughput, high-concurrency batching, and the long tail of models and features that general-purpose decode covers.

Both stacks expose the same OpenAI-compatible surface, so moving a workload between them is a routing change, not a client change.

**Current limitations.** In this release a TileRT decode node serves one in-flight request at a time, with the router providing gated dispatch and back-pressure. Model coverage in this release is GLM-5/5.1 and DeepSeek-V3.2, with more to come.

## Getting started

TileRT 0.1.5 is available on [PyPI](https://pypi.org/project/tilert/) (`pip install tilert`; Python 3.12, CUDA 13 wheels) and the [TileRT repository](https://github.com/tile-ai/TileRT). Install it on both the prefill and decode nodes; the prefill side needs it for the connector plugin.

```bash
# 0. One-time: convert the HF checkpoint to TileRT's weight format
python -m tilert.models.preprocess.weight_converter \
    --model_type glm-5 \
    --model_dir /path/to/GLM-5.1 \
    --save_dir /path/to/tilert-glm5.1-weights
 
# 1. TileRT decode node
python -m tilert.pd_vllm.decode_server \
    --engine tilert --model glm5 \
    --model-weights-dir /path/to/tilert-glm5.1-weights \
    --with-mtp --max-seq-len 202752 \
    --kv-cache-dtype fp8 \
    --ctrl-port 5556 --http-port 5557
 
# 2. vLLM prefill (stock vLLM; the connector loads as a plugin).
#    The MTP speculative config is required: prefill populates the
#    draft-layer KV that decode-side speculation resumes from.
vllm serve /path/to/GLM-5.1 \
    --served-model-name glm5.1 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --enforce-eager \
    --trust-remote-code \
    --return-tokens-as-token-ids \
    --gpu-memory-utilization 0.8 \
    --kv-cache-dtype fp8_ds_mla \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' \
    --kv-transfer-config '{
        "kv_connector": "TileRTConnector",
        "kv_connector_module_path": "tilert.pd_vllm.prefill_connector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config":{
            "tilert_host":"[TILERT_DECODE_SERVER_IP]",
            "tilert_ctrl_port":5556,
            "tilert_model":"glm5",
            "tilert_max_seq_len":202752
        }
    }'
 
# 3. Router: OpenAI-compatible ingress for the TileRT pool
python -m tilert.pd_vllm.pd_router \
    --vllm-url http://prefill-node:8000 \
    --decode decode-node:5556:5557 \
    --model-path /path/to/GLM-5.1 \
    --port 23333
```

To run the TileRT pool and a native vLLM decode pool behind one shared prefill instance, compose both connectors under `MultiConnector`. The configuration we validated runs NIXL end to end (vLLM's standard `NixlConnector` for the native pool, the TileRT connector in NIXL mode for the TileRT pool), so the shared prefill uses a single transfer library; only the prefill's `--kv-transfer-config` changes.

## Looking ahead

We think disaggregation is quietly changing what an inference stack is: less a single engine, and more a composition of specialized engines behind a shared serving layer. vLLM's connector interface is what makes that composition possible today, and this integration is one concrete example. It is also why an engine like TileRT can afford to specialize this deeply: with the serving layer shared and the interfaces open, going deep on one dimension no longer means rebuilding everything else.

We would love feedback from the community: on the integration surface, on the workloads where this helps, and on which models to support next.

## Acknowledgements

We thank the vLLM community for designing the V1 connector interface that made a zero-modification integration possible, and the Mooncake and NIXL projects for the RDMA transfer engines. We also appreciate [Inferact Inc.](https://inferact.ai/) for the collaboration to improve vLLM-TileRT integration.