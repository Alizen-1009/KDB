---
title: "Model Runner V2: A Modular and Faster Core for vLLM"
source: "https://vllm.ai/blog/mrv2"
author:
  - "[[vLLM Team]]"
published: 2026-03-24
created: 2026-04-23
description: "We are excited to announce Model Runner V2 (MRV2), a ground-up re-implementation of the vLLM model runner. MRV2 delivers a cleaner, more modular, and more effic"
tags:
  - "clippings"
---
We are excited to announce **Model Runner V2 (MRV2)**, a ground-up re-implementation of the vLLM model runner. MRV2 delivers a cleaner, more modular, and more efficient execution core—with **no API changes**. The goal is simple: better code and better performance.  
我们很高兴地宣布推出 **模型运行器 V2 (MRV2)** ，它是对 vLLM 模型运行器的彻底重新实现。MRV2 提供了一个更简洁、更模块化、更高效的执行核心，且 **无需更改任何 API** 。目标很简单：更好的代码和更好的性能。

Like the vLLM V1 release last year, this is an architectural upgrade driven by hard-earned lessons from vLLM's large user base and feedback from the community. We revisited persistent batching, async scheduling, input preparation, and sampling, then rebuilt the model runner around three core principles:  
与去年发布的 vLLM V1 版本一样，本次架构升级基于 vLLM 庞大用户群的宝贵经验和社区反馈。我们重新审视了持久批处理、异步调度、输入准备和采样，并围绕三个核心原则重建了模型运行器：

- **Be modular.** Isolate model-specific logic from the common execution path.  
	**采用模块化设计。将** 模型特定的逻辑与通用执行路径隔离。
- **Be GPU-native.** Move bookkeeping off the CPU and onto the GPU.  
	**充分利用 GPU 性能。** 将簿记工作从 CPU 转移到 GPU 上。
- **Be async-first.** Treat overlapped CPU/GPU execution as a design constraint, not a retrofit.  
	**优先考虑异步编程。** 将 CPU/GPU 执行重叠视为设计约束，而不是后期改造。

MRV2 is not yet feature-complete, but you can try it today:  
MRV2 的功能尚未完全完善，但您可以立即试用：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

We plan to make MRV2 the default in the near future.  
我们计划在不久的将来将 MRV2 设为默认设置。

## Why Model Runner V2?为什么选择 Model Runner V2？

Since vLLM V1 shipped last year, the model runner has accumulated significant technical debt as features and optimizations were added incrementally. Many of those changes were useful in isolation, but the implementation grew harder to reason about over time—especially once async scheduling and speculative decoding became central to the execution model.  
自去年 vLLM V1 发布以来，随着功能的逐步添加和优化，模型运行器积累了大量的技术债务。许多改动单独来看都很有用，但随着时间的推移，其实现变得越来越难以理解——尤其是在异步调度和推测性解码成为执行模型的核心之后。

In practice, this led to several recurring issues:  
实际上，这导致了几个反复出现的问题：

- **Tangled persistent batch state.** Persistent state was tightly coupled to per-step model inputs, making request insertions, removals, and reordering more complex than necessary.  
	**纠缠不清的持久化批处理状态。** 持久化状态与每一步模型输入紧密耦合，使得请求的插入、删除和重新排序比必要的要复杂得多。
- **Fragile async execution.** Async scheduling was retrofitted onto the V1 runner, so many features required unnatural and unreasonably complex logic to coexist with it.  
	**异步执行脆弱。** 异步调度是后期添加到 V1 运行器中的，因此许多功能需要不自然且过于复杂的逻辑才能与之兼容。
- **CPU-bound bookkeeping.** Input preparation and sampling relied on many small CPU-side operations, leaving performance on the table as GPUs kept getting faster.  
	**以 CPU 为中心的记账方式。** 输入准备和采样依赖于许多小型 CPU 端操作，随着 GPU 速度的不断提升，性能却未能得到充分发挥。
- **Difficult extensibility.** The runner as a whole became harder to understand and extend cleanly as new models and features arrived.  
	**扩展性差。** 随着新模型和功能的出现，整个运行程序变得越来越难以理解和清晰地扩展。

MRV2 addresses these issues by rethinking the model runner with cleaner state ownership and more explicit abstractions.  
MRV2 通过重新思考模型运行器来解决这些问题，它具有更清晰的状态所有权和更明确的抽象。

## What's New in Model Runner V2?Model Runner V2 有哪些新特性？

### 1\. A Better Persistent Batch Design and GPU-Native Input Preparation1. 更优的持久批处理设计和 GPU 原生输入准备

vLLM performs substantial bookkeeping for batching, paged attention, sampling parameters, and more. Historically, much of this work was implemented as many small CPU-side operations.  
vLLM 负责大量的簿记工作，包括批处理、分页注意力机制、采样参数等等。过去，这些工作大多是通过许多小型 CPU 端操作来实现的。

To reduce that overhead, vLLM V1 introduced persistent batching: because consecutive batches are usually similar, it is much cheaper to update cached state incrementally than to rebuild large tensors from scratch every step. However, the V1 design used persistent state directly as model and sampler inputs, which created awkward layout constraints and complicated bookkeeping.  
为了降低这种开销，vLLM V1 引入了持久批处理：由于连续批次的数据通常相似，因此增量更新缓存状态比每一步都从头重建大型张量要便宜得多。然而，V1 设计直接将持久状态用作模型和采样器的输入，这造成了布局上的不便和复杂的簿记工作。

![Figure 1: Persistent batch in V1. Request ordering is tightly coupled to the block table layout, requiring complex reordering when requests are added or removed.](https://vllm.ai/blog-assets/figures/2026-03-24-mrv2/persistent_batch_v1.png)

Figure 1: Persistent batch in V1. Request ordering is tightly coupled to the block table layout, requiring complex reordering when requests are added or removed. 图 1：V1 中的持久批处理。请求排序与块表布局紧密耦合，因此在添加或删除请求时需要复杂的重新排序。

MRV2 **decouples persistent request state from per-step input tensors**. Each live request gets a stable row in a fixed-size state table for its active lifetime. At each step, the runner gathers the step-specific inputs from that persistent state according to the current request ordering. This preserves the performance benefit of incremental updates while removing a large amount of state-management complexity. It also eliminates redundant backup state such as `CachedRequestState`, since active requests no longer depend on fragile tensor-wide reordering.  
MRV2 **将持久化请求状态与每步的输入张量解耦** 。每个活动请求在其生命周期内都会在固定大小的状态表中拥有一个稳定的行。在每一步，运行器会根据当前请求的顺序从该持久化状态中收集特定于该步骤的输入。这既保留了增量更新的性能优势，又大大降低了状态管理的复杂性。此外，由于活动请求不再依赖于脆弱的张量级重排序，因此也消除了冗余的备份状态（例如 `CachedRequestState` 。

![Figure 2: Persistent batch in MRV2. A stable state table is maintained independently of the per-step input layout. A gather operation produces the correctly ordered input block table each step.](https://vllm.ai/blog-assets/figures/2026-03-24-mrv2/persistent_batch_mrv2.png)

Figure 2: Persistent batch in MRV2. A stable state table is maintained independently of the per-step input layout. A gather operation produces the correctly ordered input block table each step. 图 2：MRV2 中的持久批处理。稳定的状态表独立于每一步的输入布局而维护。每次操作都会生成正确排序的输入块表。

MRV2 also **moves input preparation to the GPU** using Triton kernels. Request state is largely kept on the device, and tensors such as `input_ids`, `positions`, `query_start_loc`, and `seq_lens` are now built directly on the GPU. This provides three concrete benefits:  
MRV2 还使用 Triton 内核 **将输入准备工作转移到 GPU 上** 。请求状态主要保存在设备上，诸如 `input_ids` 、 `positions` 、 `query_start_loc` 和 `seq_lens` 等张量现在直接在 GPU 上构建。这带来了三个实际好处：

- **Lower CPU overhead** by avoiding a large amount of Python and CPU tensor manipulation.  
	避免大量的 Python 和 CPU 张量操作，从而 **降低 CPU 开销** 。
- **Lower code complexity** by removing the constraints imposed by CPU-side tensor operations.  
	通过消除 CPU 端张量运算的限制来 **降低代码复杂度** 。
- **Better async and speculative decoding compatibility**, since GPU-resident preparation can directly consume device-side results without synchronization (see next section).  
	**更好的异步和推测性解码兼容性** ，因为 GPU 驻留准备可以直接使用设备端结果而无需同步（见下一节）。

### 2\. Async-First Design 2. 异步优先设计

Async scheduling is now fundamental to vLLM. The scheduler and worker prepare step `N+1` while the GPU executes step `N`, overlapping host work and device work to maximize utilization. While this was already supported in vLLM V1, it was largely a retrofit rather than a first-class design constraint.  
异步调度现在是 vLLM 的核心功能。调度器和工作进程在 GPU 执行步骤 `N` 同时准备步骤 `N+1` ，通过重叠主机工作和设备工作来最大化资源利用率。虽然 vLLM V1 已经支持此功能，但它主要是一种后期改进，而非一项重要的设计约束。

![Figure 3: Async scheduling in V1. The CPU schedules and prepares the next step while the GPU executes the current step, overlapping CPU and GPU work.](https://vllm.ai/blog-assets/figures/2026-03-24-mrv2/async_scheduling.png)

Figure 3: Async scheduling in V1. The CPU schedules and prepares the next step while the GPU executes the current step, overlapping CPU and GPU work. 图 3：V1 中的异步调度。CPU 调度并准备下一步，而 GPU 执行当前步骤，CPU 和 GPU 工作重叠。

MRV2 treats async execution as a core assumption and aims for **zero synchronization** between CPU and GPU across all supported model and feature combinations.

Importantly, MRV2 naturally enables async scheduling and speculative decoding together—a combination that was difficult to support cleanly in V1. Because MRV2's input preparation runs on the device, the preparation kernels can directly consume rejection sampling results produced by the GPU. Outputs from each step are transferred asynchronously to the CPU in a separate CUDA stream, fully decoupled from the main computation stream. The same design extends to speculative decoding with structured outputs as well.

![Figure 4: MRV2 async scheduling with speculative decoding. GPU-side prep kernels consume rejection sampling results directly, eliminating CPU–GPU sync points.](https://vllm.ai/blog-assets/figures/2026-03-24-mrv2/async_spec_decoding.png)

Figure 4: MRV2 async scheduling with speculative decoding. GPU-side prep kernels consume rejection sampling results directly, eliminating CPU–GPU sync points.

### 3\. A Triton-Native Sampler

MRV2 reworks sampling with optimized Triton kernels for better control over memory usage and numerics. Specific improvements include:

- **Gumbel-Max sampling kernel** that avoids explicit softmax materialization and uses stateless in-kernel RNG.
- **More efficient top-k logprobs** by finding top-k logits first and computing logprobs only for the selected candidates.
- **More memory-efficient prompt logprobs** through finer-grained chunking, including chunking within a single prompt.
- **Better speculative decoding compatibility** by using indirection (`idx_mapping`) inside kernels rather than expanding request state to match every logits vector.

Together, these changes reduce peak memory usage and make it easier to support rich combinations of sampling parameters.

### 4\. Stronger Modularization

vLLM needs to support a wide range of model architectures, and the existing model runner accumulated considerable complexity as a result. MRV2 addresses this with a new abstraction: **`ModelState`**.

```python
class ModelState(ABC):
    def add_request(self, ...):
    def remove_request(self, ...):
    def get_mm_embeddings(self, ...):
    def prepare_inputs(self, ...):
    def prepare_attn(self, ...):
    def prepare_dummy_inputs(self, ...):
    ...
```

`ModelState` defines the interface for model-specific logic—multimodal embeddings, extra model inputs, attention metadata, CUDA graph capture—so the main runner can stay focused on the common path. This directly addresses a common complaint from both users and contributors: vLLM supports so many models that the shared code can feel convoluted, especially for developers who only care about one model family such as DeepSeek, Qwen, Kimi, or a private internal model.

In addition, MRV2 breaks the runner into smaller files with clearer responsibilities. The existing runner (`gpu_model_runner.py`) had grown into a single file exceeding 6,700 lines; the largest file in MRV2 is now under 1,300 lines.

## Performance

MRV2 is not just a cleanup project. It already delivers measurable wins.

We stress-tested MRV2 by running a very small model (`Qwen3-0.6B`) on a powerful GPU (`1×GB200`), intentionally choosing a small model so that host-side overhead would be proportionally large. In this setup, **MRV2 delivered a 56% throughput increase** by offloading input preparation to GPU.

![Figure 5: Throughput comparison between MRV1 and MRV2 on Qwen3-0.6B with 1×GB200. MRV2 achieves 25K output tok/s vs 16K for MRV1, a 56.2% improvement.](https://vllm.ai/blog-assets/figures/2026-03-24-mrv2/throughput_comparison.png)

Figure 5: Throughput comparison between MRV1 and MRV2 on Qwen3-0.6B with 1×GB200. MRV2 achieves 25K output tok/s vs 16K for MRV1, a 56.2% improvement.

We also measured gains for speculative decoding: **6.3% lower TPOT** on `4×GB200` with `GLM-4.7-FP8` and `MTP=1`. The improvement comes from MRV2's zero-synchronization design, which completely eliminates CPU–GPU sync points when speculative decoding is enabled.

![Figure 6: Mean TPOT comparison between MRV1 and MRV2 on GLM-4.7-FP8 with MTP=1 on 4×GB200. MRV2 achieves 6.3% lower TPOT across request rates.](https://vllm.ai/blog-assets/figures/2026-03-24-mrv2/tpot_mtp.png)

Figure 6: Mean TPOT comparison between MRV1 and MRV2 on GLM-4.7-FP8 with MTP=1 on 4×GB200. MRV2 achieves 6.3% lower TPOT across request rates.

We expect this architectural foundation to matter even more as serving stacks continue to combine async scheduling, speculative decoding, multimodal preprocessing, and increasingly heterogeneous model state.

## Limitations and Current Status

MRV2 is still experimental and under active development. The design is significantly cleaner and early results are strong, but MRV2 is not yet feature-complete. As of v0.18.0, the following features are **not supported**:

- Linear attention models (Qwen3.5, Nemotron 3 Super)
- Spec decoding methods other than Eagle/Eagle3/MTP
- EPLB and DBO
- Logits processors
- LoRA

For a full list, refer to the second page of the [design doc](https://docs.google.com/document/d/1gFqtDkcoqhy9j-X0ndshzbhapX1uNey1-wBENwGPI80/edit?usp=sharing).

We are holding MRV2 to a higher quality bar: when a V1 feature is brought into MRV2, we want to reconsider it from first principles rather than copy over complexity mechanically. For this reason, it may take longer than usual to land changes that touch MRV2.

## Getting Started

1. Install the latest vLLM build.
2. Set `export VLLM_USE_V2_MODEL_RUNNER=1`.
3. Use the existing vLLM APIs as usual—Python API or `vllm serve`.

There are **no user-facing API changes** required.

## Acknowledgments

Woosuk Kwon, Nick Hill, Giancarlo Delfin, Santino Ramos (Inferact), Wentao Ye, Zhanqiu Hu, Lucas Wilkinson (Red Hat), Haoran Zhu (Alibaba)