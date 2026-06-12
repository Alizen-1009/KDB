---
title: "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B"
source: "https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles"
author:
published:
created: 2026-06-12
description:
tags:
  - "clippings"
---
There are some applications that benefit from running LLMs really, really fast. This low-latency regime encompasses applications like chatbots and human-in-the-loop workflows, where users care a lot about seeing responses come back immediately.  
有些应用确实能从运行非常非常快的大型语言模型中受益。这种低延迟体系涵盖了聊天机器人和人机协作流程等应用，用户非常关心能立即收到回复。

Given the importance of these low-latency workloads, we wanted to explore just how fast we can run open-source models on modern GPUs. To really stress-test existing systems, we consider an aggressive low-latency scenario where we generate a single sequence with Llama-3.2-1B. This workload is strongly memory bound – our performance is dominated by how fast we can load model weights from GPU global memory.  
鉴于这些低延迟工作负载的重要性，我们希望探索在现代 GPU 上运行开源模型的速度。为了真正对现有系统进行压力测试，我们考虑了一个激进的低延迟场景，即生成一个包含 Llama-3.2-1B 的序列。这种工作负载对内存非常依赖——我们的性能主要取决于从 GPU 全局内存加载模型权重的速度。

It turns out that popular LLM inference engines – vLLM and SGLang – are only able to use at most 50% of available GPU bandwidth when running this workload on an H100. The root of the problem, which we'll describe more below, is that existing systems break down a model forward pass into around **a hundred separate kernels** that each implement a few operations (e.g. RMS norm, attention, an MLP layer + activation, rotary). Each kernel comes with a setup and teardown period and during this time no useful work gets done – for instance, the all-important task of loading model weights is stalled.  
事实证明，流行的大型语言模型推理引擎——vLLM 和 SGLang——在 H100 上运行该工作负载时，最多只能使用 50%的 GPU 带宽。问题的根源，我们将在下文详细描述，是现有系统将模型前向传递拆分为大约 **一百个独立的内核，每个内核** 实现一些操作（例如，RMS 规范、注意、MLP 层+激活、旋转）。每个内核都有一个设置和拆除期，在此期间没有任何有用的工作——例如，加载模型权重这一至关重要的任务会被搁置。

![Performance comparison graph](https://hazyresearch.stanford.edu/static/posts/2025-05-27-no-bubbles/result.png)

*Figure 1: Speed! Results generated with a 32-token prompt and 128 generated tokens, with no speculation  
图1：速度！结果由32个代币和128个代币生成，无任何猜测*

In this post, we show how we can bypass this problem by merging the entire Llama-1B forward pass into a single "megakernel" that eliminates kernel boundaries altogether. Doing this achieves brr – on an H100, we use 78% of memory bandwidth and outperform existing systems by over 1.5x. (To our knowledge, this is the lowest-latency forward pass for Llama-1B in bfloat16!) In the rest of this post, we'll walk through how and why one would do this. Specifically:  
本文展示了如何通过将整个 Llama-1B 前向传递合并为单一“巨核”，完全消除核边界来绕过这个问题。这样做能实现 BRR——在 H100 上，我们占用了 78%的内存带宽，性能比现有系统高出 1.5 倍以上。（据我们所知，这是 bfloat16 中 Llama-1B 的最低延迟前向传递！）在这篇文章的后半部分，我们将讲解如何以及为什么要这样做。具体来说：

- First, we'll talk about how small kernels lead to AI systems that underutilize the GPU's full bandwidth.  
	首先，我们将讨论小内核如何导致 AI 系统未能充分利用 GPU 的全部带宽。
- Second, we'll describe three important points about how we built our megakernel: how we fused lots of kernels together, how we share hardware resources across them to minimize overhead, and how we synchronize them efficiently.  
	其次，我们将介绍我们构建超级内核的三个重要点：如何融合大量内核，如何共享硬件资源以减少开销，以及如何高效同步。

If you're interested in learning more of the details or using these ideas yourself, we're [open-sourcing all of our code here](https://github.com/HazyResearch/Megakernels).  
如果你有兴趣了解更多细节或自己动用这些想法，我们这里将 [所有代码开源](https://github.com/HazyResearch/Megakernels) 。

## Separate Kernels Kill the Vibe分开的玉米粒会破坏氛围

In general, the way one runs code on a GPU is by launching a "kernel" – a small program that does a well-defined operation (e.g. RMS norm, MLP). Today, all AI workloads run as long sequences of relatively small kernels. To get an initial sense, let's look at the operations in the Llama-1B transformer block, and some example kernel boundaries of how they might be divided up (Figure 2).  
一般来说，在 GPU 上运行代码的方式是启动一个“内核”——一个执行明确定义操作的小程序（例如 RMS 范数、MLP）。如今，所有 AI 工作负载都以相对较小的内核组成的长序列运行。为了初步了解，我们来看 Llama-1B 变换器块中的操作，以及一些可能划分的核边界示例（见图 2）。

![Kernel boundaries diagram](https://hazyresearch.stanford.edu/static/posts/2025-05-27-no-bubbles/kernel_boundaries.png)

*Figure 2: An example set of kernel boundaries for the Llama-1B transformer block. Red boxes delineate the work done by individual kernels.  
图 2：Llama-1B 变压器块的核边界示例集。红色方框标示单个玉米粒所做的功。*

As we described earlier, decoding a single sequence with Llama-1B is a purely memory-bound workload: our performance depends on being able to **always** be loading weights from GPU global memory. So, why are existing approaches so far from using the full bandwidth of the GPU?  
正如我们之前所描述的，用 Llama-1B 解码单个序列是纯内存负载：我们的性能依赖于 **始终** 能从 GPU 全局内存加载权重。那么，为什么现有的方法距离充分利用 GPU 的全部带宽还差得远呢？

When we dug into it, we noticed a key problem was that the current kernel-based approach to running models introduces stalls that prevent us from constantly loading memory:  
深入研究后，我们发现一个关键问题是当前基于内核的模型运行方式引入了停滞，导致我们无法持续加载内存：

- First: GPU kernels are launched with a strict ordering, so that a thread block in one kernel can't start until all thread blocks in previous kernels have completely finished. Consequently, every time we start a kernel, we have to wait for all the straggler thread blocks from the prior one to finish. For example, if a kernel runs 512 thread blocks (like our Llama-1B down projection), but we only have 148 streaming multiprocessors (like on a B200), we end up with 80 empty SM's at the end.  
	首先：GPU 内核启动时有严格的顺序，因此一个内核中的线程块必须在之前内核的所有线程块完全完成后才能启动。因此，每次启动内核时，都必须等待前一个内核所有落后线程块都完成。例如，如果一个内核运行 512 个线程块（类似我们的 Llama-1B 下投影），但我们只有 148 个流式多处理器（类似 B200），最终会有 80 个空的 SM。
- Second, as we've [previously highlighted](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla), each kernel launch and teardown incurs costs. In principle, NVIDIA's CUDA graphs can help hide costs, but by our measurements they still leave a lot on the table. For a simple dummy kernel (which dumps a start time, sleeps, and dumps an end time) on an H100, we find that running on a CUDA stream incurs a launch cost of about 2.1 microseconds, and with CUDA graphs the launch cost only decreases to around 1.3 microseconds – time spent with the GPU doing no useful work! We'd like to have the GPU spend all of its time doing useful work.  
	其次，正如 [我们之前强调](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla) 的，每次内核的启动和重组都会产生成本。原则上，英伟达的 CUDA 图表可以帮助掩盖成本，但根据我们的衡量，它们仍然留下了不少风险。对于一个简单的假内核（它会倾出启动时间、休眠时间和结束时间）在 H100 上运行，我们发现运行在 CUDA 流上大约需要 2.1 微秒的启动成本，而使用 CUDA 图表时，启动成本仅降低到大约 1.3 微秒——这也就是 GPU 花费的时间没有任何有用的工作！我们希望 GPU 能把所有时间都用来做有用的工作。
- Finally, even after we start the next kernel, we still have to wait to load weights and activations before any compute can start. These latencies leave the GPU sitting idle for thousands of cycles! Ideally, we'd start loading the next weights while the previous computations and stores are happening. NVIDIA has also built a mechanism for this called [Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization) (PDL), which allows the next kernel to start preparing while the previous kernel is running, but we found it still introduces unnecessary stalls because the PDL synchronization mechanism (cudaGridDependencySynchronize) is very coarse. For example, it means we have to wait for all queries, keys, and values to complete in order to start attention, as opposed to starting heads as soon as they are ready. We'll later show another specific case of where this is useful in Llama-1B.  
	最后，即使启动下一个内核，我们还得等加载权重和激活，计算才能开始。这些延迟让 GPU 闲置了数千个周期！理想情况下，我们应该在之前的计算和存储还在进行时，开始加载下一个权重。NVIDIA 还为此构建了一个机制，称为 [程序依赖启动](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization) （PDL），允许下一个内核在上一个内核运行时开始准备，但我们发现由于 PDL 同步机制（cudaGridDependencySynchronize）非常粗糙，仍会引入不必要的停滞。例如，这意味着我们必须等待所有查询、键和值都完成后才能开始关注，而不是一准备好就开始关注。稍后我们将展示另一个具体案例，说明这在 Llama-1B 中非常有用。

Taken together, these form the "memory pipeline bubbles" our title references – and they represent a key reason that we're **not always loading from memory**. For short operations, these pauses add up, wasting a huge chunk of potential bandwidth. In part, this is because Llama-1B (actually 1.24B parameters) in batch size 1 is just so... small: if each operation is really fast, then the time spent in-between them really starts to matter.

To illustrate the magnitude of the problem: for single-sequence generation in 16-bit precision on a single H100, the **memory limit** is 3.35TB/s / 2.48GB = ~1350 forward passes per second. But with 7 kernel launches per layer, and 16 layers, even with an optimistic 5 us of stalling per kernel (counting stragglers, kernel launch, and memory latencies), generation would run at just ~770 forward passes per second. In practice, it's often worse. On low-latency workloads, GPUs spend only a fraction of their time actually doing any useful work!

So while CUDA does provide some existing features (e.g. graphs, streams, PDL) to partially solve these problems, we wanted to see if a different approach could solve all of these problems, where we just fuse the entire model forward pass into a single kernel.

## How to Megakernel

Next, we'll show you how we fused a whole Llama forward pass into a single kernel, and our methods for resolving three key problems:

1. Fusing dozens of operations is hard to do from scratch. We need a mechanism for executing these operations within the megakernel.
2. In order to overlap multiple operations on the same hardware, we need to prevent contention over limited resources, such as shared memory.
3. The GPU synchronizes after each kernel in the traditional kernel model. Without kernels, we have to synchronize the GPU all by ourselves!

Let's start with the first issue:

#### Issue 1/3: Fusing Lots of Operations

Traditional kernel fusion generally merges just two or three operations together. In contrast, we need to fuse about a hundred. Consequently, we need to have a sensible abstraction for how we can actually program a megakernel.

Our approach is built on an on-GPU interpreter – essentially a more sophisticated version of our infrastructure underlying [ThunderMLA](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla). Our interpreter is designed such that each streaming multiprocessor (SM) within the GPU receives a sequence of **instructions** (each implemented using the same CUDA template) and executes them. We **schedule** each SM's instruction sequence ahead of time on the Python side, and notably we can reuse each schedule for hundreds of forward passes!

For our end-to-end Llama forwards pass megakernel, we define the following set of instructions:

- A fused RMS norm & QKV & RoPE instruction.
- An attention computation instruction.
- An attention reduction instruction (for ThunderGQA on long sequences).
- An O-projection + residual instruction.
- A fused RMS norm & up-gate & SiLU instruction.
- A down-projection + residual instruction.
- An RMS norm & language modeling head instruction, for computing the final token logits.

We implement each of these instructions using a common [CUDA template](https://github.com/HazyResearch/Megakernels/blob/main/util/mk_init/sources/src/%7B%7BPROJECT_NAME_LOWER%7D%7D.cu) (with load, store, compute boilerplate functions), facilitating interoperability within our interpreter framework.

#### Issue 2/3: Sharing Shared Memory to Eliminate Memory Bubbles

The instruction-and-interpreter structure lets us cleanly organize our megakernel. However, we haven't yet addressed the key issue: making sure that model weights are always being loaded in order to maximize memory bandwidth utilization.

The reason why a megakernel lets us solve this problem is that we can pipeline memory loads across instructions: our interpreter will start loading the model weights for an instruction as soon as it can, even if a previous instruction is still finishing up (e.g. storing out its results to global memory). It's this tight transitioning between instructions that minimizes the memory bubbles that would otherwise appear if we launched multiple kernels.

However, there's a catch: loading the weights from global memory for the next instruction doesn't do you much good if you have no place to put the data you loaded! More precisely, all of our weight matrices are loaded from GPU global memory into our SM's "shared memory" – NVIDIA's term for the fast memory on each SM. Shared memory is a scarce resource on each SM, and we can't start a load for a new instruction if a previous instruction is using all of it. This necessitates a way to keep track of which instruction is using which piece of shared memory and quickly transition shared memory to the next instruction when the current instruction is done with it.

We accomplish this by **paging** shared memory. We first divide the first 213kB of shared memory on an H100 into 13 16KiB pages, and use remaining shared memory for special purposes, like storing instruction parameters. To use one of these pages, instructions have to explicitly request and release them from the interpreter. The interpreter automatically passes released pages to the next instruction, allowing them to start issuing memory loads as early as shared memory becomes available.

#### Issue 3/3: Synchronization

![Thanos illustration](https://hazyresearch.stanford.edu/static/posts/2025-05-27-no-bubbles/thanos.png)

While megakernels let us minimize pipeline bubbles, they also introduce a new problem: synchronization. The performance limitation with the normal many-kernel execution model is that no thread blocks in a kernel can start until all thread blocks in previous kernels are finished. However, it's precisely this property that makes it easy to manage data dependencies. When a kernel launches, CUDA guarantees that all of the kernel's input tensors have already been produced and are safe to read from immediately.

With megakernels, we have no such guarantees: when an SM starts to execute a new instruction, its inputs might not be ready! To address this, we explicitly synchronize the instructions inside of our megakernel. We accomplish this with a simple counter system. Before the megakernel launches, we initialize an array of counters (i.e. integers) in GPU global memory with a starting value of zero. Whenever an instruction completes, it increments one of these counters. Similarly, whenever a new instruction starts, it must wait for some of these counters to reach a target value, indicating that all of its dependencies have finished.

One optimization this enables is in the big multi-layer perceptrons (MLPs) in Llama-1B.

- In a naive implementation using PDL, one must await completing the whole hidden state before beginning the down projection matrix multiply.
- We instead produce and consume the intermediate state in four chunks, each with their own counter. This way, an instruction for the down projection only needs to wait for its input chunk to finish.

## Putting It All Together

To our knowledge, our H100 megakernel represents the first time anyone has run the forward pass for a 16-bit 1B+ parameter language model in under one millisecond on a GPU. Our B200 implementation pushes this even further to under 680 microseconds per forward pass!

As shown in Figure 1, our megakernel outperforms vLLM and SGLang baselines (which use CUDA graphs and torch compilation):

- On an H100, our megakernel runs almost 2.5x faster than vLLM and over 1.5x faster than SGLang.
- On a B200, the gap with vLLM rises to over 3.5x, and we remain more than 1.5x faster than SGLang, too.

We're still actually quite a ways off from the theoretical limit on a B200, which is around ~3,000 forward passes per second. Part of this gap is because this theoretical limit is based purely on memory bandwidth – but we still have to wait to load activations. And although these activations are small (and don't cost a lot of bandwidth), there are still latencies in loading them that we can't hide. A breakdown of the runtime of our current B200 forward pass (total runtime 600 microseconds):

- 250 microseconds are spent storing activations, awaiting consistency, and loading them. This is about 20% higher than a simple model would suggest: since each instruction has a dependence on the last one, we need to pay two load latencies (check ready, and then load activations) and two store latencies (store activations, then mark ready) per instruction. Using ~500 nanoseconds latency per load / store, this would impose about 200 microseconds of overhead. (We suspect some of the remaining 50 microseconds comes from time spent processing atomics in global memory.)
- 200 microseconds are spent actually running RMS norm and matrix-vector computations. 95% of this portion is devoted to matrix-vector. On Blackwell, we find that using the tensor cores is marginally helpful for this; on Hopper, we find it better to simply run on the CUDA cores. This difference comes from the fact that both GPUs have relatively similar CUDA core performance, but Blackwell tensor cores are much faster.
- 30 microseconds are spent awaiting weights from global memory (pipelining works!) Of these, 40% are spent in the LM head, which is the best-pipelined part of the whole megakernel due to its homogeneity and huge size.
- 40 microseconds are spent on low-level synchronization overhead across warps. A key issue here is that CUDA's asynchronous barriers are relatively slow, even when they're already in the "pass" state, requiring about 60 nanoseconds each time.
- 80 microseconds are on setup and various other overheads (e.g. passing instruction barriers, marking pages as complete, etc.)

We think there's probably more to do on each of these, but that'll have to wait for a future update!

## The Megakernel Cinematic Universe

In this blog, we focus narrowly on designing a megakernel for low-latency, batch-size one LLM inference. However, we believe that the ability to more precisely control GPU execution with megakernels can more generally be applied to accelerate a much broader set of AI workloads. Stay tuned!

![Sonic illustration](https://hazyresearch.stanford.edu/static/posts/2025-05-27-no-bubbles/sonic.png)

**The Main Message of this Blog Post**

If you'd like to learn more, please reach out to Ben or Jordan! Please include a tribute of at least five pictures of kittens in your email.

And many, many thanks to Together AI for generously providing us with B200s and H100s to do this work, which would not have been possible without them!

See also: [**pretty big kernels**](https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla) | [**regular kernels**](https://hazyresearch.stanford.edu/blog/2024-05-12-tk)