---
type: map
topic: GPU 编程
---

# GPU 编程

## 导读

这是本库最大的主题，分五层读。

1. **执行模型**：[[../concepts/GPU执行模型|GPU执行模型]] + [[../concepts/CUDA内存层次|CUDA内存层次]] —— 后面所有优化都在解释这两页里的约束。
2. **性能陷阱**：[[../concepts/Occupancy|Occupancy]]、[[../concepts/Warp Divergence|Warp Divergence]]、[[../concepts/Bank Conflict|Bank Conflict]]、[[../concepts/内存合并访问|内存合并访问]]、[[../concepts/Tail Effect|Tail Effect]]。
3. **手撕套路**：[[../concepts/Block Reduce|Block Reduce]]、[[../concepts/Warp Shuffle Reduce|Warp Shuffle Reduce]]、[[../concepts/Grid-stride Loop|Grid-stride Loop]]、[[../concepts/Histogram|Histogram]]、[[../concepts/Tiling|Tiling]]、[[../concepts/动态共享内存|动态共享内存]] —— 面试高频，配 [[../sources/秋招CUDA手撕题复盘（附代码）|手撕题复盘]] 看。
4. **减少调度边界**：[[../concepts/Persistent Kernel|Persistent Kernel]] 让驻留 worker 连续处理多个 tiles，[[../concepts/Cluster Launch Control|Cluster Launch Control]] 为 Blackwell 补动态取活；[[../concepts/算子融合|算子融合]] → [[../concepts/Programmatic Dependent Launch|Programmatic Dependent Launch]] → [[../concepts/Megakernel|Megakernel]] 则逐步跨越更大的 kernel/算子边界。
5. **写 kernel 的抽象层**：[[../concepts/Triton|Triton]]、[[../concepts/CuTe DSL|CuTe DSL]]、[[../concepts/Torch Compile|Torch Compile]]、[[../concepts/CODA|CODA]]。

架构代际先读 [[../entities/NVIDIA Ampere|NVIDIA Ampere]] → [[../entities/NVIDIA Hopper|NVIDIA Hopper]] → [[../entities/NVIDIA Blackwell|NVIDIA Blackwell]]，观察 `cp.async → TMA/WGMMA → tcgen05/TMEM` 如何扩大异步流水；再读 [[../entities/NVIDIA Rubin|NVIDIA Rubin]]，把 SM/Tensor Core/SFU/HBM 的硬件变化映射到 TMA 动态 expert 搬运、L2 priority、跨 kernel dependency、counted NVLink communication 与 sparse Attention staging。

<!-- BEGIN AUTO：以下由 scripts/update_index.py 生成，改动会被覆盖 -->

## 概念（26）

- [[../concepts/Bank Conflict|Bank Conflict]]
- [[../concepts/Block Reduce|Block Reduce]]
- [[../concepts/CODA|CODA]]
- [[../concepts/CUDA Graph 执行模式|CUDA Graph 执行模式]]
- [[../concepts/CUDA Kernel|CUDA Kernel]]
- [[../concepts/CUDA内存层次|CUDA内存层次]]
- [[../concepts/Cluster Launch Control|Cluster Launch Control]]
- [[../concepts/CuTe DSL|CuTe DSL]]
- [[../concepts/GPU执行模型|GPU执行模型]]
- [[../concepts/Grid-stride Loop|Grid-stride Loop]]
- [[../concepts/Histogram|Histogram]]
- [[../concepts/MegaMoE|MegaMoE]]
- [[../concepts/Megakernel|Megakernel]]
- [[../concepts/Occupancy|Occupancy]]
- [[../concepts/Persistent Kernel|Persistent Kernel]]
- [[../concepts/Programmatic Dependent Launch|Programmatic Dependent Launch]]
- [[../concepts/Tail Effect|Tail Effect]]
- [[../concepts/Tensor Memory|Tensor Memory]]
- [[../concepts/Tiling|Tiling]]
- [[../concepts/Torch Compile|Torch Compile]]
- [[../concepts/Triton|Triton]]
- [[../concepts/Warp Divergence|Warp Divergence]]
- [[../concepts/Warp Shuffle Reduce|Warp Shuffle Reduce]]
- [[../concepts/内存合并访问|内存合并访问]]
- [[../concepts/动态共享内存|动态共享内存]]
- [[../concepts/算子融合|算子融合]]

## 实体（11）

- [[../entities/CAKE KDA|CAKE KDA]]
- [[../entities/Colfax Research|Colfax Research]]
- [[../entities/FlashInfer|FlashInfer]]
- [[../entities/FlashKDA|FlashKDA]]
- [[../entities/HazyResearch|HazyResearch]]
- [[../entities/Megakernels|Megakernels]]
- [[../entities/NVIDIA Ampere|NVIDIA Ampere]]
- [[../entities/NVIDIA Blackwell|NVIDIA Blackwell]]
- [[../entities/NVIDIA Hopper|NVIDIA Hopper]]
- [[../entities/NVIDIA Rubin|NVIDIA Rubin]]
- [[../entities/阿里云 PAI 团队|阿里云 PAI 团队]]

## 来源（14）

- [[../sources/CUDA优化维度框架|CUDA优化维度框架]]
- [[../sources/CUDA内存层次与动态共享内存问答整理|CUDA内存层次与动态共享内存问答整理]]
- [[../sources/Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs|Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs]]
- [[../sources/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B|Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B]]
- [[../sources/MegaMoE — 让 all-to-all 消失|MegaMoE — 让 all-to-all 消失]]
- [[../sources/Nvidia Rubin架构分析预览|Nvidia Rubin架构分析预览]]
- [[../sources/PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化|PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化]]
- [[../sources/REMINDER FF-KDA & CAKE KDA Highlights|REMINDER FF-KDA & CAKE KDA Highlights]]
- [[../sources/你一定要知道：CUDA优化六要|你一定要知道：CUDA优化六要]]
- [[../sources/多卡GPU监控与SM执行模型面试整理|多卡GPU监控与SM执行模型面试整理]]
- [[../sources/斯坦福CS336 Lecture 5 - GPUs|斯坦福CS336 Lecture 5 - GPUs]]
- [[../sources/秋招CUDA手撕题复盘（附代码）|秋招CUDA手撕题复盘（附代码）]]
- [[../sources/译 NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell|译 NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell]]
- [[../sources/还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速|还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速]]
