# 现代推理框架中的 Torch Compile 作用

## 背景

问题是：成熟推理框架已经有 FlashAttention、FlashInfer、CUTLASS、fused MoE、CUDA Graph 等专用实现，`torch.compile` 是否已经很少使用、性能作用也基本不大。

核对口径：本文基于本地 wiki、vLLM / SGLang / TensorRT-LLM 官方 `main` 源码与文档进行机制判断。精确版本分别固定到：

- vLLM `2dfb8ba59098eb489197e1b4c643addffd51592e`
- SGLang `04374ba5e0b95be63990f6c64a3b210a6b70e7cc`
- TensorRT-LLM `aafc4ebf808f4a69e69c129721c918f90ce2fb1f`

## 核心观点

**不能笼统地说 `torch.compile` 已经很少使用、作用基本不大。** 更准确的判断是：

1. 在成熟 serving 引擎中，Attention、GEMM、MoE、量化等重型热点通常由手写 CUDA/Triton/CUTLASS 或专用库 kernel 承担；因此期待“原生 PyTorch 模型套一层 stock `torch.compile` 就获得巨大端到端加速”，通常不现实。
2. 但 `torch.compile` 的角色已经从“替所有算子自动生成最快 kernel”扩展为**图捕获与统一 IR、图分区、自定义 pattern rewrite、轻量融合、shape specialization、编译缓存，以及 Piecewise CUDA Graph 的组织基础设施**。
3. 当前 vLLM V1 明确把 `torch.compile` 默认启用并称为框架关键组成；SGLang 和 TensorRT-LLM 也在 Piecewise CUDA Graph 或图级融合路径中使用它。
4. 所以出现的是**作用迁移和分层**，不是简单退出：重型算子交给专用 kernel，`torch.compile` 负责它们之间的图级 glue、跨模块优化和执行分区。

## 机制拆解

### 1. 为什么会产生“作用不大”的印象

成熟推理框架的主耗时通常集中在：

- Attention / MLA backend；
- GEMM 与量化 GEMM；
- MoE dispatch、grouped GEMM 与通信；
- collectives；
- KV Cache 读写和 serving 调度。

这些路径往往已经是 custom op，Dynamo/Inductor 会把它们视作黑盒，无法进入内部重新做 kernel fusion。CUDA Graph 又能减少剩余 kernel launch 的 CPU 提交成本。因此，**只看 stock Inductor 自动 fusion 对端到端 tokens/s 的边际贡献，它确实可能不如早期或不成熟模型路径显眼。**

但这不等于 `torch.compile` 没有参与执行系统。

### 2. `torch.compile` 在推理框架中的四种角色

#### A. 捕获模型图与建立统一 IR

框架可得到完整或分段的 FX/ATen 图，在图上识别模块边界、动态 token 维和依赖关系。即使后端最终仍调用专用 kernel，这个图也可用于调度和变换。

#### B. 轻量融合与跨模块 pattern rewrite

典型目标不是重写 FlashAttention，而是处理它周围的：

- residual / RMSNorm；
- RoPE / QK norm；
- quant/dequant；
- activation、gate、elementwise chain；
- AllReduce + residual + RMSNorm 等跨模块组合。

这类融合的价值主要是减少小 kernel 和中间 HBM 往返。

#### C. 图分区与 Piecewise CUDA Graph

Attention、MoE 或 collective 可作为 split/custom op 留在图外，其他 graph-safe 区段由框架 capture/replay。这里 `torch.compile` 的关键价值是**获得图、识别 split point、拼接 eager op 与 captured segment**；不一定要求所有区段都由 Inductor 重新生成 kernel。

#### D. Shape specialization、autotune 与缓存

框架可以为常见小 batch/token shape 生成特化版本，并把编译结果缓存或 AOT 分发，避免在线请求触发不可预测的编译延迟。

## 框架实现差异

### vLLM V1

**已知事实：**

- 官方 `docs/design/torch_compile.md` 直接写明：V1 中 `torch.compile` 默认开启，是框架的关键组成。
- `CompilationMode.VLLM_COMPILE=3` 是 V1 默认模式，使用 vLLM 自定义 Inductor backend、缓存、piecewise compilation、shape specialization 和 custom passes。
- Attention 通常被包装成 custom op，Dynamo 不追踪其内部；图再按 attention 等 splitting ops 分区。
- vLLM 还用 compile backend 管理 piecewise CUDA Graph；特定静态 shape 可额外 autotune，但由于启动成本，面向具体 shape 的 autotune 并非默认全开。

**解释：** vLLM 不是依赖 stock `torch.compile(model)` 自动优化整个 serving 系统，而是把 PyTorch compile 栈改造成自己的图编译与 CUDA Graph 基础设施。

### SGLang

**已知事实：**

- 独立的 `--enable-torch-compile` 仍标为 experimental，默认关闭；只看这个开关会让人觉得 SGLang 很少使用 compile。
- 但官方 Piecewise CUDA Graph 文档说明，该路径使用 `torch.compile(..., backend=SGLangBackend)` 捕获模型图、按 split op 分割，再把可捕获区段交给 piecewise CUDA Graph；支持配置中 PCG 默认启用。
- Piecewise backend 的 compiler 可以是 `eager` 或 `inductor`。即使选择 `eager`，仍可能使用 Dynamo/FX 捕获与分区，而不依赖 Inductor codegen。
- 兼容性仍有限：MoE A2A、LoRA、部分多模态/并行/PD 路径可能触发自动禁用或回退，具体以部署 commit 为准。

**解释：** SGLang 区分了“完整模型 Inductor 优化”和“用 compile 栈组织 piecewise graph”。前者不是默认主路径，后者的系统作用更重要。

### TensorRT-LLM

**已知事实：**

- 官方文档写明其 PyTorch 路径用 `torch.compile` 做 lightweight vertical fusion 和 Piecewise CUDA Graph。
- Attention、MoE routed experts、MTP 等热模块可包装为大型 custom op，作为 compile 黑盒；compile 主要处理其外围图。
- 自定义 backend 在 ATen IR 上进行 pattern rewrite，例如 AllReduce + residual + RMSNorm，以及量化组合，还用于 re-inplace、自动多流和 Piecewise CUDA Graph。

**解释：** TensorRT-LLM 的核心性能仍高度依赖专用 kernel 和 runtime，但它也在利用 `torch.compile` 作为图级控制面，而不只是 Inductor kernel generator。

### Kimi K3 / MLA 的反例边界

[[../../wiki/sources/A Preview of Production-Scale Kimi K3 Support on vLLM|Kimi K3 Preview]] 记录：此前模型的 vLLM 路径较依赖 `torch.compile` custom fusion，出现启动慢且仍有小 kernel 未融合的问题；K3 改成手工融合 MLA module，并为 prefill/decode 采用不同 fusion pattern。

这个案例支持的是：

- **模型关键热路径上，通用 compile fusion 可能不够，手工融合会取代它；**
- 但它不支持“整个推理框架不再需要 compile”，因为框架仍可在外围图、图分区、CUDA Graph 和其他模型路径中使用 compile。

## 对比分析

| 判断 | 是否成立 | 边界 |
|---|---|---|
| 成熟框架很少直接依赖 stock `torch.compile(model)` 获得全部性能 | 基本成立 | 热点通常已有专用 kernel/runtime |
| `torch.compile` 对 Attention/GEMM/MoE 核心 kernel 的自动优化空间有限 | 常常成立 | custom op 内部对 compiler 不可见 |
| `torch.compile` 在当前推理框架里很少使用 | 不成立 | vLLM 默认启用；SGLang/TRT-LLM 用于 piecewise graph 或图优化 |
| CUDA Graph 已经替代 `torch.compile` | 不成立 | CUDA Graph 负责 replay，compile 可负责捕获、分区和融合 |
| `torch.compile` 对任何模型的端到端提升都很大 | 不成立 | 收益取决于小算子比例、shape、compile coverage 和现有 kernel 优化程度 |
| 专用手工融合正在取代部分 Inductor 自动融合 | 成立 | 尤其是模型特有、性能关键和执行形态复杂的路径 |

## 性能权衡

### 更可能有价值的场景

- 新模型刚接入、尚未拥有完整手工 fused kernel；
- residual/norm/activation/quant 等外围小算子很多；
- 小 batch decode 或短 prefill 中 host/kernel launch 比例较高；
- 需要 Piecewise CUDA Graph 支持动态 token shape；
- 常见 shape 稳定，编译与 autotune 成本可以缓存并摊销；
- 需要在全图上做跨模块 fusion 或多流调度。

### 更可能边际较小的场景

- 大 prefill 中 GEMM/Attention 已占绝大多数时间；
- 热点全部落在高度优化 custom op 内部；
- CUDA Graph 已经消除大部分 host launch gap，且外围 HBM traffic 很少；
- shape/控制流过于动态，导致 graph break、回退或大量变体；
- 冷启动和部署弹性比稳态吞吐更重要，而编译缓存/AOT 尚未解决。

## 工程含义

不要只比较“开/关 `torch.compile`”两个总开关，而应拆成四类问题：

1. **Compile coverage**：多少时间在 compiled segments，多少在 custom/eager ops？
2. **Kernel effect**：kernel 数、中间 HBM traffic、fusion pattern 是否变化？
3. **Graph effect**：Piecewise/full CUDA Graph 的 replay 命中率和 host gap 是否变化？
4. **Lifecycle cost**：冷启动、编译、autotune、cache 命中与额外显存是多少？

建议至少比较：

```text
Eager
Compile only
CUDA Graph only
Compile + CUDA Graph
专用手工 fused path
```

结论应绑定具体模型、框架 commit、GPU、batch/token 分布和目标指标，不能从某个单模型的优化迁移直接推断整个框架。

## 结论

更准确的一句话是：

> 在现代推理框架里，`torch.compile` 作为“万能自动 kernel 优化器”的边际作用确实下降了；但它作为图捕获、图分区、定制 pass、轻量融合和 Piecewise CUDA Graph 基础设施，反而正在被更深地集成。重型热点由专用 kernel 负责，compile 负责把这些 kernel 组织成高效执行图。

## 官方实现核对

- [vLLM `torch.compile` integration @ `2dfb8ba`](https://github.com/vllm-project/vllm/blob/2dfb8ba59098eb489197e1b4c643addffd51592e/docs/design/torch_compile.md)
- [vLLM `CompilationConfig` @ `2dfb8ba`](https://github.com/vllm-project/vllm/blob/2dfb8ba59098eb489197e1b4c643addffd51592e/vllm/config/compilation.py)
- [SGLang Piecewise CUDA Graph @ `04374ba`](https://github.com/sgl-project/sglang/blob/04374ba5e0b95be63990f6c64a3b210a6b70e7cc/docs/docs/advanced_features/piecewise_cuda_graph.mdx)
- [SGLang server args @ `04374ba`](https://github.com/sgl-project/sglang/blob/04374ba5e0b95be63990f6c64a3b210a6b70e7cc/python/sglang/srt/server_args.py)
- [TensorRT-LLM Torch Compile & Piecewise CUDA Graph @ `aafc4eb`](https://github.com/NVIDIA/TensorRT-LLM/blob/aafc4ebf808f4a69e69c129721c918f90ce2fb1f/docs/source/features/torch_compile_and_piecewise_cuda_graph.md)

## 待核实

- 上述均为 `main` commit，未必等同于用户生产环境安装的 release；默认开关和兼容矩阵需按实际版本复核。
- 各框架没有可跨模型复用的统一 compile 加速百分比；必须做本地消融实验。
