---
type: concept
topic: GPU 编程
sources: 2
updated: 2026-06-21
---

# Torch Compile

## 定义

PyTorch 2.x 提供的图捕获与编译入口，用于把 Python 层模型或函数自动转换成更高效的执行计划和底层 kernel。

## 它解决什么问题

- 自动完成部分算子融合与代码生成
- 减少开发者手工写 CUDA / Triton kernel 的次数

## 核心机制

- 捕获 Python / PyTorch 计算图
- 对图做优化、融合和代码生成
- 将高层实现降到更适合当前硬件和算子模式的执行路径
- 在 CODA 来源中，`cuBLAS + torch.compile` 被作为基线之一；这体现了通用图编译融合与专门的 `GEMM + epilogue` 重写之间的分工差异。

### 捕获、编译与缓存

- **TorchDynamo** 在 Python frame 层解释字节码，将可捕获的 Tensor 运算提取为 FX 图，并生成 guards 与剩余 Python 字节码。
- **AOTAutograd** 在训练路径提前构造反向计算，划分前向和反向图，并处理需要保存或重算的中间值。
- **TorchInductor** 对图做降低、融合与调度，生成设备代码，也可调用现有算子。以 `(x + bias) * scale` 为例，融合有机会省掉一次 kernel launch 和中间张量的全局内存往返。

**Graph break** 发生在追踪时。默认模式下可编译已捕获的区域，运行断点处的 Python，再尝试继续追踪。**Recompile** 通常发生在后续调用时。若已有版本的 guards 都无法匹配当前 shape、stride、dtype 或 Python 参数，可能编译新版本。即使整段可捕获，也可能发生重编译。

### 如何选择编译边界

1. 先 profile eager 的真实负载。连续小 kernel、临时张量和 CPU dispatch 空隙是优先验证的信号。若耗时集中在已有的 FlashAttention、GEMM 或 GDN recurrent kernel，先量化其周围仍可融合的工作占比。
2. 从稳定的较高层 Tensor 入口试起，例如模块 forward 或单个 train step。热路径断图、guard 频繁失败或编译时间过长时，再缩到 block、MLP 或连续的 norm/gate 子图。
3. 用 `TORCH_LOGS="graph_breaks,recompiles"` 查断点和 guard 失败，用 `output_code` 查生成的 kernel。大模型可用 `TORCH_TRACE` 与 `tlparse` 看全局。局部用 `backend="eager", fullgraph=True` 暴露断点。排查编译报错时依次试 `eager`、`aot_eager`、`inductor`，缩小问题所在层。
4. 用实际 shape 分布测首次编译和稳态延迟。对比 eager 与 compiled 的 kernel 数、GPU 时间和 launch 间隙。Serving 分开测 prefill/decode 的 TTFT/TPOT，并保持 CUDA Graph 设置一致以隔离 compile 收益。动态 shape 或分桶应以真实 guard 失败原因和 A/B 结果为依据。

- [PyTorch 编译模型与边界选择](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html)
- [PyTorch 编译 profiling](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)

## 与算子融合、CUDA Graph 的关系

- `torch.compile` 可能通过图优化与代码生成实现 [[算子融合]]，但二者不等价：手写 Triton/CUDA、CUTLASS epilogue 和库内 fused kernel 也能完成融合，且编译器未必能自动发现所有数学重写与硬件特化机会。
- [[CUDA Graph 执行模式|CUDA Graph]] 位于更靠后的执行提交层：它 capture/replay 编译后的一串 kernel，主要降低 CPU launch overhead，不会自动把多个 kernel 融成一个，也不会自动消除中间张量的 HBM 往返。
- 常见组合是先完成 compile 与 warm-up，再 capture 稳定的 kernel 序列；dynamic shape、graph break、重新编译、lazy initialization 和 capture-size 覆盖会共同决定该组合是否有效。

## 在现代推理框架中的角色

成熟 serving 引擎通常不会把 Attention、GEMM、MoE 等重型热点完全交给 stock Inductor 自动生成；这些路径更多由手写 CUDA/Triton/CUTLASS、FlashInfer 或框架专用 custom op 承担。但这不等于 `torch.compile` 很少使用，它的角色正在转向：

- 图捕获与统一 FX/ATen IR；
- custom op 之间的轻量融合和跨模块 pattern rewrite；
- dynamic token shape 下的图分区；
- Piecewise CUDA Graph 的 split/capture/replay 组织；
- shape specialization、autotune、AOT artifact 与编译缓存。

截至官方 `main` 实现核对：

- [[../entities/vLLM|vLLM]] V1 默认使用 `VLLM_COMPILE`，官方称 `torch.compile` 为架构关键组成；它把 Attention 等复杂路径包装成 custom op，并用自定义 backend 做缓存、piecewise compilation、shape specialization 和 custom passes。
- [[../entities/SGLang|SGLang]] 的独立全模型 `--enable-torch-compile` 仍是默认关闭的实验开关，但其 Piecewise CUDA Graph 在支持配置中使用 `torch.compile(..., backend=SGLangBackend)` 捕获和分割模型图；piecewise compiler 可选 eager 或 Inductor。
- [[../entities/TensorRT-LLM|TensorRT-LLM]] 的 PyTorch 路径用 `torch.compile` 做 lightweight vertical fusion、跨模块 pattern rewrite 和 Piecewise CUDA Graph；Attention、MoE、MTP 等热模块仍可作为大型 custom op 黑盒。

因此应区分两种判断：`torch.compile` 作为“万能自动 kernel 生成器”的边际收益在成熟热路径上可能有限；作为图捕获、图变换和执行分区基础设施，它仍可能是默认或关键路径。[[../../output/reports/现代推理框架中的Torch Compile作用|详细报告]]。

## 关键权衡

- 对很多常见模式能带来“低改动、高收益”的优化
- 但收益依赖动态图特征、输入形状、图可捕获性和后端成熟度
- 面对需要代数重排和 GEMM epilogue 特化的路径，通用图编译未必能自动发现所有优化机会；这类场景可能仍需要 [[CODA]]、[[Triton]]、CUTLASS/[[CuTe DSL]] 或手写 kernel。
- 在 `vLLM` 语境里，`torch.compile` / vLLM compile 与 CUDA Graphs 是两层优化：`cudagraph_mode=NONE` 只关闭 CUDA Graphs，`--enforce-eager` 才表示关闭 compile 集成并完全走 eager mode。

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/vLLM]]
- [[../entities/SGLang]]
- [[../entities/TensorRT-LLM]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]
- [[../sources/还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速]]

## 相关概念

- [[Triton]]
- [[算子融合]]
- [[Profiling]]
- [[CODA]]
- [[CUDA Graph 执行模式]]

## 官方实现核对

- [vLLM `torch.compile` integration @ `2dfb8ba`](https://github.com/vllm-project/vllm/blob/2dfb8ba59098eb489197e1b4c643addffd51592e/docs/design/torch_compile.md)
- [vLLM `CompilationConfig` @ `2dfb8ba`](https://github.com/vllm-project/vllm/blob/2dfb8ba59098eb489197e1b4c643addffd51592e/vllm/config/compilation.py)
- [SGLang Piecewise CUDA Graph @ `04374ba`](https://github.com/sgl-project/sglang/blob/04374ba5e0b95be63990f6c64a3b210a6b70e7cc/docs/docs/advanced_features/piecewise_cuda_graph.mdx)
- [TensorRT-LLM Torch Compile & Piecewise CUDA Graph @ `aafc4eb`](https://github.com/NVIDIA/TensorRT-LLM/blob/aafc4ebf808f4a69e69c129721c918f90ce2fb1f/docs/source/features/torch_compile_and_piecewise_cuda_graph.md)

## 研究备注

- 可继续补充 `torch.compile` 与 CODA 这类 domain-specific kernel abstraction 的边界：前者偏通用图优化，后者偏固定模式的数学重写和内核特化。
