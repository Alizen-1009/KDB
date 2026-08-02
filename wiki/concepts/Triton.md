---
type: concept
topic: GPU 编程
sources: 2
updated: 2026-06-21
---

# Triton

## 定义

一种面向 GPU kernel 编写的高层编程与编译框架，允许用 Python 风格代码描述块级计算，再编译为高性能 GPU kernel。

## 它解决什么问题

- 降低手写 CUDA kernel 的门槛
- 在保持较强控制力的同时，让一部分 memory coalescing、shared memory 和代码生成细节交给编译器处理

## 核心机制

- 以 block / program 为中心组织 kernel，而不是手工管理每个线程细节
- 用 JIT 编译把 Python 风格 kernel 转成底层 GPU 实现
- 适合实现融合算子、softmax、matmul、norm 等常见高性能 kernel
- [[CODA]] 来源中提到的 [[CuTe DSL]] / CUTLASS Python DSL 与 Triton 同属降低 GPU kernel 编写门槛的方向，但 CODA 更聚焦 `GEMM + epilogue` 抽象和 Transformer 小算子的代数重写。

## 跨芯片支持

Triton 不是只能生成 CUDA。官方仓库 `main@07a1b120` 当前内置两套 backend：

- NVIDIA：`cuda` / NVPTX backend；
- AMD：`hip` / AMDGPU backend，仓库包含 MFMA/WMMA、ROCm driver 与 AMD 专属 lowering。

编译接口把目标抽象为 `GPUTarget(backend, arch, warp_size)`，例如 `cuda + compute capability` 或 `hip + gfx*`。backend 需要实现 target 检查、编译 stages、设备 dialect 与 runtime driver。

官方还支持通过 Python `triton.backends` entry points 发现 out-of-tree/downstream backend，因此 Intel XPU、Ascend、MLU、MUSA 等其它芯片原则上可以由厂商或社区实现 Triton-compatible backend；但这些不等于上游内置支持，安装方式、语言覆盖、PyTorch 集成和性能成熟度必须逐项核实。普通 `pip install triton` 不能自动在任意加速器上运行。

CPU 上的 `TRITON_INTERPRET=1` 是解释器/调试路径，可以无 GPU 检查语义，不应当作高性能 CPU backend。

### Ascend：Triton-Ascend

Ascend 可以通过独立的 [Triton-Ascend](https://github.com/triton-lang/triton-ascend) 项目运行 Triton 风格 kernel；它不是上游 `triton` wheel 内置的 CUDA/HIP backend，而是适配 CANN、TorchNPU 和 Ascend AI Core 的专门发行版。

按官方仓库 `main@70635d0d`（2026-07-30）：

- 正式版为 `triton-ascend 3.2.1`；
- 支持 Linux aarch64/x86_64 与 Atlas A2/A3/950 系列；
- 推荐 CANN 9.0.0，绑定 TorchNPU `2.7.1.post4`，Python 3.9–3.11；
- 可用 `pip install triton-ascend==3.2.1 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi` 安装。

已有 GPU Triton kernel 不能假设原样获得高性能：host 侧需把 CUDA device/runtime 改为 `torch_npu`/`npu`，kernel 侧要重新检查 AI Core/Vector Core grid、UB 容量、32/512-byte 对齐、`tl.dot` M/N/K tiling、dtype 和 `coreDim≤65535`。官方项目曾以“85% Triton Python API”作为 2025-06 里程碑，后续继续补 Scan/Sort、atomic、FP8 和 vLLM/SGLang 关键算子，因此仍应逐 op 核对 API 与性能覆盖。

官方依据：

- [Triton-Ascend 支持矩阵与安装](https://github.com/triton-lang/triton-ascend/blob/70635d0de7e80021a64c70b5e0e29cbc8b44173f/README.md#L28-L80)
- [GPU Triton 算子迁移到 Ascend 的官方指南](https://github.com/triton-lang/triton-ascend/blob/70635d0de7e80021a64c70b5e0e29cbc8b44173f/docs/en/migration_guide/migrate_from_gpu.md#L1-L46)

## 可移植性边界

- 只使用通用 `triton.language` load/store、mask、reduce、`tl.dot` 等原语的 kernel，通常最容易在 NVIDIA 与 AMD 间迁移。
- “能编译”不等于“性能可移植”：warp/wavefront 宽度、shared memory/LDS、Tensor Core/MFMA tile、寄存器压力和最佳 `num_warps/num_stages` 不同，通常要分别 autotune。
- NVIDIA TMA/WGMMA、PTX inline asm、CUDA tensor descriptor，或 AMD MFMA/WMMA 专属原语会绑定 backend，需要条件分支或独立 kernel。
- dtype、atomic、数学函数、低精度矩阵指令和框架 device tensor 支持也可能不对称。

官方依据：

- [官方构建同时安装 NVIDIA/AMD backends](https://github.com/triton-lang/triton/blob/07a1b120fc47bddb859c641772b2ae0ca0ae5fae/setup.py#L391-L397)
- [GPUTarget 与 BaseBackend 接口](https://github.com/triton-lang/triton/blob/07a1b120fc47bddb859c641772b2ae0ca0ae5fae/python/triton/backends/compiler.py#L8-L48)
- [out-of-tree backend entry-point discovery](https://github.com/triton-lang/triton/blob/07a1b120fc47bddb859c641772b2ae0ca0ae5fae/python/triton/backends/__init__.py#L38-L63)
- [NVIDIA/AMD 共用的 block-scaled matmul 教程](https://github.com/triton-lang/triton/blob/07a1b120fc47bddb859c641772b2ae0ca0ae5fae/python/tutorials/10-block-scaled-matmul.py#L1-L10)

## 关键权衡

- 比原生 CUDA 更易写、更适合快速迭代
- 仍然要求开发者理解底层数据布局、tile 设计和性能瓶颈
- 高层 DSL 不会自动消除性能问题；真正收益仍取决于数据流重排、访存减少、tile 设计和具体后端代码质量。

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]
- [[../sources/还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速]]

## 相关概念

- [[CUDA Kernel]]
- [[Torch Compile]]
- [[Tiling]]
- [[CODA]]

## 研究备注

- 后续可补 Triton 在 softmax、attention、matmul 和 fused MLP kernel 中的典型使用方式
- 可进一步比较 Triton、[[CuTe DSL]] / CUTLASS、TileLang、ThunderKittens 在抽象层级、可控性和适合的 kernel 类型上的差异。
