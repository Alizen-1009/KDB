# 算子融合与 Torch Compile、CUDA Graph 的分层关系

## 背景

讨论“要不要做算子融合”时，不能只看两段计算能否写进同一个 kernel。真正要判断的是：当前瓶颈来自中间张量的 HBM 往返、kernel launch、计算吞吐、动态调度，还是资源占用。`torch.compile` 和 CUDA Graph 确实都要考虑，但它们处在不同层次。

## 核心观点

1. **算子融合优化 GPU 内部的数据流**：减少 kernel 数量以及中间张量的写回/重读。
2. **`torch.compile` 是产生融合的一种上层机制**：捕获 PyTorch 图，做图优化、分区和代码生成；它不是所有融合的同义词，也不保证自动得到最优融合。
3. **CUDA Graph 优化 CPU 到 GPU 的提交路径**：capture/replay 一串已经确定的 kernel、memcpy 等工作，主要减少逐 kernel launch 的 CPU 开销；它本身不会把多个 kernel 合成一个 kernel，也不会消除 kernel 之间的中间张量。
4. 三者可以叠加：先由 `torch.compile` 或手写 kernel 改变 kernel 序列，再由 CUDA Graph capture/replay 编译后的稳定执行序列。

可用下面的层次理解：

```text
PyTorch eager operators
        │
        ├─ torch.compile：捕获/优化计算图，可能生成 fused kernel
        │
        ├─ 手写 Triton/CUDA/CUTLASS：显式实现特化 fused kernel
        ▼
GPU kernel / library-op launch sequence
        │
        └─ CUDA Graph：捕获并重放这串 launch，减少 CPU 提交开销
```

## 机制拆解

### 1. 算子融合解决什么

假设执行：

```text
x -> bias -> activation -> residual add -> y
```

如果每一步都是独立 kernel，中间结果可能反复落到 HBM。融合后可让中间值留在寄存器或 shared memory，最后只写一次 `y`。收益可能同时包括：

- 减少中间张量 HBM traffic；
- 减少 kernel launch 次数；
- 在一次加载后完成更多计算，提高 arithmetic intensity；
- 把 bias、activation、quant/dequant 等并入 GEMM epilogue。

但过度融合可能增加寄存器和 shared memory 压力，降低 [[../../wiki/concepts/Occupancy|Occupancy]]，触发 spilling，并迫使计算形态不同的阶段共用次优 tile 或线程映射。

### 2. `torch.compile` 解决什么

[[../../wiki/concepts/Torch Compile|Torch Compile]] 是自动优化入口。它适合先作为低改动基线，观察编译器能否自动完成 pointwise chain、epilogue 等常见融合。

工程上还需检查：

- 是否发生 graph break；
- dynamic shape 是否引入 guards、重新编译或大量变体；
- 自定义 op 是否成为阻止跨边界优化的黑盒；
- 编译时间和缓存成本是否能被长期运行摊销；
- 生成的 kernel 是否真的减少了 HBM traffic，而不只是减少了图节点数；
- 自动生成的 tile、寄存器使用和 shape specialization 是否适合目标 workload。

当编译器看不到代数重写机会、需要特殊 layout、复杂 reduction、跨 GEMM epilogue 特化或极窄 shape 优化时，仍可能需要 Triton、CuTe/CUTLASS 或手写 CUDA kernel。

### 3. CUDA Graph 解决什么

[[../../wiki/concepts/CUDA Graph 执行模式|CUDA Graph]] 主要解决 launch-bound：把稳定的 GPU 工作序列 capture 后 replay，避免 CPU 每次逐个提交 kernel。

它与融合的差别可用一个反例说明：如果原执行序列包含 20 个 kernel，CUDA Graph 可以更便宜地重放这 20 个 kernel，但它们仍然是 20 个 kernel，kernel 间的中间张量通常仍然存在。融合也许能把它们变成 5 个 kernel；随后 CUDA Graph 再低成本重放这 5 个 kernel。

使用 CUDA Graph 时还要检查：

- shape、控制流和地址是否能稳定到可 capture/replay 的形式；
- capture 前是否已经完成编译、warm-up、lazy initialization 与必要的内存分配；
- 动态 batching、ragged attention、LoRA、投机解码或 collective 是否与目标 runtime 的 capture 路径兼容；
- 为不同 capture size 保存静态 buffer/graph 带来的额外显存；
- 真实流量命中已 capture shape 的比例，而不只看理想固定 shape benchmark。

## 对比分析

| 维度 | 手工/专用算子融合 | `torch.compile` | CUDA Graph |
|---|---|---|---|
| 主要优化对象 | kernel 内数据流与融合边界 | 高层计算图、分区与代码生成 | 一串 GPU work 的提交与重放 |
| 是否减少中间 HBM 往返 | 通常是核心目标 | 可能，通过自动融合实现 | 通常不会 |
| 是否减少 kernel 数量 | 是 | 可能 | 否，只减少逐次提交成本 |
| 是否减少 CPU launch overhead | 是，因 kernel 变少 | 可能，因 kernel 变少 | 是，核心目标 |
| 对动态 shape 的容忍 | 取决于实现 | 可能产生 guards/重编译 | 通常要求更强的静态化或 size buckets |
| 主要风险 | 资源压力、维护成本、形状过特化 | graph break、编译成本、自动优化不稳定 | capture 兼容性、额外显存、低命中率 |

## 算子融合需要检查的维度

### A. 正确性

- 广播、layout、stride、alias/in-place 语义；
- reduction 顺序、数值稳定性、精度和 dtype accumulation；
- 训练时 autograd、保存中间量和反向重计算；
- RNG、dropout 以及确定性要求。

### B. Workload 与目标

- 固定 shape 还是 dynamic/ragged shape；
- prefill、decode、训练还是离线 batch；
- 优化 TTFT、TPOT、吞吐还是显存；
- dtype、量化格式、batch/context 分布和目标 GPU。

### C. 性能瓶颈

- 用 [[../../wiki/concepts/Roofline 模型|Roofline]] 判断 memory-bound 或 compute-bound；
- 判断是否 launch-bound，尤其是小 batch decode 和许多短 kernel；
- 统计最大中间张量、重复 HBM 读写以及可复用数据；
- 检查 Tensor Core、SFU、HBM、L2、互联或同步谁在限制端到端性能。

### D. Kernel 资源与映射

- global memory coalescing、向量化和对齐；
- tile、线程/warp/block 映射；
- 寄存器、shared memory、occupancy 和 spilling；
- reduction/global synchronization 边界；
- warp divergence、tail effect、不同阶段是否需要不同 launch geometry。

### E. 编译与执行系统

- `torch.compile` 是否已经自动融合热点；
- 手写 custom op 会不会截断编译器可见的数据流；
- 编译、warm-up、cache 与重新编译成本；
- CUDA Graph capture safety、capture-size 覆盖和 replay 命中率；
- serving scheduler 是否不断改变 batch 形态，使理论优化无法命中。

## 工程建议

推荐按下面的顺序做，而不是一开始就写巨型 fused kernel：

1. **建立 eager/reference baseline**，先保证正确性。
2. **打开 `torch.compile` 作为自动优化基线**，检查 graph break、生成 kernel 数与编译开销。
3. **端到端 profile**，区分 HBM-bound、compute-bound、launch-bound 和调度瓶颈。
4. **只手工融合仍然显著的热点**，优先处理大中间张量和 pointwise/GEMM epilogue。
5. **若剩余瓶颈主要是 CPU launch，且执行形态稳定，再引入 CUDA Graph**。
6. **做消融实验**：至少比较 eager、compile、graph、compile+graph（在 runtime 支持的前提下），同时记录延迟、吞吐、显存、编译/capture 时间和 shape 覆盖率。

一句话回答：**是的，需要考虑 `torch.compile` 和 CUDA Graph；但不是把它们当作两个额外“融合选项”。`torch.compile` 决定编译器能否自动改写和融合图，CUDA Graph 决定生成后的 kernel 序列能否低开销重放。先判断瓶颈，再决定分别在哪一层优化。**

## 待核实

- `torch.compile` 的具体融合、graph break 与 dynamic-shape 行为依赖 PyTorch 版本和 backend。
- CUDA Graph 对 attention、collective、LoRA、speculative decoding 等路径的支持依赖具体 runtime 与版本。
- 不存在通用的融合加速百分比；结论必须绑定模型、shape、dtype、硬件和流量分布做 benchmark。
