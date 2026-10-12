# GEMM 常用的 GPU 优化手段有哪些？

- 整理状态：已整理
- 题目出处：用户提供的一面面经（PyTorch 图编译与推理系统专题）

## 30 秒回答

GEMM 优化先固定 `M/N/K`、dtype、layout 和目标硬件，再从数据搬运与计算映射入手：做 CTA/warp/MMA 多级 tiling，把 A/B tile 搬到 shared memory 并在寄存器复用；使用合并、对齐和向量化访问，避免 bank conflict；用 double buffering、`cp.async`/TMA 重叠搬运与 MMA；选择 Tensor Core 指令和合适的 warp 分工；最后通过 epilogue fusion、split-K 或 persistent scheduling处理特定 shape。tile 不是越大越好，还要与寄存器、shared memory、occupancy 和尾块浪费一起权衡。

## 深入解释

常见检查顺序：

1. 数学和 layout：是否能转置/重排以形成连续访问，是否命中 Tensor Core dtype/shape。
2. 多级分块：global→shared→register，增加同一 A/B tile 的复用。
3. 流水：多 stage、异步 copy、ping-pong buffer，把 load latency 藏在计算后面。
4. 线程映射：warp tile、MMA atom、K loop、边界处理和负载均衡。
5. 融合：bias、activation、quant/dequant、residual 等尽量进入 epilogue，但不能因寄存器压力导致 spill。
6. 实测：先比较 cuBLASLt/CUTLASS 等强基线，再用 profiler 看 Tensor Core、DRAM、stall、occupancy 和 wave/tail。

## 10 月 10 日补充：CUDA Core 的 SIMT GEMM 还能怎么优化？

- 追问出处：用户提供的 2026-10-10 面试回忆。
- 已提出的手段：多级 tile、double buffer、shared memory、Z-curve 重排改善 L2 局部性。
- CUDA Core 口述答案：围绕复用与指令执行做优化。CTA/warp/thread tiling 配合合并访问和无冲突 shared 布局；多个独立 accumulator、循环展开、地址计算简化和寄存器预取提高 ILP；联合调 tile、线程数和 stage 控制资源占用。输出并行度不足才考虑 Split-K，小 GEMM 汇集批次，最后融合 epilogue。

| 优化 | 收益机制 | 权衡/条件 |
|---|---|---|
| 多个独立 FMA 累加链 | 打破单 accumulator 串行依赖，同时复用寄存器操作数 | Register tile 大了可能降低 occupancy 或 spill |
| 合并、对齐、向量化 load/store | 减少 global 事务或访存/地址指令 | 向量化不自动减少字节数，需对齐和边界支持 |
| Shared padding/转置/swizzle | 让计算读取避开 bank conflict | 搬运布局和计算布局可以不同；同地址广播不算冲突 |
| K 内循环适度展开、指针递增 | 减少循环/地址开销，暴露独立指令 | 代码体积与寄存器压力 |
| Shared→register 的 fragment 预取 | 当前 FMA 与后续操作数加载重叠 | 增加寄存器占用；与 global→shared 流水分层设计 |
| Tile、线程数和 stage 联合调参 | 平衡复用、延迟隐藏与驻留 CTA | 不单独追求最大 tile 或最高 occupancy |
| Split-K | M/N 小、K 大时增加 CTA 数 | Partial sums 归约、额外流量/原子与累加顺序变化 |
| Shape 专门路径、batched/grouped GEMM | 减少尾块浪费并增加独立工作 | 调度、边界与工作规模相关 |
| Epilogue fusion 与合并写回 | 寄存器内做 bias/activation/residual，减少中间往返 | 融合资源压力；β=0 时可省旧 C 读取 |

术语与边界：

- Thread block tile 与 block tile 通常同义；SIMT 路径可按 CTA→warp→thread/register 分层。
- Z-curve 重排若作用于输出 tile 与 block ID 的映射，属于 block rasterization/swizzle，用于争取跨 CTA 的 L2 复用，不等于 warp 内合并访问，也不保证硬件严格按 Z 顺序执行。
- 分配双 buffer 不自动实现重叠，必须有实际的加载/计算调度或异步搬运及正确同步。
- Tensor Core 属于另外的计算路径；纯 CUDA Core FMA 调优不能用 MMA/WMMA 的加速替代解释。
- [[矩阵乘法的计算量、读取量与 tile 大小是什么关系|M/N/K 与 tile 的定量账本]] 是本题的独立追问，计算量、读取量与资源占用在该页推导。

依据：[NVIDIA CUTLASS 分层设计](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)、[CUTLASS Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)、[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。具体最优映射、ILP 深度、swizzle 和 Split-K 收益依赖 shape 与硬件，本次未运行 benchmark。

## 关联知识

- [[../../../../wiki/concepts/Tiling|Tiling]]
- [[../../../../wiki/concepts/算子融合|算子融合]]
- [[CUDA Thrust、CUB 与 CUTLASS 各自解决什么问题|CUTLASS]]

## 参考来源与待核实

- 本页是稳定的优化路线图，不代表任意 shape 都应使用 split-K、persistent 或相同 tile；最终以目标 GPU 和 shape sweep 为准。

## 所属题单

- [[../sets/10月10号面试遇到的问题|10月10号面试遇到的问题]]

- [[../sets/PyTorch图编译与推理系统一面|PyTorch 图编译与推理系统一面]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
