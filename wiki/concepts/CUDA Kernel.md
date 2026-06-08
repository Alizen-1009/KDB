# CUDA Kernel

## 定义

运行在 GPU 上、由大量线程并行执行的函数，是 CUDA 编程模型中的基本计算单元。

## 它解决什么问题

- 让开发者能直接控制 GPU 上的数据并行计算
- 为融合、自定义访存模式和手工优化提供最低层的实现入口

## 核心机制

- 开发者编写单线程视角下的计算逻辑
- 通过 `grid / block / thread` 索引把逻辑映射到大规模并行执行
- 更复杂的 kernel 需要显式考虑 shared memory、同步和数据布局
- Kernel launch 第三个参数控制每个 block 的 [[动态共享内存]] 大小；不传时默认是 `0`，只影响 `extern __shared__`，不影响编译期大小固定的静态 shared memory
- 真正影响性能的常见检查项往往集中在访存模式、shared memory 组织、occupancy、控制流分歧和 launch 配置
- 对很多 `memory-bound` kernel，一个很实用的排障顺序是：先看 [[内存合并访问]]，再看 [[Bank Conflict]] / [[Tiling]]，随后检查 [[Occupancy]]、[[Warp Divergence]] 和 [[Tail Effect]]
- 如果 workload 长期只落在少数固定 shape 上，还可以把 tile、线程映射、数据布局和 epilogue 固化成 shape-specialized kernel，再结合 autotune 去逼近该 shape 簇的局部最优
- 在工程面试和手写题语境里，很多 kernel 又会进一步收束成少量可复用模板，如 [[Warp Shuffle Reduce]]、[[Block Reduce]] 和 [[Grid-stride Loop]]
- [[CODA]] 把这种 `epilogue` 视角推进到 Transformer block 层面：固定优化过的 GEMM mainloop，把 RMSNorm、SwiGLU、RoPE、残差等局部操作重写进 GEMM epilogue，减少中间张量落 HBM。

## 常见算子分类

- `Pointwise / Elementwise`：每个元素独立，如 add、mul、SiLU、bias、mask；通常 memory-bound，重点是 coalesced load/store、vectorized load、融合和减少 launch。
- `Reduction`：多元素聚合成少量结果，如 sum、max、norm、softmax 的行归约；重点是 warp/block 归约、shared memory、数值稳定和跨 block 二阶段归约。
- `Scan / Prefix`：前缀和、前缀最大等带顺序依赖的并行扫描；重点是分层 scan、warp primitive 和跨 block 合并。
- `GEMM / Matmul`：矩阵乘和线性层；通常 compute-bound 或复用受限，重点是 [[Tiling]]、Tensor Core、layout、对齐、pipeline 和 epilogue fusion；CODA 这类方法进一步把部分 Transformer 小算子表达成 `GEMM + epilogue` 程序。
- `Attention`：QK、softmax、PV 的复合数据流；重点是 tiling、[[Online Softmax]]、避免 score/probability 矩阵落 HBM，以及 varlen/ragged batch 的调度。
- `MLA backend`：以 [[FlashMLA]] 为代表，重点是 latent KV cache layout、paged cache metadata、Split-KV、变长序列调度和 Hopper/SM90 特化能力。
- `Gather / Scatter / Indexing`：间接寻址、KV block table、embedding lookup、[[MoE]] token dispatch；常受访存随机性和负载不均限制，重点是数据重排、coalescing、分桶和减少原子冲突。
- `Stencil / Window`：卷积、滑窗、局部滤波；重点是 tile + halo、shared memory 复用和边界处理。
- `Sort / TopK / Sampling`：带比较、选择和不规则控制流；重点是分块选择、warp divergence 控制、随机数/采样路径和小 batch latency。
- `Communication-adjacent`：all-reduce 前后的 pack/unpack、quant/dequant、reduce-scatter buffer 准备；重点是通信 overlap、buffer layout 和减少碎片 kernel。

## 通用设计流程

- 先定义 shape、dtype、layout、batch 分布和目标指标，区分优化 TTFT、ITL、吞吐还是显存。
- 画出数据流，列出每个输入/输出和中间张量的 bytes、FLOPs、复用次数，先用 [[Roofline 模型]] 判断偏 memory-bound 还是 compute-bound。
- 选择并行粒度：一个 thread / warp / block / CTA 负责多少元素、行、tile、head 或 sequence。
- 决定数据驻留位置：哪些值留在寄存器，哪些进 shared memory，哪些必须回 HBM。
- 设计 tile、向量化和内存布局，优先保证 global memory coalescing，再处理 [[Bank Conflict]]。
- 对 reduction 类路径优先使用 warp-level primitive，再上 block reduce，必要时二阶段 reduce。
- 控制资源占用：检查寄存器、shared memory、block size、[[Occupancy]] 和 [[Tail Effect]]，避免为了局部复用压垮并发。
- 根据数据依赖决定是否 [[算子融合]]，重点融合大中间张量和 pointwise epilogue，不机械制造超大 kernel。
- 用 reference 实现保证正确性，再 benchmark 固定 shape，最后用 [[Profiling]] 定位 bottleneck 并做 autotune。

## 关键权衡

- 控制力最强，适合极致优化
- 实现复杂度高，调试、移植和维护成本也更高
- 自定义 kernel 想稳定超越官方库，通常依赖更窄的 workload 假设；shape 一旦变化，历史最优配置很可能失效

## 相关实体

- [[../entities/Stanford CS336]]

## 相关来源

- [[../sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]
- [[../sources/你一定要知道：CUDA优化六要]]
- [[../sources/CUDA优化维度框架]]
- [[../sources/秋招CUDA手撕题复盘（附代码）]]
- [[../sources/CUDA内存层次与动态共享内存问答整理]]
- [[../sources/陈巍：DeepSeek 开源Day（1）-FlashMLA 深入分析（收录于：DeepSeek技术详解系列）]]
- [[../sources/还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速]]

## 相关概念

- [[GPU执行模型]]
- [[CUDA内存层次]]
- [[动态共享内存]]
- [[Profiling]]
- [[Triton]]
- [[Bank Conflict]]
- [[Occupancy]]
- [[Warp Divergence]]
- [[Warp Shuffle Reduce]]
- [[Block Reduce]]
- [[Grid-stride Loop]]
- [[Roofline 模型]]
- [[算子融合]]
- [[Tiling]]
- [[MoE]]
- [[FlashMLA]]
- [[CODA]]

## 研究备注

- 常见的超越官方库路径不是“更底层”本身，而是利用固定 shape、固定 layout 和可融合算子链，把通用问题改造成特化问题
- 后续可补 CUDA C++、CUTLASS、PyTorch extension 与自定义 op 之间的关系，以及常见 kernel 优化 checklist 的 profiler 对应信号
- 现有来源已经覆盖“性能原则”和“面试模板”两条线；后续可以继续补 `compute-bound kernel`，把 `GEMM / Tensor Core / FlashAttention` 接进来
- FlashMLA 相关性能数字和 SM90 细节应回到官方实现与 benchmark 核实，不宜脱离硬件配置和具体 commit 独立引用。
- CODA 相关论文和 repo 尚未 ingest；当前仅根据二手文章记录其 epilogue 编程抽象和待核实 benchmark。
