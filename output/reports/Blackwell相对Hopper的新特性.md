# Blackwell 相对 Hopper 的新特性

## 范围

本文主要讨论数据中心 Blackwell `SM100/B100/B200/GB200` 的 kernel 编程能力。消费级 RTX 50、工作站与其他 Blackwell compute capability 的具体指令和资源不能直接按 SM100 推广。

## 核心观点

Blackwell 的变化不只是“增加 FP4”。其 kernel 主线是：

```text
TMA：GMEM → SMEM
        ↓
tcgen05.mma：SMEM operands → Tensor Core
        ↓
TMEM：保存 accumulator
        ↓
CUDA Core epilogue：TMEM → RF/SMEM → GMEM
```

再由 2-SM cooperative MMA 扩大单次计算 tile，由 Cluster Launch Control（CLC）给 persistent workers 动态分配 work tiles。

## SM100 Kernel 侧真正重要的新能力

### 1. Tensor Memory（TMEM）

[[../../wiki/concepts/Tensor Memory|TMEM]] 是 Blackwell 新增的 Tensor Core 专用片上存储。SM100 每个 SM 的来源口径为 256 KB，主要保存 `tcgen05.mma` accumulator，缓解 Hopper WGMMA 把大块 accumulator 放在普通 register file 导致的寄存器压力。

关键边界：

- 需要显式 alloc/dealloc；
- 具有协作式 row/column layout；
- 通过 `tcgen05.ld/st/cp` 等受限路径移动数据；
- 输出通常仍走 `TMEM → RF/SMEM → GMEM`，不能把 TMEM 当作通用 cache 或直接任意写 HBM。

### 2. `tcgen05.mma` / 第五代 Tensor Core

`tcgen05.mma` 取代 Hopper `wgmma.mma_async` 成为 SM100 主要异步 MMA 路径：

| Hopper | Blackwell SM100 |
| --- | --- |
| `wgmma` warpgroup issue | `tcgen05.mma` 可由一个 elected thread issue |
| accumulator 在 registers | accumulator 在 TMEM |
| FP8 是主要新增窄精度 | 增加 FP6、FP4 和 block-scaled variants |
| warpgroup commit/wait | TMEM 生命周期、completion 与 fence 管理 |

单线程 issue 不代表只有一个线程做矩阵计算；它只是由一个控制线程提交 Tensor Core 工作，硬件 Tensor Core 异步完成运算。

### 3. 原生 FP4/FP6 与 Block Scaling

第五代 Tensor Core 支持 FP4、FP6，并把 block scale 纳入 MMA 数据路径。重点包括：

- `FP4 E2M1`；
- `NVFP4`：每 16 个 FP4 元素共享更高精度 scale，并可叠加 tensor-level scale；
- `MXFP4/MXFP8` 等 microscaling 路径；
- `tcgen05` 的 block-scaled MMA variant 直接消费数据和 scale descriptor。

这与旧 GPU 的 INT4/FP4 weight-only 量化不同：旧路径往往先 unpack/dequantize 再执行 FP16/BF16/FP8 MMA；Blackwell 可以原生执行 FP4 Tensor Core matrix operation。

### 4. 2-SM Cooperative MMA

`tcgen05.mma.cta_group::2` 允许同一 cluster/TPC 中的两个 CTA、两个 SM 协作完成一个更大的 MMA tile。每个 SM 提供部分 operands/TMEM rows，逻辑 M tile 可相对 1-SM 模式扩大。

适合大型、compute-bound GEMM 或 attention tile，但代价包括：

- 两个 CTA 的 layout 必须匹配；
- 需要 cluster barrier/mbarrier；
- cluster 占用更强，较小或不规则 shape 未必受益；
- 2-CTA work item 必须作为整体调度，不能只重排其中一个 CTA。

### 5. Cluster Launch Control（CLC）

[[../../wiki/concepts/Cluster Launch Control|CLC]] 是 Blackwell 的 cluster-level launch cancellation/work inheritance 机制。活跃 cluster 可尝试取消一个尚未启动的 cluster，并获得它的逻辑 CTA 坐标继续处理。

它使 [[../../wiki/concepts/Persistent Kernel|Persistent Kernel]] 不必总依赖集中式 global atomic queue，尤其适合：

- grouped/variable-sized GEMM；
- 变长 attention；
- tile 成本不均、尾部明显的 workload；
- speculative workload 中取消尚未开始的任务。

CLC 不保证总是更快；response/barrier 成本与 L2 locality 变化可能抵消负载均衡收益。

### 6. 第二代 Transformer Engine

Blackwell Transformer Engine 把第五代 Tensor Core、FP4/FP6、microscaling format 和框架级 quantization recipe 结合起来，重点面向 LLM/MoE 训练与推理。

它不是简单的“把 dtype 改为 FP4”：scale 粒度、异常值处理、accumulator 精度、校准/训练 recipe 和 kernel 是否真正走原生 block-scaled MMA 都决定最终精度与性能。

## Hopper 已有、Blackwell 继续使用的能力

以下能力在 Blackwell 很重要，但不是 Blackwell 首次引入：

- TMA；
- Thread Block Cluster；
- Distributed Shared Memory；
- mbarrier / asynchronous transaction barrier；
- warp specialization；
- Persistent Kernel；
- Programmatic Dependent Launch 的基本思想。

Blackwell 的新点是把这些原语与 TMEM、`tcgen05`、2-SM MMA、CLC 和 block scaling 重新组合成新的主流水线。

## 系统与产品层面的变化

这些不是单 kernel 指令，但影响端到端 AI 系统：

- 双 reticle die 通过高带宽 chip-to-chip interconnect 组成统一 GPU；
- HBM3e、更大的产品级 memory/L2 配置，具体数字随 B100/B200/GB200 而变；
- 第五代 NVLink 与 NVLink Switch，面向 NVL72 等 rack-scale domain；
- 固定功能 Decompression Engine，可在数据搬运时处理 LZ4、Snappy、Deflate，主要服务数据库/数据管线，对 LLM kernel 是间接收益；
- 独立 RAS Engine、predictive maintenance；
- TEE-I/O、Confidential Computing 等安全能力。

## 一张记忆表

| 类别 | Blackwell 新重点 |
| --- | --- |
| 计算 | 第五代 Tensor Core、`tcgen05.mma`、FP4/FP6 |
| Accumulator | TMEM |
| 精度 | NVFP4/MXFP4、native block scaling |
| 多 SM | 2-SM cooperative MMA |
| 调度 | CLC + persistent scheduling |
| Transformer | 第二代 Transformer Engine |
| 芯片 | 双 die unified GPU |
| 互联 | 第五代 NVLink/NVLink Switch |
| 数据管线 | Decompression Engine |
| 可用性与安全 | RAS Engine、TEE-I/O |

## 工程判断

对手写 Blackwell GEMM/Attention kernel，优先理解的顺序是：

1. TMA/SMEM layout；
2. `tcgen05.mma` descriptors 与 completion；
3. TMEM allocation、layout、readout 和 epilogue；
4. mbarrier 与 warp specialization；
5. 1-SM 还是 2-SM tile；
6. static persistent 还是 CLC；
7. FP8/FP4 block scale 的 layout 与数值 recipe。

## 待核实与边界

- `tcgen05`、TMEM、CLC 和 2-SM 的具体支持必须绑定 compute capability（如 SM100/SM100a）与 PTX/CUDA/CUTLASS 版本，不能笼统外推到所有“Blackwell”产品。
- B100、B200、GB200、B300、RTX 50 等 SKU 的 SM、L2、HBM、NVLink 和精度吞吐口径不同。
- 任何性能数字必须同时给出 GPU、dtype、shape、metric、value 与代码版本；本报告不把峰值宣传数字等同于模型实测性能。

## 资料

- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
- KernelWiki：`wiki/hardware/tcgen05-mma.md`、`tmem.md`、`2sm-cooperative.md`、`clc.md`、`nvfp4.md`
- Wiki：[[../../wiki/entities/NVIDIA Blackwell|NVIDIA Blackwell]]
