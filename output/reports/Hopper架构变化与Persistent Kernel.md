# Hopper 架构变化与 Persistent Kernel

## 背景

问题：[[../../wiki/entities/NVIDIA Hopper|NVIDIA Hopper]] 除了 TMA 和 WGMMA 还改了什么？[[../../wiki/concepts/Persistent Kernel|Persistent Kernel]] 是什么？

本文以 NVIDIA Hopper Tuning Guide 和 NVIDIA Hopper Architecture In-Depth 为一手资料，并结合知识库中的 Ampere→Hopper→Blackwell 流水线整理。规格数字默认指 H100/compute capability 9.0；H100 SXM、PCIe 与完整 GH100 芯片并不相同。

## 核心观点

1. Hopper 不只是增加两条“更快的矩阵和搬运指令”，而是同时升级了**精度、计算、跨 SM 协作、异步同步、片上存储、HBM 和多 GPU 互联**。
2. 对 LLM kernel 最关键的新增能力，除 TMA/WGMMA 外，是 **FP8 Transformer Engine、thread block cluster、Distributed Shared Memory、异步 transaction barrier，以及更大的 L1/SMEM**。
3. Persistent Kernel **不是 Hopper 才发明的硬件功能**。它是一种 kernel 组织方式：只启动一组长期驻留的 CTA/cluster，让它们循环处理多个逻辑 tiles。Hopper 的 TMA、WGMMA、cluster 和异步 barrier 只是让这种设计更容易做成深流水线。

## Hopper 除 TMA/WGMMA 外的主要变化

### 1. 第四代 Tensor Core、FP8 与 Transformer Engine

**已知事实：**

- Hopper Tensor Core 支持 FP8、FP16、BF16、TF32、FP64 和 INT8 MMA。
- 新增两种 FP8 输入格式：`E4M3` 偏精度，`E5M2` 偏动态范围；可使用 FP16 或 FP32 accumulator。
- NVIDIA 官方称，在同数据类型、同频率的 per-SM 口径下，Hopper Tensor Core 的 dense/sparse MMA rate 是 A100 SM 的约 2 倍。
- Transformer Engine 不只是硬件单元名称，而是硬件与软件协同：按 layer/tensor 的统计信息在 FP8 与 16-bit 计算间选择，并执行 scaling/recasting。

**工程含义：**

这使训练和推理从“固定 BF16/FP16 kernel”转向**精度调度问题**：算力、带宽和张量 footprint 会下降，但 scale 管理、异常值、accumulator 精度与数值验证成为新成本。

### 2. Thread Block Cluster

**已知事实：**

Hopper 在 `thread → block → grid` 中加入可选的 `thread block cluster` 层级。一个 cluster 内的 blocks：

- 保证并发调度在同一 GPC 内的一组 SM 上；
- 可以做 cluster 范围 barrier；
- 可以通过专用 SM-to-SM 路径协作；
- portable cluster size 上限是 8；H100 可 opt-in 到 nonportable 16，但更大 cluster 可能降低全 GPU active blocks。

**工程含义：**

一个 tile 不再只能局限于单 SM/CTA。实现可以用多个 CTA 合作完成更大的 tile，或分摊 shared-memory footprint，但必须承担 cluster-level occupancy、同步和 layout 复杂度。

### 3. Distributed Shared Memory（DSMEM）

Cluster 内的线程可以对其他 CTA 所在 SM 的 shared memory 执行 load、store 和 atomic。它提供了介于本地 SMEM 与 global/L2 之间的显式数据交换层。

典型用途包括：

- 多 CTA 共享或交换 tile；
- cluster histogram/atomic；
- 避免中间结果先写 HBM 再由另一个 SM 读取。

它不是“把所有 SMEM 自动合并成一个零成本大缓存”。Remote DSMEM 仍需考虑对齐、coalescing、stride、cluster 生命周期和 barrier；官方 tuning guide 建议尽量按 32-byte segment 对齐并避免 non-unit stride。

### 4. 异步 Transaction Barrier

Ampere 已有异步 copy，但 Hopper 增强了“线程和异步硬件单元如何确认工作完成”的同步模型。Transaction barrier 除了统计 thread arrivals，还可以统计预期完成的数据 transaction/byte count；consumer 只有在 arrivals 与 transaction 都完成后继续。

它是 TMA、cluster 数据交换和 warp-specialized pipeline 的关键胶水：producer 发起异步任务后不必由所有线程忙等，consumer 按 barrier phase 等待数据真正 ready。

### 5. 更大的 L1/Shared Memory 与明确的资源上限

H100 compute capability 9.0 的官方 tuning guide 给出：

- unified L1/texture/shared memory：256 KB/SM；
- 可配置 shared-memory capacity：最高 228 KB/SM；
- 单 block 最多可寻址 227 KB，因为 CUDA 为每 block 保留 1 KB；
- static SMEM 仍限 48 KB，超过后需要 dynamic SMEM 与显式 opt-in；
- 64 warps/SM、64K 32-bit registers/SM、255 registers/thread、32 blocks/SM。

**工程含义：**

大 SMEM 让更大 tile、更深 stage 和 warp specialization 成为可能，但不等于 occupancy 自动提高。一个 CTA 用到接近 227 KB 时通常只能占据一个 SM，正是许多 Hopper persistent kernel 的资源形态。

### 6. 通用计算和 DPX

- Compute capability 9.0 的 FP32 operations/cycle/SM 是 compute capability 8.0 的 2 倍。
- Hopper 新增 DPX 指令，融合 `add + min/max`、三输入 min/max、predicate 等动态规划内循环；主要面向 Smith-Waterman、Floyd-Warshall 等，不是 LLM GEMM 的核心路径。

### 7. Memory System

H100 产品级变化包括：

- HBM3/HBM2e，H100 HBM3 口径最高约 3 TB/s；
- H100 L2 从 A100 的 40 MB 增至 50 MB，并提高 L2→SM 带宽；
- inline compression 可为可压缩 allocation 减少实际传输量，但不会缩小虚拟/分配 footprint，也不能保证数据可压缩。

这里的容量和带宽必须绑定 H100 SXM/PCIe 与资料版本，不能写成所有 Hopper 产品的统一规格。

### 8. 多 GPU 与系统能力

- 第四代 NVLink：H100 最多 18 links，合计 900 GB/s bidirectional GPU I/O；A100 官方对比口径为 600 GB/s。
- PCIe Gen 5 x16：总计 128 GB/s，双向各 64 GB/s。
- 第三代 NVSwitch 增加 multicast 与 NVIDIA SHARP in-network reduction。
- 第二代 MIG 增加实例级资源与监控，并引入 confidential-computing/TEE 相关能力。

这些主要影响多 GPU strong scaling、隔离和部署，不直接等同于单 kernel 更快。

## 什么是 Persistent Kernel

### 普通 tiled kernel

假设 GEMM 有 1000 个 output tiles：

```text
launch 一个包含 1000 个 CTAs 的 grid
CTA 0 处理 tile 0 后退出
CTA 1 处理 tile 1 后退出
...
硬件分多波把这些 CTA 调度到 SM
```

每个 CTA 通常只处理一个逻辑 tile。后续 wave 会建立新的 CTA 实例，重新初始化该 CTA 的 pipeline、barrier 和 tile-local state。

### Persistent Kernel

Persistent 方式更像固定 worker pool：

```cuda
persistent_kernel(work_queue):
    initialize_pipeline_once()

    while (true):
        tile = acquire_next_tile()
        if (tile == END):
            break

        async_load(tile)       // Hopper 常用 TMA
        matrix_compute(tile)   // Hopper 常用 WGMMA
        epilogue_and_store(tile)
```

如果 GPU 有 120 个可用 workers，可以只让接近一波 CTA/cluster 长期驻留，由每个 worker 循环处理多个 tiles。任务可以通过以下方式分配：

- **Static persistent**：worker 按固定 stride 处理 `worker_id, worker_id + W, ...`；
- **Atomic queue**：完成后从 global counter 获取下一个 tile；
- **CLC**：Blackwell 的 [[../../wiki/concepts/Cluster Launch Control|Cluster Launch Control]] 可取消尚未启动的 cluster，并让活跃 cluster 接手其坐标。

### 它真正节省什么

- 复用 SMEM buffer、barrier、TMA descriptor 和 scheduler state；
- 摊销后续 tile 的 CTA/pipeline warm-up；
- 让 `tile N+1 load`、`tile N compute`、`tile N-1 store` 跨 tile 重叠；
- 动态取活时减少不规则 workload 的 straggler 与 [[../../wiki/concepts/Tail Effect|tail effect]]。

### 它不等于什么

- **不等于零 kernel launch**：host 仍要 launch 这一个 persistent kernel。
- **不等于全 GPU 只有一个 CTA**：一般是一组 CTA/cluster，常与可驻留 worker 数同量级。
- **不等于 CUDA Graph**：CUDA Graph 降低 host 提交和多 kernel launch 开销；persistent 把任务循环放进一个 kernel 内。
- **不等于 [[../../wiki/concepts/Megakernel|Megakernel]]**：persistent 描述 worker 生命周期；megakernel 描述跨算子/跨层的融合范围。
- **不是 Hopper 专属**：早期 GPU 也能写 persistent threads/kernel；Hopper 只是提供更合适的异步与 cluster 原语。

## 为什么 Hopper 特别适合 Persistent Kernel

```text
Load warp      : 用 TMA 预取 tile N+1
MMA warpgroup  : 用 WGMMA 计算 tile N
Epilogue warps : 处理并写回 tile N-1
Barrier        : 跟踪各 stage 的数据 ready / buffer 可复用状态
Resident CTA   : 下一轮继续处理新 tile，而不是退出
```

这形成了典型 warp specialization：少量 producer threads 管数据搬运，MMA consumer 专注 Tensor Core，其他 warps 做 epilogue。Hopper 的 cluster/DSMEM 还允许两个或多个 CTA 协同完成一个更大工作单元。

## 性能权衡

| 收益 | 代价 |
| --- | --- |
| 跨 tile overlap | scheduler、barrier 和 queue 开销 |
| 复用 pipeline/descriptor/SMEM | 长期占据 SM，影响其他 kernels |
| 动态负载均衡 | atomic 竞争或更差的 L2 locality |
| 减少 CTA warm-up/tail | 大 SMEM/register 占用限制 occupancy |
| cluster cooperative tile | cluster size 增大可能降低 active blocks |

因此 persistent 并非默认更快。规则 dense GEMM 若静态 tiled grid 已均衡且 library scheduler 已优化，额外队列可能没有收益；变长 attention、grouped GEMM、MoE 等 tile 成本不均场景通常更值得考虑。

## 待核实与边界

- WGMMA 的精确 PTX shape、register footprint 和 pipeline group 限制需绑定 CUDA/PTX/CUTLASS 版本。
- 具体 CUTLASS/FlashAttention kernel 是否使用 persistent scheduler、cluster shape 和任务队列，必须查对应 commit，不能由“运行在 Hopper”推断。
- NVIDIA 架构文章中的 speedup 是官方特定 workload/初期产品口径，不应当作任意模型的实测加速。

## 资料

- NVIDIA, [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- NVIDIA, [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- Wiki：[[../../wiki/sources/译 NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell|译 NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell]]
