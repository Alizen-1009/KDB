# 多卡 GPU 监控与 SM 执行模型面试整理

## 总览

这组题的主线是：不要把 GPU 性能理解成一个单一的“利用率”。多卡 LLM 推理要分层看业务延迟、作业调度、设备管线和通信链路；单卡 kernel 性能则要回到 `thread / warp / block / SM` 的执行模型，理解 `SM Active`、`SM Issue`、`Occupancy`、`Tensor Active`、`FP32/BF16 pipe active` 分别在回答什么问题。

一句话版本：

> `nvidia-smi` 只能告诉我们驱动/进程视角的资源与健康状态；真正判断 LLM 推理有没有跑在预期高效路径上，需要把 prefill/decode 阶段拆开，结合 `SM Active / SM Issue / Occupancy / Tensor Active / FP32 或 BF16 pipe active / DRAM Bandwidth / NCCL 或 NVLink` 一起看。

相关概念：[[../../wiki/concepts/Profiling|Profiling]]、[[../../wiki/concepts/GPU执行模型|GPU执行模型]]、[[../../wiki/concepts/Occupancy|Occupancy]]、[[../../wiki/concepts/混合精度训练与推理|混合精度训练与推理]]、[[../../wiki/concepts/KV Cache|KV Cache]]、[[../../wiki/concepts/FlashAttention|FlashAttention]]。

## 问题一：多卡集群 GPU 指标监测怎么设计？

### 1. 监控设计要分四层

多卡集群监控不能只看 `GPU util` 或 `nvidia-smi`，而要把指标分成业务层、作业层、设备层和通信层。

| 层级 | 关注指标 | 回答的问题 |
|---|---|---|
| 业务层 | QPS、tokens/s、TTFT、ITL、P95/P99、队列长度、prefill/decode 占比 | 用户是否真的变快，延迟和吞吐是否达标 |
| 作业层 | job_id、model、rank、GPU UUID、TP/PP/DP rank、batch size、seq len、KV cache 使用率、OOM、重启次数 | 哪个作业、哪个 rank、哪个模型实例出了问题 |
| 设备层 | SM Active、SM Issue、Tensor Active、FP32/BF16 pipe active、HBM read/write、显存、温度、功耗、时钟、throttle、ECC、Xid | GPU 本身是算力不足、访存受限、掉频，还是健康异常 |
| 通信层 | NVLink/PCIe/IB 带宽、NCCL all-reduce/all-gather/all-to-all 延迟、overlap、rank skew | 多卡是否被通信、拓扑或负载不均拖住 |

落地方式可以分三类：

- 常驻监控：`DCGM Exporter / NVML / nvidia-smi` + Prometheus + Grafana。
- 作业埋点：推理引擎上报 batch、KV cache、请求延迟、prefill/decode token 数、rank 信息。
- 深度诊断：`Nsight Systems` 看全局 timeline，`Nsight Compute` 看单 kernel 的 occupancy、pipe utilization、stall、L2/DRAM。

多卡场景尤其要看 rank 间差异。比如 TP 里某张卡 `SM Active` 或 `Tensor Active` 明显低，可能不是这张卡 kernel 写得差，而是它在等通信、等其他 rank、拓扑不匹配、CPU NUMA 亲和性差，或者该 rank 被分到了更小的 shape。

## 2. `nvidia-smi` 看到的显存占用真实吗？

结论是：真实，但不等于“模型当前活跃张量占用”。

`nvidia-smi` 看到的是驱动/NVML 视角下进程 GPU context 使用的 framebuffer memory。它适合回答：

- 哪个进程占着 GPU
- 显存总量和剩余量大概是多少
- 是否存在异常进程、残留进程、MIG/健康问题

但它不适合单独回答：

- 当前模型权重、KV cache、临时激活各占多少
- 当前 tensor 是否真的还活着
- 为什么 PyTorch 明明释放了 tensor，显存看起来没降

原因包括：

- PyTorch/CUDA 有 caching allocator，释放的 tensor memory 可能仍被 allocator 缓存。
- `nvidia-smi` 中还包含 CUDA context、cuBLAS/cuDNN workspace、CUDA Graph pool、fragmentation、KV cache 预留池。
- `torch.cuda.memory_allocated()` 更接近 tensor 实际占用。
- `torch.cuda.memory_reserved()` 更接近 PyTorch allocator 从 CUDA driver 预留的总量。
- reserved 很高不一定坏，可能是缓存；但 reserved/fragmentation 也可能导致大块分配失败。

所以面试里可以说：

> `nvidia-smi` 是进程/驱动视角的显存账本，不是框架内部 tensor 生命周期账本。排查 OOM 或显存泄漏时，我会同时看 `nvidia-smi`、框架的 `allocated/reserved/max_memory`、KV cache 预算和 allocator snapshot。

## 3. 为什么看 FP32 pipeline active 和 BF16 pipeline active？

因为 LLM 推理性能很大程度取决于有没有走到预期的低精度高吞吐路径。

如果模型声称是 BF16 推理，理想情况是：

- 大 GEMM、QKV projection、MLP、prefill attention 主要命中 Tensor Core/BF16 MMA 路径。
- `Tensor Active` 或 BF16 相关 pipe 指标较高。
- FP32 pipe 主要出现在 LayerNorm、RMSNorm、Softmax、采样、少量 reduction 或累加上。

如果观察到：

- `FP32 pipe active` 异常高
- `Tensor Active` 很低
- kernel name 显示没有命中高效 GEMM/attention kernel

就要怀疑发生了退化：dtype、shape、layout、backend 或 attention mask 不匹配，导致没有走 Tensor Core，而是退回普通 FP32/SIMT CUDA core 路径。

### 4. Tensor Core Active 和 CUDA Core Active 怎么解释？

`Tensor Core Active` 表示 SM 上 tensor pipe 在发射/执行 MMA 类矩阵乘加指令的周期比例。它对应 FP16/BF16/TF32/FP8/INT8 这类矩阵乘加高吞吐路径。LLM 中的 Linear、QKV projection、MLP GEMM、prefill attention 的大矩阵计算，理想上应该主要吃 Tensor Core。

`CUDA Core Active` 是更口语化的说法。profiling 里通常看的是 FP32/FMA/ALU/SFU/LSU 等普通执行管线是否活跃。它负责普通 FP32 算术、elementwise、地址计算、部分 reduction、类型转换、未融合小算子，以及没命中 Tensor Core 的 fallback matmul。

严格说 profiler 里没有一个统一的、跨架构完全固定的“CUDA Core Active”指标，常用 `FP32 pipe active / FMA pipe utilization / SM Issue` 近似判断普通 CUDA core 路径是否繁忙。

### 5. 如何判断 LLM 推理是否因精度或维度不匹配而退化？

要先拆 prefill 和 decode。

`Prefill` 通常是长 prompt 的并行编码，大矩阵多，更容易 compute-bound，也更应该看到 Tensor Core 活跃。

`Decode` 每次生成一个 token，batch 如果不够大，很多计算会变成小 GEMM/GEMV，同时还要大量读取 [[../../wiki/concepts/KV Cache|KV Cache]]，因此更容易 memory-bound。

常见判断表：

| 指标现象 | 可能原因 | 进一步确认 |
|---|---|---|
| Tensor Active 低，FP32 pipe 高 | BF16/FP16 没命中 Tensor Core，退回 FP32/SIMT | 看 kernel name、SASS 是否有 HMMA/MMA，检查 autocast、权重/激活 dtype |
| Tensor Active 低，DRAM 带宽高 | decode memory-bound，KV cache 读写主导 | 分 prefill/decode 看，检查 batch、KV layout、PagedAttention |
| Tensor Active 低，SM Active 低 | batch 太小、kernel 太碎、CPU launch 断流 | 用 Nsight Systems 看 gap、CUDA Graph、continuous batching |
| Tensor Active 有但不高，tail/occupancy 差 | shape 太小或维度不对齐 | 检查 M/N/K、head_dim、hidden size 是否满足 kernel 对齐偏好 |
| rank 间指标差异大 | 多卡负载不均或通信等待 | 看 NCCL timeline、NVLink/IB 带宽、每 rank token/batch |

常见退化点包括：

- `head_dim`、hidden size、intermediate size、GEMM 的 M/N/K 不满足 Tensor Core kernel 的对齐偏好。
- attention backend 因 mask、head_dim、GQA/MQA、dtype、layout 不支持，退回 unfused 或普通 PyTorch kernel。
- 权重是 BF16，但中间频繁 cast 到 FP32，或者 dequant/cast kernel 太多。
- decode batch 太小，矩阵从 GEMM 退化成很多小 GEMV，Tensor Core 吃不饱。
- 量化模型里 group size/layout 不合适，dequant 开销抵消低精度收益。

## 问题二：详细讲讲 GPU 里的 SM 和线程束

### 1. SM 是什么？

SM，全称 `Streaming Multiprocessor`，中文常叫流多处理器。可以把它理解成 NVIDIA GPU 里真正执行 CUDA kernel 的基本计算岛。

一个 GPU 有很多 SM。每个 SM 里大致包含：

- warp scheduler：选择哪个 warp 发射下一条指令。
- dispatch unit：把指令发到对应执行管线。
- register file：线程私有寄存器。
- FP32/FMA pipe：普通 FP32、整数、elementwise 等。
- Tensor Core/tensor pipe：MMA 矩阵乘加指令，负责 FP16/BF16/TF32/FP8/INT8 高吞吐计算。
- LD/ST unit：load/store 访存。
- SFU：特殊函数，比如 exp、sqrt、sin/cos 等。
- shared memory / L1 cache：SM 内片上存储。

CUDA 代码的层级是：

```text
grid
  -> block
      -> warp
          -> thread
```

一个 block 会被调度到某个 SM 上执行。block 不能跨 SM。block 内线程按 32 个一组组成 warp。SM 真正调度执行的基本单位不是单个 thread，而是 warp。

### 2. Warp 是什么？

warp 是 32 个线程组成的执行单位。NVIDIA GPU 是 SIMT 模型：`Single Instruction, Multiple Threads`。意思是同一个 warp 里的 32 个线程通常一起执行同一条指令，只是每个 thread 处理不同数据。

例如向量加法：

```cuda
c[i] = a[i] + b[i];
```

一个 warp 里的 32 个线程可能分别处理连续 32 个元素。这样访存也容易合并成 coalesced memory access。

warp 的关键点：

- 32 个线程一起被调度。
- warp 内线程最好走相同控制流。
- warp 内线程最好访问连续或规则地址。
- 如果同一个 warp 内线程走不同 `if/else`，就会出现 [[../../wiki/concepts/Warp Divergence|Warp Divergence]]。
- divergence 时，硬件往往要串行执行多个分支，没走当前分支的线程被 mask 掉。

所以 CUDA kernel 优化的一个核心目标是：让 warp 内控制流和访存尽量规则。

### 3. SM 和 warp 怎么协作？

一个 SM 上会同时驻留很多 warp。这里的“驻留”不是所有 warp 同时执行，而是这些 warp 的上下文已经在 SM 上，scheduler 可以随时切换。

为什么要很多 warp？因为 GPU 单个访存延迟很高。一个 warp 发出 global memory load 后可能要等很多 cycle。SM 不会空等，它会切到另一个 ready warp 继续执行。

大概流程是：

```text
很多 warp 驻留在 SM 上
    -> warp scheduler 每个周期找 eligible warp
        -> 发射指令
            -> 指令进入 FP32 pipe / Tensor pipe / LDST pipe 等
```

所以性能好坏要看四层：

```text
有没有 warp 驻留：Occupancy / SM Active
有没有 ready warp：eligible warps
有没有发射指令：SM Issue
发到了什么管线：Tensor Active / FP32 pipe / LSU / DRAM
```

## 4. SM、warp 和 LLM 的关系

LLM 主要算子包括：

- Linear / MLP / QKV projection：大 GEMM。
- Attention：QK、PV、softmax、mask。
- LayerNorm / RMSNorm：reduce + elementwise。
- KV Cache：读写历史 K/V。
- Sampling：top-k/top-p、softmax、随机采样。
- 多卡通信：all-reduce、all-gather、all-to-all。

这些都会落到 CUDA kernel 上，最终由 SM 和 warp 执行。

`Prefill` 阶段通常 prompt 很长，矩阵比较大：

```text
[B * S, hidden] x [hidden, intermediate]
```

这类大 GEMM 容易产生大量 block/warp，SM Active 高，Tensor Active 高，也更容易打满 Tensor Core。

`Decode` 阶段每次只生成一个 token：

```text
[B, hidden] x [hidden, intermediate]
```

如果 batch 不够大，就会变成很多小 GEMM/GEMV。问题是并行度不够，SM 可能吃不满。再加上 decode 要频繁读 KV Cache，很多时候是 memory-bound，不是 compute-bound。

因此 LLM 推理里的经验判断是：

- prefill 更 compute-bound，更看 Tensor Core 利用率。
- decode 更 memory-bound，更看 KV cache layout、HBM bandwidth、continuous batching。
- 长上下文会推高 attention/KV cache 读写压力。
- 小 batch 会让 SM Active 和 occupancy 上不去。
- shape 不对齐会导致 Tensor Core 命中差，退回普通 CUDA core/FP32 pipe。
- 变长序列/ragged batch 容易带来分支、mask 和 tail effect。

## 5. Warp 和 SM Activity / SM Active / SM Occupancy 的关系

这几个指标很容易混。

`SM Active`：表示 SM 在某段采样周期内是否至少有一个 warp in flight。粗略理解是“SM 有没有活”。但它不代表 SM 算满了。

`SM Issue`：表示 warp scheduler 是否在发射指令。这个比 SM Active 更接近“SM 有没有真正推进计算”。

`Occupancy`：表示一个 SM 上实际/理论可驻留 warp 数的比例。它回答的是“SM 上有多少 warp 可供调度器切换”。

公式直觉：

```text
occupancy = active/resident warps per SM / max warps per SM
```

如果一个 SM 理论最多 64 个 warp，当前 kernel 因为寄存器或 shared memory 限制只能驻留 32 个 warp，那么 occupancy 大概是 50%。

但是 occupancy 不等于性能。高 occupancy 只是说明有更多 warp 可以隐藏延迟，不代表每个 warp 做的事高效。

最清楚的链条是：

```text
Occupancy：SM 上能放多少 warp
SM Active：SM 上是否至少有 warp 在跑
SM Issue：scheduler 是否真的发射了指令
Pipe Active：指令发到 Tensor Core、FP32、LD/ST 等哪条管线
```

判断表：

| 指标现象 | 解释 |
|---|---|
| SM Active 低 | GPU 没被喂饱，可能 batch 太小、kernel 太碎、CPU launch 断流、NCCL 等待 |
| SM Active 高，SM Issue 低 | 有 warp 驻留但大量 stall，可能等 HBM、等依赖、等 barrier、等同步 |
| Occupancy 低，性能差 | 可能寄存器/shared memory 太重，warp 不够隐藏延迟 |
| Occupancy 低，但性能好 | 可能 tile 大、数据复用好、Tensor Core 吃得饱，不必强行追满 occupancy |
| Occupancy 高，但 Tensor Active 低 | warp 很多，但没走高效 Tensor Core 路径，可能 dtype/shape/kernel 不对 |
| SM Active 高，DRAM 高，Tensor Active 低 | 多半 memory-bound，例如 decode 读 KV Cache |

## 面试总结版

可以这样收束：

> SM 是 GPU 的基本执行单元，block 被调度到 SM 上，block 内线程按 32 个组成 warp。SM 的 warp scheduler 每周期从 resident warp 中选择 ready warp 发射指令。Occupancy 描述 SM 上可驻留 warp 的比例，主要影响隐藏延迟的能力；SM Active 描述 SM 是否有 warp 在运行；SM Issue 描述调度器是否真的发射指令；Tensor Active/FP32 pipe active 则说明指令实际打到了哪类执行管线。  
>  
> 对 LLM 来说，prefill 的大 GEMM/attention 通常应该有较高 SM Active 和 Tensor Active；decode 因为单 token、小 batch 和 KV cache 读写，常常 SM 吃不满或被 HBM 带宽限制。分析性能时不能只看 GPU util 或 occupancy，要把 prefill/decode 拆开，结合 SM Active、SM Issue、Tensor Active、DRAM Bandwidth 和 kernel shape 一起判断。

## 相关页面

- [[../../wiki/concepts/Profiling|Profiling]]
- [[../../wiki/concepts/GPU执行模型|GPU执行模型]]
- [[../../wiki/concepts/Occupancy|Occupancy]]
- [[../../wiki/concepts/Warp Divergence|Warp Divergence]]
- [[../../wiki/concepts/混合精度训练与推理|混合精度训练与推理]]
- [[../../wiki/concepts/KV Cache|KV Cache]]
- [[../../wiki/concepts/FlashAttention|FlashAttention]]
- [[../../wiki/concepts/Continuous Batching|Continuous Batching]]
