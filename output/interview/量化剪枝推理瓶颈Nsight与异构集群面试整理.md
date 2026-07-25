# 量化、剪枝、推理瓶颈、Nsight 与异构集群面试整理

## 总览

这份整理覆盖几组连续面试追问：

- `10`：量化主流方案，尤其是 `GPTQ / AWQ` 的数学和工程技巧，`scaling` 与 Hessian 的原理。
- `11`：除量化外的显存/成本优化：稀疏、剪枝、蒸馏，重点解释结构化剪枝和非结构化剪枝。
- `12`：LLM 推理瓶颈判断、roofline、prefill/decode 的 GEMM/GEMV 与计算/访存属性、FlashAttention、Marlin、PD 解耦/分离。
- `13`：A100/H20 异构多卡集群的算力和通信比较、Hopper 对 FP8 等格式的支持、单机单卡/单机多卡/多机多卡差异。
- 补充：DCGM/NVML 不够时，如何用 Nsight Systems 和 Nsight Compute 做多卡性能分析。

核心面试主线：

> 大模型推理优化不是单点技术，而是围绕“权重、激活、KV Cache、kernel、调度、通信、硬件精度路径”共同做账本。量化主要减少权重/带宽，剪枝和稀疏减少有效计算或参数，PD 分离解决 prefill/decode 资源争用，Nsight 则用于把“慢”拆成 CPU 调度、通信、访存、kernel 或精度路径退化。

相关 KDB 页面：[[../../wiki/concepts/Profiling|Profiling]]、[[../../wiki/concepts/Roofline 模型|Roofline 模型]]、[[../../wiki/concepts/KV Cache|KV Cache]]、[[../../wiki/concepts/FlashAttention|FlashAttention]]、[[../../wiki/concepts/PD分离|PD分离]]、[[../../wiki/concepts/混合精度训练与推理|混合精度训练与推理]]、[[../../wiki/concepts/GPU执行模型|GPU执行模型]]。

---

## 10. 量化主流方案，GPTQ 和 AWQ 的数学/工程技巧

### 10.1 主流量化方案怎么分类

可以从四个维度分类：

| 分类维度 | 常见方案 | 重点 |
|---|---|---|
| 训练流程 | PTQ、QAT、量化微调 | PTQ 成本低，QAT/微调质量更稳但成本高 |
| 量化对象 | weight-only、weight+activation、KV cache quantization | LLM 推理里 W4A16、W8A8、FP8、KV FP8 都常见 |
| 数值格式 | INT8、INT4、FP8、NF4、MXFP、NVFP 等 | INT 更依赖 scale/zero point，FP8 更吃硬件支持和 scale 策略 |
| 粒度 | per-tensor、per-channel、per-group、per-token | 粒度越细越准，但元数据和 kernel 复杂度越高 |

常见方法：

- `LLM.int8()`：把大部分矩阵乘放到 INT8，但对 outlier feature 用混合精度路径保护。
- `SmoothQuant`：把 activation outlier 的量化难度迁移到 weight 侧，做 W8A8 更友好。
- `GPTQ`：后训练 weight-only 量化，用 Hessian/二阶信息做逐层误差补偿，常见 W4A16。
- `AWQ`：activation-aware weight-only 量化，用 activation 统计识别重要通道，通过 scaling 保护 salient weights，常见 W4A16。
- `FP8`：Hopper/Transformer Engine 之后很重要，训练和推理都可用，但 scale、amax history、校准和 kernel 支持很关键。
- `KV cache quantization`：压 KV cache 显存和带宽，常用于长上下文推理，但要注意长上下文质量和误差累积。

一句话：

> LLM 推理里最常见的是 weight-only 低比特量化，因为 decode 阶段经常 memory-bound，权重读带宽是大头之一；如果要更进一步吃硬件吞吐，则会看 W8A8、FP8 或专门的 W4A16 kernel。

### 10.2 scaling 的基本原理

量化可以理解成：

```text
q = clamp(round(x / s) + z)
x_hat = s * (q - z)
```

其中：

- `s` 是 scale，决定实数区间怎么映射到整数格点。
- `z` 是 zero point，非对称量化时用来表示 0。
- symmetric quantization 常令 `z=0`。

scale 的选择很关键：

- scale 太大：格点很粗，小值误差大。
- scale 太小：大值溢出或被 clip。
- per-tensor scale 简单但容易被 outlier 绑架。
- per-channel/per-group scale 更准，但元数据更多，kernel 更复杂。

LLM 里 outlier 很常见，所以 scaling 通常不是一个纯工程细节，而是决定量化误差分布的核心。

### 10.3 GPTQ：Hessian 二阶误差补偿

GPTQ 目标是做后训练、逐层、weight-only 量化。它不是只看权重本身的 MSE，而是看量化后这一层输出误差：

```text
min || W X - Q(W) X ||^2
```

其中：

- `W` 是某层权重。
- `X` 是校准数据经过这一层时的输入 activation。
- `Q(W)` 是量化后的权重。

把误差展开后，会出现近似 Hessian：

```text
H ≈ 2 X X^T
```

直觉是：

- 如果某个输入方向 activation 很大，那个方向上的权重误差会被放大。
- Hessian 描述不同权重维度对输出误差的敏感性和耦合关系。
- 所以不是所有权重同样重要，也不是只看权重绝对值。

GPTQ 的核心技巧是：

1. 逐列/逐块量化权重。
2. 每量化一个权重，就计算量化误差。
3. 用 Hessian inverse 估计这个误差对其他未量化权重的影响。
4. 对剩余权重做补偿更新，减少最终输出误差。

一个常见直觉公式是：

```text
quantize w_i -> q_i
error e_i = w_i - q_i
update remaining weights using H^{-1}
```

更具体地说，`H^{-1}` 负责告诉我们：当前维度的量化误差应该怎样分摊到后续维度，才能让整体层输出误差最小。

工程技巧：

- layer-wise reconstruction，避免全模型联合优化过贵。
- 使用校准集估计 activation Hessian。
- block/group 处理，控制复杂度。
- damping，避免 Hessian 病态或不可逆。
- act-order，把更重要/更敏感的列优先量化。
- 和高性能 W4A16 kernel 配合，否则只压显存不一定提速。

一句话：

> GPTQ 的数学核心是用 activation Hessian 近似输出误差曲率，用二阶信息指导量化顺序和误差补偿；工程核心是把原本很贵的二阶优化压缩成逐层、分块、可在大模型上跑完的 PTQ 流程。

### 10.4 AWQ：activation-aware scaling

AWQ 的观察是：LLM 中只有少量 salient weight channels 对输出特别重要，而且这些重要性更应该从 activation 分布判断，而不是只看 weight magnitude。

AWQ 的目标也是 weight-only 低比特量化，但它不依赖反向传播或 Hessian reconstruction。它用校准数据统计 activation，找到重要通道，然后用等价变换保护这些通道。

对线性层：

```text
y = x W
```

可以插入一个通道 scale `S`：

```text
y = (x S^{-1}) (S W)
```

在全精度下这是等价变换；但量化时，`S W` 的量化误差会改变。AWQ 通过放大重要通道的权重，让这些权重在低比特格点中更“可分辨”，从而降低 salient channels 的相对量化误差。

关键点：

- 用 activation 统计识别 salient channels。
- 保护少量重要权重，而不是混合精度保存大量 FP16 权重。
- 通过 scaling 做数学等价变换，避免硬件不友好的 mixed-precision 分支。
- scale 离线搜索/校准，推理时可融合到相邻层或权重中。

GPTQ vs AWQ：

| 维度 | GPTQ | AWQ |
|---|---|---|
| 核心信息 | Hessian/二阶信息 | activation 统计 |
| 优化目标 | 层输出重构误差 | 保护 activation 重要通道 |
| 是否做误差补偿 | 是 | 通常不做 Hessian 式补偿 |
| 工程特点 | 校准和求逆/分块更重 | 更轻量，泛化较好 |
| 常见部署 | W4A16 + GPTQ/Marlin kernels | W4A16 + AWQ kernels |

一句话：

> GPTQ 是“用二阶信息补偿量化误差”，AWQ 是“用 activation 统计找到重要通道，再用等价 scaling 保护它们”。

---

## 11. 稀疏、剪枝、蒸馏，以及结构化/非结构化剪枝

### 11.1 除量化外还有哪些优化路线

除了量化，常见还有：

- 稀疏化：让部分参数或激活不参与计算，例如 MoE、NVIDIA 2:4 structured sparsity。
- 剪枝：删除不重要的权重、通道、head、layer 或 FFN neuron。
- 蒸馏：用大 teacher 模型指导小 student 模型，降低部署成本。
- 低秩分解：把大矩阵近似成两个小矩阵。
- KV cache 优化：paged cache、prefix caching、KV quantization、offload。
- 架构级优化：MQA/GQA、MoE、speculative decoding、PD 分离。

### 11.2 非结构化剪枝

非结构化剪枝是按单个 weight 粒度删：

```text
W[i, j] -> 0
```

常见策略：

- magnitude pruning：绝对值小的权重删掉。
- gradient/sensitivity pruning：按梯度或二阶敏感度评估。
- iterative pruning：剪一点、微调、再剪。

优点：

- 粒度细，理论压缩率高。
- 同等稀疏率下质量损失可能更小。

缺点：

- 稀疏 pattern 不规则。
- 普通 GPU dense GEMM 不能直接加速。
- 需要稀疏 kernel、稀疏存储格式和硬件支持。
- 可能省参数存储，但不一定省真实推理时延。

面试重点：

> 非结构化剪枝容易得到高稀疏率，但如果没有硬件和 kernel 支持，实际推理可能只是模型文件变小，算子仍然按 dense 路径跑，速度不一定提升。

### 11.3 结构化剪枝

结构化剪枝删除的是一整块结构：

- attention head
- FFN hidden channel / neuron
- embedding dimension
- layer
- block
- expert

例如剪掉 FFN 的一部分 hidden channels：

```text
W1: [hidden, intermediate]
W2: [intermediate, hidden]
```

如果删除某些 intermediate channel，需要同时删除：

- `W1` 中对应输出列/行
- `W2` 中对应输入列/行
- 相关 bias、norm 或路由结构

优点：

- 形状真实变小。
- dense GEMM 也能直接变快。
- 更适合 GPU、TensorRT、cuBLAS 等已有高性能 kernel。
- 部署路径更简单。

缺点：

- 粒度粗，容易伤模型能力。
- 剪枝后通常需要微调恢复。
- Transformer 内部有残差、head、norm、KV cache 形状约束，不能随便删。

面试重点：

> 结构化剪枝更适合真实部署，因为它把矩阵 shape 变小，能直接减少 FLOPs 和显存；非结构化剪枝更像数学上稀疏，但要靠专门稀疏 kernel 才能变成实际加速。

### 11.4 蒸馏

蒸馏是让 student 学 teacher：

- logits distillation：学习 teacher 的 soft target。
- hidden-state distillation：对齐中间层表示。
- attention distillation：对齐 attention pattern。
- preference/behavior distillation：学习大模型的生成风格、推理轨迹或偏好。

优点：

- 最终模型结构规则，部署简单。
- 可以直接降低参数量、显存和延迟。

缺点：

- 需要数据、训练成本和调参。
- student 上限受容量限制。
- 对 reasoning、长上下文、工具调用这类能力压缩更难。

---

## 12. 如何判断 LLM 推理瓶颈

### 12.1 先拆端到端账本

不要一上来就说“GPU 慢”。先拆：

- 请求排队和 batching delay
- tokenizer / detokenizer
- prefill
- decode
- sampling
- KV cache 分配/回收
- NCCL / NVLink / IB 通信
- CPU launch / Python 调度

指标上同时看：

- TTFT：首 token 延迟，prefill 和排队影响很大。
- ITL/TPOT：decode 每 token 延迟。
- tokens/s、QPS、P95/P99。
- SM Active、SM Issue、Tensor Active、DRAM Bandwidth。
- NVLink/IB bandwidth、NCCL collective time、rank skew。

### 12.2 roofline 怎么用

Roofline 用两个硬件上界判断瓶颈：

```text
性能上限 = min(峰值算力, arithmetic_intensity * 峰值带宽)
```

其中：

```text
arithmetic_intensity = FLOPs / bytes moved
```

判断逻辑：

- AI 低，DRAM 带宽高，SM/Tensor Core 不高：memory-bound。
- AI 高，Tensor Core/SM 高，DRAM 未打满：compute-bound。
- 两者都低：可能是调度、同步、小 kernel、通信等待。

LLM 推理中：

- prefill 处理整段 prompt，矩阵较大，AI 高，更容易 compute-bound。
- decode 每次一个 token，小 batch + 读 KV cache，AI 低，更容易 memory-bound。

### 12.3 prefill 和 decode 是 GEMM 还是 GEMV

严格说 Transformer 每层都有线性层和 attention，底层会根据 shape 走 GEMM/GEMV 或专门 kernel。面试口径可以这样说：

`Prefill`：

```text
X: [B * S, hidden]
W: [hidden, 4hidden] 或 [hidden, hidden]
```

这是大矩阵乘大矩阵，更像 GEMM。attention 中 `QK^T` 和 `PV` 也有较大的矩阵计算。因此 prefill 通常更 compute-bound，更容易吃 Tensor Core，FlashAttention 收益也更明显。

`Decode`：

```text
X: [B, hidden]
W: [hidden, 4hidden]
```

如果 batch 很小，就更像 GEMV 或小 GEMM。attention 还要用当前 token 的 Q 去读历史 KV cache，历史长度越长，读 KV 的带宽压力越大。因此 decode 常常 memory-bound。

注意：如果 continuous batching 把 batch 做大，decode 的 MLP/linear 也可以变成比较像 GEMM；但 attention/KV cache 路径仍然经常受访存限制。

### 12.4 FlashAttention 主要用在哪个阶段

FlashAttention 主要解决 attention 中间矩阵的 HBM IO 问题，在 prefill/训练阶段最典型，因为 prefill 有长序列的 `QK^T` 和 `PV`，如果显式物化 score/probability 矩阵，IO 很重。

decode 阶段也有 attention 优化，但问题形态不完全一样：

- prefill：大块 attention，重点是减少 score/probability 中间矩阵读写。
- decode：当前 token Q 读历史 KV，batch 小时 SM 不满、KV cache 读写和长序列归约更突出。

所以 decode 更常听到 FlashDecoding、PagedAttention、FlashInfer、KV cache layout、split-k/split-kv 这类优化。

### 12.5 Marlin 算子

Marlin 是面向自回归 LLM 推理的 mixed-precision linear kernel，典型目标是 `W4A16`：

```text
activation: FP16/BF16
weight: INT4 packed
compute: dequant + matmul fused
```

它解决的问题是：weight-only 量化如果只是把权重压成 INT4，但推理时先反量化成 FP16 再调用普通 GEMM，就会浪费带宽和 kernel 时间。Marlin 把 packed INT4 权重读取、scale 应用、反量化和矩阵乘流水化/融合起来，尽量接近理论的 4x weight bandwidth 节省。

工程技巧包括：

- packed weight layout，适合 coalesced load。
- group-wise scale。
- 异步内存访问。
- pipelining 和复杂任务调度。
- 针对小到中等 batch 的 autoregressive inference 优化。

一句话：

> Marlin 的价值不是提出新的量化算法，而是让 GPTQ/AWQ 这类 W4A16 权重量化真的在 GPU 上跑快。

### 12.6 PD 解耦/分离怎么实现

先区分：

- `PD 解耦`：软件/调度层把 prefill 和 decode 队列、策略、执行路径拆开，但不一定物理分开。
- `PD 分离`：prefill worker 和 decode worker 运行在不同 GPU 或节点上，KV cache 需要通过 NVLink/IB 传输。

一个可复述的实现方案：

1. 请求先进入 router/scheduler。
2. scheduler 判断 prompt 长度、prefix cache 命中、当前 decode 队列压力。
3. 短 prompt 或 prefix 命中请求可在 decode engine 本地完成 prefill，避免 KV 传输开销。
4. 长 prompt、未命中、prefill 计算重的请求转发给 prefill engine。
5. prefill engine 计算首 token 和每层 KV cache。
6. KV cache 以 page/block 粒度传给 decode engine。
7. decode engine 接管 request state，进入 continuous batching。
8. 调度器跟踪 request_id、block table、sequence length、position、KV ownership。

关键权衡：

- prefill compute-bound，适合大 batch、Tensor Core、FlashAttention。
- decode memory-bound，适合连续批处理、KV cache 优化、低延迟调度。
- PD 分离带来资源专用化，但新增 KV 传输、状态一致性、失败恢复和跨节点调度复杂度。

实习回答模板：

> 我们当时不是简单把所有请求都走 PD 分离，而是做了条件路由：短请求、prefix cache 命中请求尽量留在 decode 侧本地处理；长 prompt 或 prefill 压力明显的请求才转发到 prefill 侧。实现上维护 request state 和 KV block handle，prefill 侧完成 KV 构建后把 block metadata 和 KV 数据传给 decode 侧，decode 侧再进入 continuous batching。优化目标是降低 decode 队列被长 prefill 阻塞的概率，同时控制 KV 跨设备传输成本。

请把其中“我们当时”的细节替换为真实实习实现，不要在面试里编具体数字。

---

## Nsight：DCGM/NVML 不够时怎么分析多卡性能

### Nsight Systems

`Nsight Systems` 回答“时间花在哪、流水线哪里断了”。

重点看：

- CPU launch 和 GPU kernel 之间有没有 gap。
- GPU timeline 是否有 idle bubble。
- memcpy 是否进入关键路径。
- stream 是否被同步点串行化。
- NCCL 和 compute 是否 overlap。
- 每个 rank 的 timeline 是否一致。
- prefill、decode、sampling、NCCL 各阶段占比。

常见命令：

```bash
nsys profile \
  -t cuda,nvtx,osrt,cublas,cudnn \
  --output=nsys_rank_%q{LOCAL_RANK} \
  torchrun --nproc_per_node=8 run.py
```

最好配合 NVTX：

```text
prefill / decode / attention / mlp / sampling / nccl / kv_transfer
```

### Nsight Compute

`Nsight Compute` 回答“某个 kernel 为什么慢”。

重点看：

- achieved occupancy
- SM Active / SM Issue
- Tensor Core utilization
- FP32/BF16/FP16 pipe active
- warp stall reasons
- DRAM throughput
- L2 hit rate
- coalescing
- shared memory bank conflict
- register spilling

判断套路：

| 现象 | 可能瓶颈 |
|---|---|
| GPU timeline 有 gap | CPU launch、同步、小 kernel、CUDA Graph 缺失 |
| NCCL 在关键路径 | 通信瓶颈、rank skew、拓扑问题 |
| Tensor Active 低，FP32 pipe 高 | dtype/shape/backend fallback |
| DRAM 高，SM Issue 不高 | memory-bound，常见于 decode/KV cache |
| Occupancy 低 | 寄存器/shared memory/block size 限制 |
| Occupancy 高但性能低 | 可能访存、分支、同步或 Tensor Core 没命中 |

面试总结：

> DCGM/NVML 是常驻监控和告警，Nsight Systems 是系统级时间线，Nsight Compute 是 kernel 显微镜。多卡 LLM 问题要先用 Systems 判断是 CPU、通信、同步还是 kernel，再用 Compute 下钻热点 kernel 的 occupancy、pipe、stall 和访存。

---

## 13. A100/H20 异构集群与单卡/多卡/多机差异

### 13.1 A100 和 H20 算力/通信对比

先说明：H20 公开规格资料不像 A100/H100 官方 datasheet 那么稳定，实际面试最好以你们集群里的 `nvidia-smi -q`、DCGM、NCCL tests、厂商 BOM 为准。下面是公开资料口径下的常见比较。

| 项 | A100 80GB SXM | H20/HGX H20 常见公开口径 |
|---|---:|---:|
| 架构 | Ampere | Hopper/GH100 export-limited |
| 显存 | 80GB HBM2e | 96GB HBM3 |
| 显存带宽 | 约 2.0 TB/s | 约 4.0 TB/s |
| BF16/FP16 Tensor | 约 312 TFLOPS dense | 约 148 TFLOPS dense |
| INT8 Tensor | 约 624 TOPS dense | 约 296 TOPS dense |
| FP8 | 无原生 Hopper FP8 Transformer Engine | 支持 Hopper FP8/Tensor Core 路径，但峰值被限制 |
| NVLink | 约 600 GB/s | 公开资料常见 900 GB/s |

重要结论：

- A100 的 BF16/FP16 dense tensor 算力通常高于 H20。
- H20 的显存容量和带宽更强，且属于 Hopper，有 FP8/Transformer Engine 路径。
- 对 compute-bound prefill，A100 可能更有优势。
- 对 memory-bound decode、长上下文、KV cache 压力大的任务，H20 的 96GB/4TB/s 可能更有吸引力。
- 异构调度不能只按“卡名”分，要按任务瓶颈分：长 prompt/prefill、短 decode、量化 FP8、KV cache 容量、多卡通信需求都不一样。

### 13.2 H20/Hopper 对什么精度更友好

Hopper 的重要变化是 Transformer Engine 和第四代 Tensor Core，支持混合 FP8/FP16 精度路径，适合 Transformer 训练和推理。与 A100 相比：

- A100 主要强在 TF32、FP16、BF16、INT8/INT4 Tensor Core。
- Hopper 增强 FP8，Transformer Engine 可以在 FP8/FP16 混合精度间切换。
- H20 作为 Hopper 限制版，仍应关注 FP8/BF16/FP16 Tensor Core 路径，但峰值规格被削弱。

面试口径：

> H20 的亮点不是绝对 BF16 算力压过 A100，而是 Hopper 架构带来的 FP8/Transformer Engine、较大显存和更高带宽。实际部署时，我会看模型是否有 FP8 kernel、KV cache 是否带宽受限、batch/seq length 是否让任务偏 compute-bound 还是 memory-bound。

### 13.3 单机单卡、单机多卡、多机多卡差异

#### 单机单卡

优点：

- 最简单，没有跨卡通信。
- debug 容易，性能瓶颈主要在 kernel、显存、CPU launch、KV cache。

限制：

- 模型必须放得下。
- 吞吐和并发受单卡容量/带宽/算力限制。

适合：

- 小模型、量化模型、低并发服务、debug baseline。

#### 单机多卡

主要新增：

- NVLink/NVSwitch/PCIe 拓扑。
- TP/PP/DP/EP 等并行策略。
- NCCL collective。
- rank 间负载均衡。

优点：

- 可放更大模型。
- 节点内 NVLink/NVSwitch 通信相对快。

问题：

- TP 每层通信，容易被 all-reduce/all-gather 卡住。
- PCIe 拓扑不佳时卡间通信差异明显。
- 某个 rank 慢会拖全局。

#### 多机多卡

主要新增：

- IB/RoCE 网络。
- 跨节点 NCCL。
- 拓扑感知放置。
- 网络拥塞、链路抖动、节点故障。
- checkpoint、容错、调度和数据面复杂度。

和单机多卡最大区别：

- 跨节点带宽和延迟远差于节点内 NVLink/NVSwitch。
- 通信优化从“选并行策略”变成“策略 + 拓扑 + 调度 + 网络运维”。
- 多机上 TP 拉太宽通常不划算，PP/DP/EP/PD 分离要结合流量和模型结构设计。

面试总结：

> 单卡主要看 kernel 和显存；单机多卡开始看 NVLink/PCIe 和 NCCL；多机多卡则必须把 IB/RoCE、拓扑、rank placement、通信 overlap 和故障恢复一起考虑。A100/H20 异构时，还要根据 prefill/decode、FP8/BF16、KV cache、显存容量和互联带宽做任务放置。

---

## 14. 大模型或深度学习的数据预加载与 Fluid Dataset

### 14.1 数据预加载解决什么问题

训练或离线推理中，GPU 经常不是纯算力不够，而是被数据链路饿住：

```text
storage -> CPU decode/parse -> transform/tokenize -> CPU batch -> H2D copy -> GPU compute
```

数据预加载的目标是把这条链路流水化：

- CPU 读下一批数据时，GPU 正在算当前批。
- CPU transform/tokenize 和 H2D copy 尽量异步。
- 数据格式尽量顺序读、少小文件、少随机 seek。
- 多卡训练中，每个 rank 读取不同 shard，避免重复和热文件。

常见手段：

- 多 worker 数据加载：`num_workers`、线程/进程池。
- prefetch queue：提前准备若干 batch。
- pinned memory：加快 CPU 到 GPU 的异步拷贝。
- async H2D：`non_blocking=True` 或框架等价能力。
- mmap / LMDB / Parquet / WebDataset / RecordIO：减少小文件和 Python 解析开销。
- dataset shard + distributed sampler：每个 rank 读不同数据。
- cache：热样本、tokenized result、image decode result 放内存/本地 SSD。
- double buffering：一个 buffer 给 GPU 算，另一个 buffer 准备下一批。

面试口径：

> 数据预加载本质是把 I/O、CPU 预处理、H2D copy 和 GPU compute 做 pipeline overlap。好的 dataloader 不是“多开几个 worker”这么简单，还要考虑数据格式、shuffle 粒度、分布式切分、缓存位置和异常恢复。

### 14.2 Dataset 怎么做灵活抽象

核心是把“数据是什么”和“怎么读、怎么变换、怎么组 batch”分开。

一个典型 Dataset 抽象包含：

- `__len__`：数据规模。
- `__getitem__` 或 iterator：按 index 或流式产生样本。
- `schema/features`：字段定义，例如 input_ids、labels、image、metadata。
- transform/tokenizer：样本级转换。
- collate_fn：把多个样本拼成 batch，处理 padding、mask、position id。
- sampler/sharder：决定样本顺序和分布式切分。
- state/checkpoint：记录当前 epoch、offset、random seed，便于断点恢复。

可以拆成几层：

```text
StorageSource: 本地文件 / 对象存储 / HDFS / Kafka / 数据库
Dataset: 样本抽象，负责读取和轻量解析
Transform: decode / tokenize / augment / filter
Sampler: shuffle / shard / curriculum / weighted sampling
Collator: padding / packing / batch-level tensor 化
DataLoader: worker pool / prefetch / memory pin / device copy
```

大模型训练还会特别关注：

- sequence packing：多个短样本 pack 到同一 context，减少 padding。
- dynamic batching：按 token 数而不是样本数组 batch。
- deterministic shuffling：多机多卡下可复现。
- failure recovery：reader offset 和 sampler state 可恢复。
- 数据质量过滤：去重、长度裁剪、脏样本跳过。

### 14.3 Fluid Dataset 口径

如果这里的 `Fluid` 指 PaddlePaddle Fluid，Dataset 体系里常见的是：

- `DatasetFactory().create_dataset(...)`
- `QueueDataset`
- `InMemoryDataset`
- `FileInstantDataset`
- `train_from_dataset(...)`

`QueueDataset` 更偏流式读取，适合数据大、不能全放内存的场景。`InMemoryDataset` 会把数据加载到内存，并支持本地 shuffle、分布式 global shuffle、释放内存等能力。

一个典型流程：

```python
import paddle.fluid as fluid

dataset = fluid.DatasetFactory().create_dataset("InMemoryDataset")
dataset.set_use_var([data, label])
dataset.set_filelist(filelist)
dataset.set_thread(8)
dataset.set_queue_num(8)
dataset.load_into_memory()
dataset.local_shuffle()

exe.train_from_dataset(program, dataset)
dataset.release_memory()
```

面试里要强调它的设计思想：

- Dataset 是数据源和训练执行器之间的统一抽象。
- `set_use_var` 绑定模型需要喂入的变量。
- `set_filelist` 把数据文件列表交给 Dataset。
- `thread_num / queue_num` 决定读取和输出队列并发。
- `InMemoryDataset` 用内存换读取速度和 shuffle 能力。
- `QueueDataset` 用流式队列降低内存占用。
- 分布式场景下可配合 fleet 做 global shuffle。

一句话：

> Fluid Dataset 的价值是把数据读取、解析、shuffle、队列和执行器训练入口统一起来，让用户只定义数据文件、变量 schema 和并发参数，底层由 Dataset/DataFeed 负责高吞吐喂数。

---

## 15. Agent 沙箱环境秒级启动怎么设计

### 15.1 sandbox 解决什么问题

Agent 会运行模型生成的代码、shell 命令、包安装、文件读写和网络访问。风险包括：

- 逃逸到宿主机。
- 读取其他用户数据。
- 滥用网络扫描或外连。
- 写爆磁盘、fork bomb、挖矿。
- 持久化恶意文件或污染后续会话。

所以 sandbox 不是简单 Docker，而是要明确隔离边界：

| 方案 | 启动速度 | 隔离强度 | 适合场景 |
|---|---|---|---|
| Linux namespace + cgroup + seccomp | 快 | 中 | 可信度较高、低成本任务 |
| gVisor | 较快 | 较强 | 不可信容器、系统调用隔离 |
| Kata/Firecracker microVM | 中到快 | 强 | 多租户、不可信代码、强隔离 |
| Wasm/WASI | 很快 | 强但能力受限 | 受控语言/runtime、小工具执行 |

面试总纲：

> 秒级启动不是靠每次冷启动完整 VM，而是靠镜像预热、microVM/container 池化、快照恢复、overlayfs、异步网络/磁盘准备和会话状态外置。

### 15.2 秒级启动架构

可以拆成控制面和数据面：

```text
API Gateway
  -> Sandbox Scheduler
      -> Warm Pool Manager
      -> Image/Rootfs Cache
      -> State Store
      -> Network Policy Manager
      -> Sandbox Runtime: gVisor / Firecracker / Kata / containerd
```

启动路径：

1. 用户请求进入 scheduler。
2. scheduler 根据语言/runtime/安全等级选择 sandbox class。
3. 优先从 warm pool 取一个已启动但未绑定用户的 sandbox。
4. 绑定 session_id、挂载只读 base image 和 per-session writable overlay。
5. 注入最小配置、token、workspace、网络策略。
6. in-sandbox agent daemon 接收 exec/file/pty 请求。
7. 会话结束后清理 overlay 或保存快照，再把 sandbox 销毁或归池。

为了秒级启动：

- base image 预拉取。
- rootfs 预解压或 snapshot 化。
- microVM snapshot/restore。
- 语言 runtime 预热，例如 Python/Node 常用包预装。
- 网络 namespace、TAP、iptables/nftables 规则预创建或池化。
- overlayfs writable layer 秒级挂载。
- 不在启动热路径做包安装、大文件下载和复杂初始化。

### 15.3 并发弹性调度

调度器要解决两个问题：高并发下快速分配，以及资源不被打爆。

核心机制：

- sandbox warm pool：按 runtime、安全等级、镜像版本维护池。
- 分级队列：短任务、交互任务、长任务分队列。
- bin packing：按 CPU、memory、disk、GPU、network quota 放置。
- backpressure：池耗尽时排队、降级、拒绝或扩容。
- autoscaling：根据队列长度、冷启动率、CPU/mem 使用率扩容节点。
- fair scheduling：按用户/租户配额，避免单用户占满。
- preemption/timeout：长时间 idle 或超时任务回收。

一个可讲的调度策略：

```text
优先 warm sandbox
  -> 没有则 snapshot restore
  -> 再没有才 cold boot
  -> 超过 SLA 则排队或扩容
```

资源隔离：

- cgroup v2 限制 CPU、memory、pids、io。
- per-sandbox disk quota。
- network egress policy。
- GPU/MIG 或完全禁用 GPU。
- syscall policy / seccomp / gVisor interception。

### 15.4 会话级状态保持

Agent sandbox 通常不是一次性函数，需要保持：

- 当前工作目录和文件。
- 安装的依赖。
- shell 环境变量。
- notebook/kernel state。
- 进程状态或至少命令历史。
- 用户上传文件和生成文件。

但状态不能和安全隔离冲突。推荐做法是状态分层：

| 状态类型 | 保存位置 | 生命周期 |
|---|---|---|
| base image | 只读镜像层 | 长期复用 |
| session workspace | per-session overlay / volume | 会话级 |
| package cache | 受控共享缓存 | 可跨会话但需校验 |
| process/kernel state | sandbox 内存态或 checkpoint | 短期 |
| metadata | 控制面数据库 | 长期 |

会话恢复有三种等级：

- 文件级恢复：只恢复 workspace，重新启动进程，最稳。
- 环境级恢复：保留依赖安装和缓存，启动较快。
- 进程级恢复：checkpoint/restore 或长期保活，体验最好但复杂。

实际系统常用：

- 会话活跃期保持 warm sandbox。
- idle 超时后停止计算资源，只保留 overlay。
- 再次进入时用同镜像 + overlay 恢复。
- 对 notebook/REPL 可选保活或 checkpoint。

### 15.5 强安全隔离

强安全隔离要做纵深防御：

- 计算隔离：microVM/gVisor/Kata，不直接共享宿主 kernel 或降低 syscall 暴露面。
- 权限隔离：非 root、drop capabilities、只读 rootfs。
- 文件隔离：per-session overlay，禁止挂载宿主敏感路径。
- 网络隔离：默认 deny，按 allowlist 放行；禁止内网探测和 metadata service。
- 资源隔离：cgroup、quota、ulimit、pids 限制。
- 供应链隔离：镜像签名、依赖源代理、安装包扫描。
- 审计：记录 exec、文件变更、网络连接、资源峰值。
- 清理：会话结束销毁 writable layer，密钥短期有效。

面试总结：

> 秒级 agent sandbox 的核心是“预热 + 快照 + 池化 + 状态外置”。安全上不能只靠 Docker，要根据威胁模型选择 gVisor 或 microVM，并配合 cgroup、网络策略、只读镜像、per-session overlay 和审计。调度上要用 warm pool、配额、backpressure 和 autoscaling 保证高并发下既快又不互相影响。

---

## 16. 推理显存优化、TTFT 加速与投机解码

### 16.1 推理主要显存组成

LLM 推理显存大头一般是：

```text
总显存 ≈ 权重 + KV Cache + 临时激活/Workspace + runtime/cache/fragmentation
```

其中：

- 权重：与参数量和 dtype 线性相关，量化主要压这一部分。
- KV Cache：与 batch、上下文长度、生成长度、层数、KV heads、head_dim、dtype 线性相关。
- 临时激活/workspace：prefill attention、GEMM workspace、CUDA graph pool 等。
- 碎片/预留：allocator cache、paged block 预留、框架 runtime。

KV Cache 公式：

```text
KV bytes ≈ 2 * B * S * L * H_kv * D_head * bytes_per_elem
```

### 16.2 显存优化技术

按对象分：

#### 权重

- FP16/BF16 -> FP8/INT8/INT4。
- GPTQ/AWQ/SmoothQuant。
- weight sharing / tensor parallel / pipeline parallel。
- CPU/NVMe offload，适合容量兜底但会牺牲时延。

#### KV Cache

- PagedAttention：减少碎片和预分配浪费。
- Prefix Caching：复用公共前缀，降低 TTFT 和 prefill 显存/算力。
- MQA/GQA：减少 KV heads，直接降低 KV cache。
- KV cache quantization：FP16/BF16 -> FP8/INT8/INT4。
- sliding window / attention sink / eviction：限制历史长度。
- KV offload：GPU 放热 KV，CPU/SSD 放冷 KV。
- Shared KV Cache：模型结构层面的 KV 共享。

#### 临时激活和 workspace

- FlashAttention：减少 prefill attention 中间矩阵物化。
- kernel fusion：减少中间 tensor。
- CUDA Graph memory pool 管理。
- 限制 max batch、max seq、max_num_batched_tokens。
- chunked prefill：把大 prompt 拆块，控制峰值显存。

#### 调度层

- admission control：按 KV cache 预算接请求。
- continuous batching：提高 decode 利用率，但要控制 KV 增长。
- prefix/cache-aware routing：让请求去已有 cache 的节点。
- PD 分离：避免长 prefill 阻塞 decode，也便于为两阶段配置不同显存策略。

### 16.3 更倾向总显存优化还是推理加速/TTFT

面试里不要二选一，要按目标拆：

| 目标 | 优先优化 |
|---|---|
| 单卡能不能放下模型 | 权重量化、TP/PP、offload |
| 长上下文/高并发 | KV cache 管理、PagedAttention、GQA/MQA、KV quant |
| TTFT | prefix caching、chunked prefill、FlashAttention、prefill batch 调度 |
| ITL/TPOT | decode batching、KV layout、Marlin/W4A16、speculative decoding |
| 成本/QPS | 量化、continuous batching、cache-aware routing |
| P99 延迟 | admission control、队列隔离、PD 分离、避免过大 batch |

我个人更倾向的面试回答：

> 如果线上服务已经能放下模型，我会优先优化 TTFT/ITL 和单位 GPU 吞吐，因为用户体验和成本最终体现在延迟与 tokens/s 上；但如果长上下文或并发导致 KV cache 爆掉，就要先做显存账本和 admission control，否则任何加速都不稳定。也就是说，先保证显存可控，再优化端到端延迟。

### 16.4 TTFT 怎么优化

TTFT 主要由：

```text
排队 + prefill + 首次采样 + 网络返回
```

优化手段：

- Prefix Caching：命中公共 system prompt 或文档前缀，直接跳过重复 prefill。
- Chunked Prefill：把超长 prompt 拆块，和其他 decode 交错，减少队头阻塞。
- FlashAttention：降低 prefill attention IO。
- 更大 prefill batch：提升 Tensor Core 利用率，但要控制排队。
- prompt compression / retrieval pruning：减少输入 token。
- PD 分离：长 prefill 交给 prefill worker，不阻塞 decode worker。
- tokenizer 异步化和缓存：避免 CPU 预处理拖慢首 token。

### 16.5 投机解码

投机解码解决的是自回归 decode 每步只能生成一个 token 的串行瓶颈。

基本流程：

1. draft model 或轻量预测机制先猜多个候选 token。
2. target model 一次 forward 并行验证这些候选。
3. 按接受/拒绝规则保留一段候选。
4. 如果接受多个 token，就减少 target model 调用次数。

常见路线：

- draft model：小模型猜，大模型验。
- Medusa/MTP：主模型额外 prediction heads 预测多个未来 token。
- EAGLE：用特征层预测候选。
- ngram/suffix decoding：利用 prompt 或历史重复模式猜 token。

收益取决于：

- draft 速度是否足够快。
- 接受率是否高。
- target 验证是否能并行吃满 GPU。
- KV cache 是否支持 speculative token 的预留、提交和回滚。
- 额外调度/采样开销是否小于减少的 decode 步数。

它更偏加速 ITL/TPOT，不一定显著降低 TTFT。对有重复模式、代码补全、格式化输出、低温采样的场景更容易有效；如果 draft 经常猜错，可能反而变慢。

面试总结：

> 投机解码不是减少 target model 单次 forward 的成本，而是减少 target model 串行 forward 的次数。它用便宜候选换昂贵验证，接受率越高越赚；系统难点在 draft-target 协同、KV cache 回滚、batch 调度和采样分布正确性。

---

## 外部参考

- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers: https://arxiv.org/abs/2210.17323
- AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration: https://arxiv.org/abs/2306.00978
- SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models: https://arxiv.org/abs/2211.10438
- LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale: https://arxiv.org/abs/2208.07339
- Marlin: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models: https://arxiv.org/abs/2408.11743
- NVIDIA Hopper Architecture: https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/
- NVIDIA A100 Datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
- H20 公开规格参考：Tom's Hardware H20/H100 comparison, https://www.tomshardware.com/pc-components/gpus/the-tale-of-nvidias-hgx-h20-how-an-ai-gpu-became-a-political-lightning-rod
- Paddle Fluid Dataset API: https://paddlepaddle-org-cn.bj.bcebos.com/documentation/docs/en/api/dataset.html
- Paddle InMemoryDataset API: https://www.paddlepaddle.org.cn/documentation/docs/en/2.2/api/paddle/fluid/dataset/InMemoryDataset_en.html
- gVisor documentation: https://gvisor.dev/docs/
- Kata Containers: https://katacontainers.io/
