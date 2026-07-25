---
type: source
source_kind: 面试整理
topic: 性能分析
updated: 2026-07-25
---

# 量化剪枝推理瓶颈Nsight与异构集群面试整理

## 来源信息

- 标题：量化、剪枝、推理瓶颈、Nsight、异构集群、数据预加载与沙箱面试整理
- 作者：LLM 归纳整理
- 日期：2026-05-07
- 类型：面试问答整理 / query report
- 原始文件：[[../../output/interview/量化剪枝推理瓶颈Nsight与异构集群面试整理|量化剪枝推理瓶颈Nsight与异构集群面试整理]]

## 2-3 条核心摘要

- 量化主流路线包括 PTQ/QAT、weight-only、W8A8、FP8、KV cache quantization；GPTQ 以 Hessian/二阶误差补偿为核心，AWQ 以 activation-aware scaling 保护重要通道为核心。
- 推理瓶颈要分 prefill/decode：prefill 更像大 GEMM、偏 compute-bound，FlashAttention 收益明显；decode 更像小 GEMM/GEMV + KV cache 读取，常偏 memory-bound。
- 多卡异构集群分析需要把 Nsight Systems/Compute、DCGM/NVML、NCCL tests 和硬件规格结合起来；A100/H20 的取舍要同时看 BF16/FP16/FP8 算力、显存容量、HBM 带宽和 NVLink/IB 拓扑。
- 新增补充覆盖数据预加载与 Paddle Fluid Dataset 抽象、Agent 沙箱秒级启动架构，以及推理显存优化、TTFT 优化和投机解码。

## 值得关注的论断

- Hessian 在 GPTQ 中不是泛泛的“二阶信息”，而是由校准 activation 形成的层输出误差曲率近似，决定量化误差如何补偿到未量化权重。
- AWQ 的 scaling 是全精度等价但量化后不等价的变换：通过放大 activation 重要通道对应的权重，让低比特量化更好保留 salient weights。
- PD 分离不是所有请求都值得走，真实系统更常用条件路由：短请求或 prefix 命中请求留在 decode 侧，长 prompt/未命中请求才转发到 prefill 侧。
- Agent sandbox 若要秒级启动，核心不是每次冷启动完整 VM，而是镜像预热、microVM/container 池化、快照恢复、per-session overlay 和状态外置。
- 推理显存优化要先拆权重、KV Cache、临时 workspace 和 allocator 预留；TTFT 更关注 prefill、prefix caching、chunked prefill 和调度，投机解码更偏 ITL/TPOT 优化。

## 关键概念

- [[../concepts/混合精度训练与推理]]
- [[../concepts/Roofline 模型]]
- [[../concepts/Profiling]]
- [[../concepts/KV Cache]]
- [[../concepts/FlashAttention]]
- [[../concepts/PD分离]]
- [[../concepts/GPU执行模型]]
- [[../concepts/集合通信]]
- [[../concepts/PagedAttention]]
- [[../concepts/Prefix Caching]]
- [[../concepts/Speculative Decoding]]

## 相关实体

- [[../entities/Stanford CS336]]
- [[../entities/vLLM]]
- [[../entities/TensorRT-LLM]]
- [[../entities/Nvidia Dynamo]]

## 与现有 wiki 的关系

- 会更新哪些概念页：本次先新增来源页和报告页；后续可拆出 `GPTQ`、`AWQ`、`剪枝`、`模型蒸馏`、`Marlin`、`A100`、`H20`、`数据预加载`、`Sandbox` 等独立概念/实体页。
- 会更新哪些实体页：暂无。
- 是否存在冲突：未发现与现有 wiki 直接冲突；H20 规格来自公开二手资料，应以实际集群和厂商资料核实。

## 待确认

- H20 的精确 SKU、NVLink 带宽、FP8/BF16 峰值和集群拓扑应以用户实际实习集群为准。
- PD 分离实习经历中的真实组件名、链路、优化幅度和指标需要用户补充后再固化为个人项目叙述。
- `Fluid` 若在面试语境中不是 PaddlePaddle Fluid，而是公司内部数据系统，应替换为对应内部 Dataset/DataFeed 抽象。
- 沙箱方案需要根据威胁模型选择 gVisor、Kata/Firecracker microVM 或普通容器；若涉及 GPU passthrough，还需补充设备隔离和调度策略。
