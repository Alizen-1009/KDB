# 知识库操作日志

按时间记录 ingest、query、lint 等操作，帮助 LLM 与人类共同追踪知识库的演化过程。

## [2026-04-23] ingest | Model Runner V2 A Modular and Faster Core for vLLM

- 读取原始资料：`raw/articles/Model Runner V2 A Modular and Faster Core for vLLM.md`
- 创建来源页：`wiki/sources/Model Runner V2 A Modular and Faster Core for vLLM.md`
- 创建概念页：`wiki/concepts/持久批处理.md`
- 创建实体页：`wiki/entities/vLLM Team.md`
- 更新实体页：`wiki/entities/vLLM.md`
- 更新概念页：`wiki/concepts/Continuous Batching.md`
- 未发现与现有 wiki 的直接冲突；本次主要把 `vLLM` 条目从 `PagedAttention / Continuous Batching` 扩展到执行核心重构视角，补入了 `MRV2`、`GPU-native input preparation` 和 `async-first` 这条主线

## [2026-04-23] query | 推理系统专题面试稿

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取概念页：`wiki/concepts/PagedAttention.md`
- 读取概念页：`wiki/concepts/Continuous Batching.md`
- 读取概念页：`wiki/concepts/Speculative Decoding.md`
- 读取实体页：`wiki/entities/TensorRT-LLM.md`
- 生成查询报告：`output/reports/推理系统专题面试稿.md`
- 更新概念页：`wiki/concepts/PagedAttention.md`
- 更新概念页：`wiki/concepts/Continuous Batching.md`
- 更新实体页：`wiki/entities/TensorRT-LLM.md`
- 本次输出围绕 `KV Cache / PagedAttention / Copy-on-Write / Batch Size / P99 延迟优化 / Speculative Decoding / TensorRT-LLM / 推理引擎调优` 形成了一份可直接复述的专题稿，并把高价值结论回填到相关页面

## [2026-04-23] query | 字节二面代码模板落地

- 基于报告：`output/reports/字节二面高压题拆解.md`
- 生成代码文件：`output/code/bytedance_pressure_round_modules.py`
- 生成测试文件：`output/code/test_bytedance_pressure_round_modules.py`
- 代码覆盖：`MHA`、`MoE`、`MLA`、`reduce`
- 运行验证：`python3 output/code/test_bytedance_pressure_round_modules.py`
- 运行验证：`python3 output/code/bytedance_pressure_round_modules.py`
- 备注：代码以“本地可运行、便于手撕复习”为目标，优先保证 shape 清晰和最小实现，而非生产级性能

## [2026-04-23] query | 字节二面高压题拆解

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/FlashAttention.md`
- 读取概念页：`wiki/concepts/Online Softmax.md`
- 读取概念页：`wiki/concepts/CUDA Kernel.md`
- 读取概念页：`wiki/concepts/Block Reduce.md`
- 读取概念页：`wiki/concepts/Warp Shuffle Reduce.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取既有报告：`output/reports/面试经验.md`
- 读取既有报告：`output/reports/大模型系统面试题全答.md`
- 生成查询报告：`output/reports/字节二面高压题拆解.md`
- 本次输出围绕 `linear attention / MoE / MHA / MLA / DeepSeek-V3 / reduce / heap sort` 给出一版高压面拆解，重点补了可手撕的最小代码模板以及 `参数量 / FLOPs / 访存` 的口算框架

## [2026-04-23] query | AI Infra面试题全答（二）

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/数据并行.md`
- 读取概念页：`wiki/concepts/Tensor Parallelism.md`
- 读取概念页：`wiki/concepts/流水线并行.md`
- 读取概念页：`wiki/concepts/ZeRO.md`
- 读取概念页：`wiki/concepts/Torch Distributed.md`
- 读取概念页：`wiki/concepts/集合通信.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取概念页：`wiki/concepts/PagedAttention.md`
- 读取概念页：`wiki/concepts/Continuous Batching.md`
- 读取概念页：`wiki/concepts/FlashAttention.md`
- 读取概念页：`wiki/concepts/Profiling.md`
- 读取实体页：`wiki/entities/NCCL.md`
- 读取实体页：`wiki/entities/vLLM.md`
- 读取实体页：`wiki/entities/SGLang.md`
- 读取实体页：`wiki/entities/TensorRT-LLM.md`
- 读取既有报告：`output/reports/大模型系统面试题全答.md`
- 读取既有报告：`output/reports/多卡与推理系统面试梳理.md`
- 生成查询报告：`output/reports/AI Infra面试题全答（二）.md`
- 本次输出补齐了更偏平台、参数服务器、调度、存储、MLOps、管理协作和行为面的题目；其中部分题采用了“可替换模板”形式，便于后续用个人真实经历替换

## [2026-04-23] query | 大模型系统面试题全答

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Roofline 模型.md`
- 读取概念页：`wiki/concepts/数据并行.md`
- 读取概念页：`wiki/concepts/Tensor Parallelism.md`
- 读取概念页：`wiki/concepts/流水线并行.md`
- 读取概念页：`wiki/concepts/ZeRO.md`
- 读取概念页：`wiki/concepts/Torch Distributed.md`
- 读取概念页：`wiki/concepts/集合通信.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取概念页：`wiki/concepts/PagedAttention.md`
- 读取概念页：`wiki/concepts/Continuous Batching.md`
- 读取概念页：`wiki/concepts/FlashAttention.md`
- 读取概念页：`wiki/concepts/Speculative Decoding.md`
- 读取实体页：`wiki/entities/NCCL.md`
- 读取实体页：`wiki/entities/vLLM.md`
- 读取实体页：`wiki/entities/SGLang.md`
- 读取实体页：`wiki/entities/TensorRT-LLM.md`
- 读取既有报告：`output/reports/面试经验.md`
- 读取既有报告：`output/reports/多卡与推理系统面试梳理.md`
- 读取既有报告：`output/reports/大模型系统面试题地图.md`
- 生成查询报告：`output/reports/大模型系统面试题全答.md`
- 本次输出按题单顺序给出一版可面试复述的全答，覆盖概念题、系统设计题、行为题和手写题；其中量化、大规模集群和平台工程部分仍属于高层答法，后续可继续拆专题

## [2026-04-23] query | 大模型系统面试题地图

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Roofline 模型.md`
- 读取概念页：`wiki/concepts/数据并行.md`
- 读取概念页：`wiki/concepts/Tensor Parallelism.md`
- 读取概念页：`wiki/concepts/流水线并行.md`
- 读取概念页：`wiki/concepts/ZeRO.md`
- 读取概念页：`wiki/concepts/Torch Distributed.md`
- 读取概念页：`wiki/concepts/集合通信.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取概念页：`wiki/concepts/PagedAttention.md`
- 读取概念页：`wiki/concepts/Continuous Batching.md`
- 读取概念页：`wiki/concepts/FlashAttention.md`
- 读取概念页：`wiki/concepts/Speculative Decoding.md`
- 读取实体页：`wiki/entities/NCCL.md`
- 读取实体页：`wiki/entities/vLLM.md`
- 读取实体页：`wiki/entities/SGLang.md`
- 读取实体页：`wiki/entities/TensorRT-LLM.md`
- 读取既有报告：`output/reports/面试经验.md`
- 读取既有报告：`output/reports/多卡与推理系统面试梳理.md`
- 生成查询报告：`output/reports/大模型系统面试题地图.md`
- 本次输出的目标不是逐题作答，而是把题单压缩成“硬件性能 -> 分布式训练 -> 推理系统 -> 平台工程”四层知识地图，并标出当前 wiki 的覆盖区与空白区

## [2026-04-23] ingest | 美团一面：请介绍 vLLM PageAttention

- 读取原始资料：`raw/articles/美团一面：请介绍 vLLM PageAttention.md`
- 创建来源页：`wiki/sources/美团一面：请介绍 vLLM PageAttention.md`
- 更新概念页：`wiki/concepts/PagedAttention.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 更新实体页：`wiki/entities/vLLM.md`
- 未发现与现有 wiki 的直接冲突，本次主要把 `PagedAttention` 从“分页式 KV 管理”推进到“logical block / physical block / block table + prefill/decode 运行流程”视角

## [2026-04-23] query | 多卡与推理系统面试梳理

- 读取索引页：`wiki/index.md`
- 读取概念页：`wiki/concepts/Tensor Parallelism.md`
- 读取概念页：`wiki/concepts/集合通信.md`
- 读取概念页：`wiki/concepts/Torch Distributed.md`
- 读取概念页：`wiki/concepts/Continuous Batching.md`
- 读取概念页：`wiki/concepts/KV Cache.md`
- 读取概念页：`wiki/concepts/PD分离.md`
- 读取概念页：`wiki/concepts/Prefix Caching.md`
- 读取概念页：`wiki/concepts/缓存感知路由.md`
- 读取概念页：`wiki/concepts/PagedAttention.md`
- 读取实体页：`wiki/entities/NCCL.md`
- 读取实体页：`wiki/entities/vLLM.md`
- 读取实体页：`wiki/entities/SGLang.md`
- 读取既有报告：`output/reports/面试经验.md`
- 生成查询报告：`output/reports/多卡与推理系统面试梳理.md`
- 本次输出以面试口径为主，重点串联 `TP 通信原语 -> MoE/EP -> Continuous Batching 与 KV Cache -> 单卡 P/D 衔接 -> PD 解耦与分离 -> Prefix Caching 执行路径`
- 备注：`EP` 与 `PD 解耦 vs PD 分离` 目前仍主要是基于现有条目做的工程归纳，后续可补独立概念页或来源

## [2026-04-23] ingest | 秋招CUDA手撕题复盘（附代码）

- 读取原始资料：`raw/articles/秋招CUDA手撕题复盘（附代码）.md`
- 创建来源页：`wiki/sources/秋招CUDA手撕题复盘（附代码）.md`
- 创建概念页：`wiki/concepts/Warp Shuffle Reduce.md`
- 创建概念页：`wiki/concepts/Block Reduce.md`
- 创建概念页：`wiki/concepts/Grid-stride Loop.md`
- 创建概念页：`wiki/concepts/RMSNorm.md`
- 创建概念页：`wiki/concepts/Histogram.md`
- 更新概念页：`wiki/concepts/CUDA Kernel.md`
- 更新概念页：`wiki/concepts/Online Softmax.md`
- 更新概念页：`wiki/concepts/Tiling.md`
- 未发现与现有 wiki 的直接冲突，本次主要把 CUDA 条目从“性能优化清单”推进到“面试高频 kernel 模板”视角，补入了 `Warp Shuffle Reduce / Block Reduce / Grid-stride Loop` 这组骨架概念

## [2026-04-22] ingest | Kimi新作《Attention Residuals》：对Transformer中残差结构的调整

- 读取原始资料：`raw/articles/Kimi新作《Attention Residuals》：对Transformer中残差结构的调整.md`
- 补充原始论文：`raw/papers/2603.15031v1.pdf`
- 创建来源页：`wiki/sources/Kimi新作《Attention Residuals》：对Transformer中残差结构的调整.md`
- 创建实体页：`wiki/entities/Moonshot AI.md`
- 创建概念页：`wiki/concepts/Attention Residuals.md`
- 创建概念页：`wiki/concepts/PreNorm Dilution.md`
- 更新概念页：`wiki/concepts/流水线并行.md`
- 更新概念页：`wiki/concepts/Scaling Laws.md`
- 未发现与现有 wiki 的直接冲突，本次主要补入了 `标准残差 -> Attention Residuals -> Block AttnRes` 这条跨层聚合改造路线，以及它和 pipeline communication / scaling-law 评估的联系

## [2026-04-23] query | 面试经验

- 生成输出文档：`output/reports/面试经验.md`
- 整理主题：`SFT`、`RLHF`、`MoE 路由`、`PPO`、`大模型架构对比`
- 本次输出以面试口径为主，强调“问题 -> 结论 -> 原因 -> 权衡”的答题结构
- 后续补充：新增 `RoPE` 的五层面试追问框架，覆盖机制理解、数学直觉、实现细节、系统影响与 research tradeoff
- 后续补充：新增简历高频八股速查，覆盖 `FlashAttention`、`算子融合`、`Nsight Systems / Nsight Compute`、基础模型架构、`KV cache` 与 `per-group quantization group size`
- 后续补充：新增 `CUDA Graphs` 速查，整理其性能原理、5 个经典坑，以及 `torch.compile(mode=\"reduce-overhead\")` 作为低成本入场方式

## [2026-04-22] ingest | mHC: Manifold-Constrained Hyper-Connections

- 读取原始资料：`raw/papers/2512.24880v2.pdf`
- 创建来源页：`wiki/sources/mHC: Manifold-Constrained Hyper-Connections.md`
- 创建实体页：`wiki/entities/DeepSeek-AI.md`
- 创建概念页：`wiki/concepts/Hyper-Connections.md`
- 创建概念页：`wiki/concepts/mHC.md`
- 更新概念页：`wiki/concepts/算子融合.md`
- 更新概念页：`wiki/concepts/重计算.md`
- 更新概念页：`wiki/concepts/流水线并行.md`
- 未发现与现有 wiki 的直接冲突，本次主要补入了 `HC -> mHC` 这条宏观拓扑设计主线，以及论文里的 kernel fusion / recomputing / pipeline overlap 系统实现视角

## [2026-04-13] bootstrap | 初始化 AI Infra LLM Wiki

- 建立 `raw/`、`wiki/`、`output/`、`scripts/` 基础结构
- 增加 `AGENTS.md` 作为 schema 规则文件
- 建立 `index.md` 与 `log.md` 作为导航层

## [2026-04-13] ingest | LLM推理优化核心技术

- 读取原始资料：`raw/articles/LLM推理优化核心技术.md`
- 创建来源页：`wiki/sources/LLM推理优化核心技术.md`
- 创建实体页：`wiki/entities/vLLM.md`
- 创建实体页：`wiki/entities/SGLang.md`
- 创建实体页：`wiki/entities/Nvidia Dynamo.md`
- 创建实体页：`wiki/entities/TensorRT-LLM.md`
- 创建概念页：`wiki/concepts/KV Cache.md`
- 创建概念页：`wiki/concepts/Prefix Caching.md`
- 创建概念页：`wiki/concepts/缓存感知路由.md`
- 创建概念页：`wiki/concepts/PagedAttention.md`
- 创建概念页：`wiki/concepts/PD分离.md`
- 创建概念页：`wiki/concepts/Tensor Parallelism.md`
- 未发现直接矛盾，`PD分离` 的适用边界标记为后续待验证

## [2026-04-13] ingest | Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构

- 读取原始资料：`raw/articles/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构.md`
- 创建来源页：`wiki/sources/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构.md`
- 创建实体页：`wiki/entities/Gemma 4.md`
- 创建实体页：`wiki/entities/Google DeepMind.md`
- 创建实体页：`wiki/entities/特里斯丹井底之娃 往上爬.md`
- 创建概念页：`wiki/concepts/Per-Layer Embeddings.md`
- 创建概念页：`wiki/concepts/Shared KV Cache.md`
- 创建概念页：`wiki/concepts/混合注意力.md`
- 创建概念页：`wiki/concepts/Dual RoPE.md`
- 创建概念页：`wiki/concepts/Double-Wide MLP.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 未发现直接矛盾，`Double-Wide MLP` 作为补偿机制的解释标记为后续待验证

## [2026-04-13] query | Gemma 开源代码结构导读

- 删除原始资料：`raw/articles/Google Gemma 4 深度解析：当开源AI进入「逐层嵌入平行化」时代-程序员茄子.md`
- 更新索引：`wiki/index.md`
- 运行健康检查：`scripts/lint.py`
- 生成查询报告：`output/reports/Gemma 开源代码结构导读.md`
- 本次查询以官方源码结构为主，重点核对 `google/gemma_pytorch` 与 `transformers/models/gemma4`

## [2026-04-13] ingest | 斯坦福CS336 Lecture 5 - GPUs

- 读取原始资料：`raw/articles/斯坦福CS336 Lecture 5 - GPUs.md`
- 创建来源页：`wiki/sources/斯坦福CS336 Lecture 5 - GPUs.md`
- 创建实体页：`wiki/entities/Stanford CS336.md`
- 创建概念页：`wiki/concepts/GPU执行模型.md`
- 创建概念页：`wiki/concepts/Roofline 模型.md`
- 创建概念页：`wiki/concepts/算子融合.md`
- 创建概念页：`wiki/concepts/重计算.md`
- 创建概念页：`wiki/concepts/内存合并访问.md`
- 创建概念页：`wiki/concepts/Tiling.md`
- 创建概念页：`wiki/concepts/FlashAttention.md`
- 标注说明：B 站镜像标题中的“分布式训练基础”范围比 Stanford 官方讲义 `Lecture 5 - GPUs` 更宽
- 未发现与现有 wiki 的直接冲突，后续建议补 `Lecture 7 - Parallelism basics` 与本讲组成前后衔接

## [2026-04-13] ingest | 斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing

- 读取原始资料：`raw/articles/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing.md`
- 创建来源页：`wiki/sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing.md`
- 创建概念页：`wiki/concepts/Benchmarking.md`
- 创建概念页：`wiki/concepts/Profiling.md`
- 创建概念页：`wiki/concepts/CUDA Kernel.md`
- 创建概念页：`wiki/concepts/Triton.md`
- 创建概念页：`wiki/concepts/torch.compile.md`
- 更新实体页：`wiki/entities/Stanford CS336.md`
- 更新概念页：`wiki/concepts/算子融合.md`
- 更新概念页：`wiki/concepts/Tiling.md`
- 未发现与现有 wiki 的直接冲突，Lecture 6 已形成对 Lecture 5 的自然延续

## [2026-04-13] ingest | 斯坦福CS336 Lecture 7 - Parallelism basics

- 读取原始资料：`raw/articles/斯坦福CS336 Lecture 7 - Parallelism basics.md`
- 创建来源页：`wiki/sources/斯坦福CS336 Lecture 7 - Parallelism basics.md`
- 创建概念页：`wiki/concepts/集合通信.md`
- 创建概念页：`wiki/concepts/数据并行.md`
- 创建概念页：`wiki/concepts/ZeRO.md`
- 创建概念页：`wiki/concepts/FSDP.md`
- 创建概念页：`wiki/concepts/流水线并行.md`
- 创建概念页：`wiki/concepts/Sequence Parallelism.md`
- 更新概念页：`wiki/concepts/Tensor Parallelism.md`
- 更新实体页：`wiki/entities/Stanford CS336.md`
- 未发现与现有 wiki 的直接冲突，Lecture 7 已把课程线推进到多机多卡训练并行

## [2026-04-13] ingest | 斯坦福CS336 Lecture 8 - Distributed communication and training code

- 读取原始资料：`raw/articles/斯坦福CS336 Lecture 8 - Distributed communication and training code.md`
- 创建来源页：`wiki/sources/斯坦福CS336 Lecture 8 - Distributed communication and training code.md`
- 创建实体页：`wiki/entities/NCCL.md`
- 创建概念页：`wiki/concepts/Torch Distributed.md`
- 更新概念页：`wiki/concepts/集合通信.md`
- 更新概念页：`wiki/concepts/数据并行.md`
- 更新概念页：`wiki/concepts/Tensor Parallelism.md`
- 更新概念页：`wiki/concepts/流水线并行.md`
- 更新实体页：`wiki/entities/Stanford CS336.md`
- 未发现与现有 wiki 的直接冲突，Lecture 8 已把并行原理推进到 `torch.distributed` 与通信骨架代码层

## [2026-04-13] ingest | 斯坦福CS336 Lecture 9 - Scaling laws basics

- 读取原始资料：`raw/articles/斯坦福CS336 Lecture 9 - Scaling laws basics.md`
- 创建来源页：`wiki/sources/斯坦福CS336 Lecture 9 - Scaling laws basics.md`
- 创建概念页：`wiki/concepts/Scaling Laws.md`
- 创建概念页：`wiki/concepts/数据缩放定律.md`
- 创建概念页：`wiki/concepts/Critical Batch Size.md`
- 创建概念页：`wiki/concepts/Chinchilla Scaling.md`
- 更新实体页：`wiki/entities/Stanford CS336.md`
- 未发现与现有 wiki 的直接冲突，Lecture 9 已把课程线从系统实现推进到模型设计与资源分配规律

## [2026-04-13] ingest | 斯坦福CS336 Lecture 10 - Inference systems and optimization

- 读取原始资料：`raw/articles/斯坦福CS336 Lecture 10 - Inference systems and optimization.md`
- 创建来源页：`wiki/sources/斯坦福CS336 Lecture 10 - Inference systems and optimization.md`
- 创建概念页：`wiki/concepts/Continuous Batching.md`
- 创建概念页：`wiki/concepts/Speculative Decoding.md`
- 更新概念页：`wiki/concepts/KV Cache.md`
- 更新概念页：`wiki/concepts/PagedAttention.md`
- 更新实体页：`wiki/entities/vLLM.md`
- 未发现与现有 wiki 的直接冲突，Lecture 10 已把课程线推进到推理系统与动态 workload 优化

## [2026-04-14] ingest | Flash Attention 详细解释推演与Pytorch代码实现

- 读取原始资料：`raw/articles/Flash Attention 详细解释推演与Pytorch代码实现.md`
- 创建来源页：`wiki/sources/Flash Attention 详细解释推演与Pytorch代码实现.md`
- 创建概念页：`wiki/concepts/Online Softmax.md`
- 更新概念页：`wiki/concepts/FlashAttention.md`
- 更新概念页：`wiki/concepts/Tiling.md`
- 未发现与现有 wiki 的直接冲突，本次主要补强了 `FlashAttention` 的 IO-aware 视角、`Online Softmax` 的状态合并逻辑，以及 `FA1/FA2` 的工作分配差异

## [2026-04-22] ingest | 十分钟读懂旋转编码（RoPE）

- 读取原始资料：`raw/articles/十分钟读懂旋转编码（RoPE）.md`
- 创建来源页：`wiki/sources/十分钟读懂旋转编码（RoPE）.md`
- 创建概念页：`wiki/concepts/RoPE.md`
- 更新概念页：`wiki/concepts/Dual RoPE.md`
- 未发现与现有 wiki 的直接冲突，本次主要补齐了通用 `RoPE` 概念锚点，并把现有 `Dual RoPE` 收束回更基础的位置编码语境

## [2026-04-22] ingest | 你一定要知道：CUDA优化六要

- 读取原始资料：`raw/articles/你一定要知道：CUDA优化六要.md`
- 创建来源页：`wiki/sources/你一定要知道：CUDA优化六要.md`
- 创建概念页：`wiki/concepts/Bank Conflict.md`
- 创建概念页：`wiki/concepts/Occupancy.md`
- 创建概念页：`wiki/concepts/Warp Divergence.md`
- 创建概念页：`wiki/concepts/Tail Effect.md`
- 更新概念页：`wiki/concepts/CUDA Kernel.md`
- 更新概念页：`wiki/concepts/GPU执行模型.md`
- 更新概念页：`wiki/concepts/内存合并访问.md`
- 更新概念页：`wiki/concepts/Tiling.md`
- 未发现与现有 wiki 的直接冲突，本次主要把 CUDA 优化经验条目整理成可复用的性能排障概念集

## [2026-04-22] ingest | Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

- 读取原始资料：`raw/papers/2601.07372v1.pdf`
- 创建来源页：`wiki/sources/Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models.md`
- 创建实体页：`wiki/entities/Engram.md`
- 创建概念页：`wiki/concepts/Conditional Memory.md`
- 创建概念页：`wiki/concepts/Sparsity Allocation.md`
- 未发现与现有 wiki 的直接冲突，本次主要补入了 `conditional memory` / `Engram` / `Sparsity Allocation` 这一条新的稀疏建模主线

## [2026-04-22] ingest | engram_demo_v1

- 读取原始资料：`raw/code/engram_demo_v1.py`
- 创建来源页：`wiki/sources/engram_demo_v1.md`
- 更新实体页：`wiki/entities/Engram.md`
- 更新概念页：`wiki/concepts/Conditional Memory.md`
- 未发现与现有 wiki 的直接冲突，本次主要把 `Engram` 从论文摘要推进到 demo 代码实现视角
