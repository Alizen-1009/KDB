# MoE 通信-计算重叠与 MegaKernel 技术谱系

## 说明

这份文档整理一次围绕 5 份资料的连续讨论，主题是：**LLM 推理/训练里如何把通信、计算、访存重叠起来提速**，以及 **MegaKernel（巨核）** 思路如何从 dense 推理一路演进到 MoE 训练。

风格遵循：先给结论 → 再讲机制 → 最后补 tradeoff。

相关概念可结合 [[../../wiki/concepts/Megakernel|Megakernel]]、[[../../wiki/concepts/MoE|MoE]]、[[../../wiki/concepts/Expert Parallelism|Expert Parallelism]]、[[../../wiki/concepts/算子融合|算子融合]]、[[../../wiki/concepts/Tensor Parallelism|Tensor Parallelism]]、[[../../wiki/concepts/集合通信|集合通信]]、[[../../wiki/concepts/确定性推理|确定性推理]]、[[../../wiki/concepts/Triton|Triton]] 一起看。

> 说明：本文涉及的 UniEP、Triton-distributed、DeepFusionKernel、NVIDIA Wide-EP 均晚于当前模型知识截止（2026-01），相关细节主要依据各自原文/博客描述，标注为**原文说法**；跨来源的归纳标注为**LLM 归纳**，存疑处标 **待核实**。

## 一张总览表：五份资料打在不同层

| 资料 | 层级 | 场景 | 核心手段 | 保数值一致 |
|---|---|---|---|---|
| DeepFusionKernel | 单算子 kernel | dense 推理 | 融合 SwiGLU（up/gate GEMM + SiLU + mul），减 HBM 读写 | — |
| Megakernel (HazyResearch) | 整模型 kernel | dense 推理 | 整个 forward 融进一个 kernel，SM 内/SM 间/GPU 间三级资源重叠 | — |
| Ladder-Residual (ICML 2025) | 模型架构 | dense 推理/训练 | 改残差连线（吃 stale 输入），让 TP 的 AllReduce 可与计算重叠 | 近似（表达漂移小） |
| NVIDIA Wide-EP | 系统/机架 | **MoE 推理** | 宽专家并行 + EPLB 负载均衡 + NVL72 高带宽域 | — |
| UniEP (ByteDance+清华, 2026) | kernel+系统 | **MoE 训练** | 把 MoE EP 的 Dispatch/Combine 融进 MegaKernel，SM 级动态调度重叠 | ✅ 严格 bitwise |

两条演进主线（**LLM 归纳**）：
- **dense → MoE**：优化重心从"减少 dense SwiGLU 的中间激活读写"转到"MoE 专家并行的 All-to-All / GroupGEMM / 专家权重加载"。
- **推理 → 训练**：MegaKernel 最初用于低延迟推理，UniEP 首次把它搬到训练，并额外解决反向传播与数值一致性。

---

## 各来源速览

### 1. DeepFusionKernel（Zhang 等, 2026, arXiv:2602.11808）
- **一句话**：把 dense Transformer 的 SwiGLU MLP 深度融合成单个 CUDA kernel，把 SiLU+Mul 融进 up/gate GEMM 的 epilogue，避免中间激活 A_gate/A_1 落 HBM。
- **收益**：集成进 SGLang，H100 最高 +13.2%、A100 最高 +9.7%（中位数约 4–5%，方差大）。
- **只融第一阶段** `A₂=(XW_Up)⊗SiLU(XW_Gate)`，第二阶段 `Y=A₂W_Down` 仍单独 kernel。配 profile 驱动的 tiling 调度器（row-major 利于激活复用、column-major 利于权重复用）。
- **局限（LLM 归纳）**：本质≈GEMM+激活的 epilogue fusion，较增量；且 dense-only，MoE 侧的 `fused_moe` 早已把激活融进 grouped GEMM，因此对 MoE 意义有限。

### 2. Megakernel / "We Bought the Whole GPU"（HazyResearch, 2025-09-28, 博客）
- **一句话**：把 Llama-70B 张量并行推理的整个 forward 融成一个 MegaKernel，用"指令+解释器"抽象在三个层次同时压榨 GPU 资源。
- **三级重叠**：SM 内（warp 专职化 loader/consumer/storer + 指令流水）、SM 间（全局工作队列 GWQ 吸收 jitter）、GPU 间（storer 线程后台跑通信，用 ThunderKittens 的 PGL 直接读写远端显存，**甩掉 NCCL**）。
- **两个技巧**：用"distributed transpose"把 O-projection 从 TP 改数据并行以减 8× 网络流量；interleaving 把不同类型指令交织进 GWQ（比 NanoFlow 预分组更细）。
- **结果**：集成进 Tokasaurus，ShareGPT 端到端比 [[../../wiki/entities/SGLang|SGLang]] 快 22%。相关实体见 [[../../wiki/entities/HazyResearch|HazyResearch]]、[[../../wiki/entities/Megakernels|Megakernels]]。

### 3. Ladder-Residual（Muru Zhang 等含 Tri Dao, ICML 2025, arXiv:2501.06589）
- **一句话**：TP 每层两次阻塞 AllReduce 占 70B/TP=8 约 38% 延迟；改残差连线让模块吃"过期一拍"的输入 `x_{i+1}=h_{i+1}(x_{i-1})+x_i`，使计算不依赖上一次 AllReduce 结果，从而可重叠。
- **依据**：Deja Vu 观察——每层更新范数相对残差流很小，喂 stale 输入不显著伤表达。
- **收益**：70B 提速 ~29%（P2P 开）/ ~59%（P2P 关）；405B 跨节点 TP=16 仍 >30%。纯 PyTorch/JAX 实现，**不写底层 kernel、跨硬件通用**。
- **精度**：从头训 1.2B/3.5B 基本追平（3.5B 略逊）；Llama-3.1-8B 改上半层 + 3B token 轻量微调即追平，快 21%。
- **对比点（LLM 归纳）**：Ladder 从**架构层**换取通信重叠；HazyResearch megakernel 从**kernel 层**做到同样的通信-计算重叠——殊途同归。关联 [[../../wiki/concepts/Tensor Parallelism|Tensor Parallelism]]、[[../../wiki/concepts/Sequence Parallelism|Sequence Parallelism]]、[[../../wiki/concepts/集合通信|集合通信]]。

### 4. NVIDIA Wide-EP on NVL72（NVIDIA 博客, 2025-10）
- **一句话**：在 GB200 NVL72 机架上，用 TensorRT-LLM 的 Wide Expert Parallelism 把 MoE 专家摊到 8+ 甚至几十张 GPU，配负载均衡与定制通信 kernel，DeepSeek-R1 每 GPU 吞吐最高 1.8×。
- **机制**：卡多→每卡专家少→权重加载压力小、GroupGEMM 算术强度高。新问题（all-to-all、动态通信尺寸、热专家扎堆）分别用 NVL72 的 130 TB/s 带宽、定制 NCCL kernel、**EPLB（专家并行负载均衡，static/online）** 解决；权重迁移容器化、不打断 CUDA Graph。
- **配套**：[[../../wiki/entities/Nvidia Dynamo|NVIDIA Dynamo]]（编排/PD 分离/SLA 扩缩）+ TensorRT-LLM Wide-EP（执行引擎）。关联 [[../../wiki/entities/NCCL|NCCL]]、[[../../wiki/entities/TensorRT-LLM|TensorRT-LLM]]、[[../../wiki/concepts/PD分离|PD分离]]、[[../../wiki/concepts/Multi-Token Prediction|Multi-Token Prediction]]。

### 5. UniEP（Size Zheng 等, ByteDance Seed + 清华, 2026, arXiv:2604.19241）
- **一句话**：把 MoE EP 的 `Dispatch+GroupGEMM` 和 `GroupGEMM+Combine` 融进单 stream 的 MegaKernel，SM 级动态调度实现细粒度通信-计算重叠，同时用确定性 token 排序保证 bitwise 一致；Hopper 上比 COMET 快 1.03–1.38×。
- **持久化 Worker**：SM 数量的持久 threadblock，动态扮演 Comm/Comp/Relay(Reduce) 三种角色；**Scoreboard 记分板**做 token/tile 级生产者-消费者同步；全局原子计数器动态负载均衡。
- **确定性 token 映射**（Algorithm 1：AllGather 专家计数 → 前缀和算全局 offset）→ 反向 Transposed GroupGEMM 累加顺序固定 → **不切 micro-batch** → 与串行逐位一致（COMET 有 22–29% 元素不一致）。
- **Relay Worker 带宽优化**：top-8 路由平均只需发到 5.25 个不同 rank（省 ~34% NVLink），一次发到目标 rank 后本地 HBM 复制。
- **工程**：基于 [[#Triton-distributed 是什么|Triton-distributed]]（约 21k 行 Python）+ 解析性能模型自动调优（~10⁵ 配置，C++/OpenMP 144ms 搜完）。128 GPU 生产训练 127B→138B tokens/天（1.09×）且保持 bitwise。
- **代价**：为 bitwise，反向比放松约束的非 bitwise 版慢 2–8%。关联 [[../../wiki/concepts/确定性推理|确定性推理]]、[[../../wiki/concepts/混合精度训练与推理|混合精度训练与推理]]。

---

## 讨论沉淀（概念澄清）

### SwiGLU 与 MoE 不是二选一
- **SwiGLU** 是 FFN/MLP 内部结构（`(XW_Up)⊗SiLU(XW_Gate)` 再 `W_Down`）；**MoE** 是 FFN 这一层怎么组织（多专家 + 路由）。
- **MoE 的每个专家本身几乎就是一个 SwiGLU FFN**——MoE 没取代 SwiGLU，只是把"一个 SwiGLU"变成"一堆 SwiGLU + 路由器"。dense（Llama/Qwen dense）和 MoE（DeepSeek-V3、Qwen3-MoE、Mixtral、Llama 4）都在用 SwiGLU。
- 参见 [[../../wiki/concepts/MoE|MoE]]、[[../../wiki/concepts/Double-Wide MLP|Double-Wide MLP]]。

### 为什么 dense 的 SwiGLU 融合对 MoE 意义有限
- MoE 的真瓶颈不是"中间激活读写"，而是**激活专家的权重加载 + grouped GEMM 调度 + all-to-all 通信**。
- vLLM/SGLang 的 `fused_moe`（Triton grouped-GEMM 内核）**已把 SiLU+Mul 融进第一段 grouped GEMM 的 epilogue**——即 DeepFusionKernel 对 dense 补的那个洞，MoE 侧早已默认做了。**待核实**：不同框架/版本 fused_moe 的融合边界略有差异。

### 为什么通信能和计算重叠（"dispatch 没完怎么就能算"）
关键：**依赖是细粒度的，不是"全部→全部"**。
- dispatch 内部是一串小传输（每 token/每 warp 独立），token **陆续到达**；GroupGEMM 按 tile（如每 128 token）分块算。
- `tile_i` 的计算只依赖路由到它的那批 token，不关心别的 tile 的 token 到没到。所以**某个 tile 的 token 先到齐就先算它，剩下的 token 还在网络上飞** → 同一瞬间部分 SM 发 token（占 NVLink）、部分 SM 算已到齐的 tile（占 Tensor Core），两种硬件资源同时忙 = 重叠。
- **Scoreboard** 负责"到齐了没"的握手（token 到达标记 → 凑够一 tile 置 ready → Comp-Worker 轮询开算），全在 GPU 内、无 CPU 介入。
- **priority-based scheduling**：让通信发送顺序对齐计算消费顺序（计算按 expert0→1→2，通信就优先发 expert0 的 token），避免 head-of-line blocking。
- 类比：青菜一到就先炒，采购员还在去拿肉的路上;不是等所有食材到齐再统一开火。
- 另一种更粗的重叠是跨 micro-batch（DeepSeek-V3 用下一批通信盖当前批计算），但会改梯度累加顺序、丢 bitwise——UniEP 特意只用层内 tile 级细粒度重叠。参见 [[../../wiki/concepts/集合通信|集合通信]]、[[../../wiki/concepts/持久批处理|持久批处理]]。

### Triton-distributed 是什么
- ByteDance 对 [[../../wiki/concepts/Triton|Triton]] 编译器的扩展（arXiv:2504.19442，作者同为 Size Zheng），用于在**多 GPU/多节点上写"通信-计算重叠"kernel**。原版 Triton 只管单 GPU。
- 补三样能力：**原生 NVSHMEM 支持**（kernel 内直接读写远端 GPU 显存，GPU 发起 P2P、不经 CPU）、**内存语义信号**（`ld_acquire`/`st_release`，是 scoreboard 跨 SM 正确同步的基础）、**warp 级控制**（`warp_id` + warp 内同步）。
- 生态位：与 CuTeDSL/CuTile、MSCCL++、TileLang 同类；差异在于长在 Triton 上、支持 AMD。UniEP、TileLink 都基于它构建。开源：`github.com/ByteDance-Seed/Triton-distributed`。**待核实**：其确切 API 与最新能力以 arXiv/GitHub 为准。

### UniEP 能否用于推理
- **方法论能**：单 stream MegaKernel 重叠 + Relay multicast 对 MoE 推理 decode 同样成立，且 decode 更通信/带宽受限、收益可能更大。
- **这个实现不宜直接用**：(1) 近半篇幅在优化反向/Transposed GroupGEMM，推理无反向；(2) 卖点 bitwise 对推理基本无用，推理反而该放开约束更激进重叠；(3) tile/warp/性能模型按训练大 batch 长 seq 拟合，需按 decode 小 batch 重调；(4) 未处理 prefill/decode 分离、KV cache、连续批处理。
- 同团队的 MegaScale-Infer（arXiv:2504.02263）已在推理侧做 disaggregated expert parallelism。

---

## 待核实 / 可继续研究
- 各框架 `fused_moe` 的融合边界（是否融第二段 down GEMM）。
- UniEP / Triton-distributed 的 API 与后续版本能力（晚于知识截止）。
- Ladder-Residual 在 MoE 架构上的效果（原文只做 dense）。
- 是否值得为本谱系新建独立 wiki 概念/实体页（见下）。

## 后续 ingest 建议（尚未落盘）
若要做完整 ingest，建议按 schema 新建/更新：
- 概念页：`通信-计算重叠`、`Scoreboard 同步`、`GroupGEMM`、`Deterministic Token Ordering`
- 实体页：`UniEP`、`Triton-distributed`、`DeepFusionKernel`、`Ladder-Residual`、`COMET`、`DeepEP`、`ByteDance Seed`
- 更新现有页：[[../../wiki/concepts/MoE|MoE]]、[[../../wiki/concepts/Expert Parallelism|Expert Parallelism]]、[[../../wiki/concepts/Megakernel|Megakernel]]、[[../../wiki/concepts/算子融合|算子融合]]

## 相关已有笔记（Clippings，主题邻近）
- [[性能相比SGLangvLLM最高提升1.7倍！Mirage Persistent Kernel：首个自动巨核化多GPU LLM推理的编译器-运行时系统，细粒度计算-通信重叠|Mirage Persistent Kernel]]
- [[AutoMegaKernel：Agent驱动、跨GPU架构的MegaKernel方案，静态校验全模型单巨内核，最高提速 1.33 倍！|AutoMegaKernel]]
- [[MoE 所有层融到一个分布式算子GPU Kernel！FlashDMoE：GPU内核-硬件协同解锁大规模分布式机器学习性能极限！|FlashDMoE]]
- [[超越 vLLM 与 SGLang！Event Tensor：以动态 MegaKernel 消除重编译，解锁GPU核间通信-计算重叠|Event Tensor]]
