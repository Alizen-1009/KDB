# AI Infra 知识库索引

该页面由 `scripts/update_index.py` 自动生成。分组依据是各页面 frontmatter 的 `topic` / `entity_type` / `source_kind`。

## 入口

- [[../00 Home|Vault Home]]
- [[../AGENTS|AGENTS]]
- [[log|操作日志]]
- [[../inbox/README|Inbox]]
- [[../output/README|Output]]

## 主题地图

- [[maps/注意力机制|注意力机制]]
- [[maps/KV Cache|KV Cache]]
- [[maps/推理服务|推理服务]]
- [[maps/并行与分布式|并行与分布式]]
- [[maps/GPU 编程|GPU 编程]]
- [[maps/性能分析|性能分析]]
- [[maps/模型架构|模型架构]]
- [[maps/投机解码|投机解码]]
- [[maps/训练与 Scaling|训练与 Scaling]]
- [[maps/位置编码|位置编码]]

## 资源统计

- 原始文章：69
- 原始论文：5
- 原始仓库：0
- 原始数据集：0
- 原始图片：19
- 原始代码：1
- 来源文件：55
- 实体文件：41
- 概念文件：108
- 报告文件：32
- HTML 导出：0
- 面试文件：11
- 卡片文件：0
- 幻灯片文件：0

## 概念页面（按主题）

### 注意力机制（11）

- [[../wiki/concepts/CSA-HCA|CSA-HCA]]
- [[../wiki/concepts/Chunked Gated Delta Rule|Chunked Gated Delta Rule]]
- [[../wiki/concepts/Flash Decoding|Flash Decoding]]
- [[../wiki/concepts/FlashAttention|FlashAttention]]
- [[../wiki/concepts/FlashMLA|FlashMLA]]
- [[../wiki/concepts/KDA|KDA]]
- [[../wiki/concepts/MLA|MLA]]
- [[../wiki/concepts/Online Softmax|Online Softmax]]
- [[../wiki/concepts/Ring Attention|Ring Attention]]
- [[../wiki/concepts/混合注意力|混合注意力]]
- [[../wiki/concepts/线性注意力递归状态|线性注意力递归状态]]

### KV Cache（8）

- [[../wiki/concepts/KV Cache|KV Cache]]
- [[../wiki/concepts/PagedAttention|PagedAttention]]
- [[../wiki/concepts/Prefix Caching|Prefix Caching]]
- [[../wiki/concepts/RadixAttention|RadixAttention]]
- [[../wiki/concepts/Shared KV Cache|Shared KV Cache]]
- [[../wiki/concepts/分层 KV Cache|分层 KV Cache]]
- [[../wiki/concepts/缓存感知路由|缓存感知路由]]
- [[../wiki/concepts/递归状态 Prefix Caching|递归状态 Prefix Caching]]

### 推理服务（15）

- [[../wiki/concepts/Attention-FFN 分离|Attention-FFN 分离]]
- [[../wiki/concepts/Chunked Prefill|Chunked Prefill]]
- [[../wiki/concepts/Constrained Decoding|Constrained Decoding]]
- [[../wiki/concepts/Context Folding|Context Folding]]
- [[../wiki/concepts/Continuous Batching|Continuous Batching]]
- [[../wiki/concepts/LLM Programs|LLM Programs]]
- [[../wiki/concepts/Model Context Protocol|Model Context Protocol]]
- [[../wiki/concepts/PD分离|PD分离]]
- [[../wiki/concepts/Recursive Language Model|Recursive Language Model]]
- [[../wiki/concepts/SGLang 与 vLLM 对比|SGLang 与 vLLM 对比]]
- [[../wiki/concepts/Sandbox|Sandbox]]
- [[../wiki/concepts/vLLM V1 统一调度器|vLLM V1 统一调度器]]
- [[../wiki/concepts/可插拔 Decode 引擎|可插拔 Decode 引擎]]
- [[../wiki/concepts/持久批处理|持久批处理]]
- [[../wiki/concepts/确定性推理|确定性推理]]

### 并行与分布式（19）

- [[../wiki/concepts/DDP|DDP]]
- [[../wiki/concepts/DP Attention|DP Attention]]
- [[../wiki/concepts/Decode Context Parallel|Decode Context Parallel]]
- [[../wiki/concepts/DeepSpeed Ulysses|DeepSpeed Ulysses]]
- [[../wiki/concepts/Dual Batch Overlap|Dual Batch Overlap]]
- [[../wiki/concepts/Expert Parallel Load Balancing|Expert Parallel Load Balancing]]
- [[../wiki/concepts/Expert Parallelism|Expert Parallelism]]
- [[../wiki/concepts/FSDP|FSDP]]
- [[../wiki/concepts/Prefill Context Parallel|Prefill Context Parallel]]
- [[../wiki/concepts/Sequence Parallelism|Sequence Parallelism]]
- [[../wiki/concepts/Tensor Parallelism|Tensor Parallelism]]
- [[../wiki/concepts/Torch Distributed|Torch Distributed]]
- [[../wiki/concepts/Wide Expert Parallelism|Wide Expert Parallelism]]
- [[../wiki/concepts/ZeRO|ZeRO]]
- [[../wiki/concepts/数据并行|数据并行]]
- [[../wiki/concepts/流水线并行|流水线并行]]
- [[../wiki/concepts/跨 Mesh 权重重分片|跨 Mesh 权重重分片]]
- [[../wiki/concepts/通信-计算重叠|通信-计算重叠]]
- [[../wiki/concepts/集合通信|集合通信]]

### GPU 编程（26）

- [[../wiki/concepts/Bank Conflict|Bank Conflict]]
- [[../wiki/concepts/Block Reduce|Block Reduce]]
- [[../wiki/concepts/CODA|CODA]]
- [[../wiki/concepts/CUDA Graph 执行模式|CUDA Graph 执行模式]]
- [[../wiki/concepts/CUDA Kernel|CUDA Kernel]]
- [[../wiki/concepts/CUDA内存层次|CUDA内存层次]]
- [[../wiki/concepts/Cluster Launch Control|Cluster Launch Control]]
- [[../wiki/concepts/CuTe DSL|CuTe DSL]]
- [[../wiki/concepts/GPU执行模型|GPU执行模型]]
- [[../wiki/concepts/Grid-stride Loop|Grid-stride Loop]]
- [[../wiki/concepts/Histogram|Histogram]]
- [[../wiki/concepts/MegaMoE|MegaMoE]]
- [[../wiki/concepts/Megakernel|Megakernel]]
- [[../wiki/concepts/Occupancy|Occupancy]]
- [[../wiki/concepts/Persistent Kernel|Persistent Kernel]]
- [[../wiki/concepts/Programmatic Dependent Launch|Programmatic Dependent Launch]]
- [[../wiki/concepts/Tail Effect|Tail Effect]]
- [[../wiki/concepts/Tensor Memory|Tensor Memory]]
- [[../wiki/concepts/Tiling|Tiling]]
- [[../wiki/concepts/Torch Compile|Torch Compile]]
- [[../wiki/concepts/Triton|Triton]]
- [[../wiki/concepts/Warp Divergence|Warp Divergence]]
- [[../wiki/concepts/Warp Shuffle Reduce|Warp Shuffle Reduce]]
- [[../wiki/concepts/内存合并访问|内存合并访问]]
- [[../wiki/concepts/动态共享内存|动态共享内存]]
- [[../wiki/concepts/算子融合|算子融合]]

### 性能分析（3）

- [[../wiki/concepts/Benchmarking|Benchmarking]]
- [[../wiki/concepts/Profiling|Profiling]]
- [[../wiki/concepts/Roofline 模型|Roofline 模型]]

### 模型架构（11）

- [[../wiki/concepts/Attention Residuals|Attention Residuals]]
- [[../wiki/concepts/Conditional Memory|Conditional Memory]]
- [[../wiki/concepts/Double-Wide MLP|Double-Wide MLP]]
- [[../wiki/concepts/Hyper-Connections|Hyper-Connections]]
- [[../wiki/concepts/LatentMoE|LatentMoE]]
- [[../wiki/concepts/MoE|MoE]]
- [[../wiki/concepts/Per-Layer Embeddings|Per-Layer Embeddings]]
- [[../wiki/concepts/PreNorm Dilution|PreNorm Dilution]]
- [[../wiki/concepts/RMSNorm|RMSNorm]]
- [[../wiki/concepts/Sparsity Allocation|Sparsity Allocation]]
- [[../wiki/concepts/mHC|mHC]]

### 投机解码（6）

- [[../wiki/concepts/DFlash|DFlash]]
- [[../wiki/concepts/DSpark|DSpark]]
- [[../wiki/concepts/MTP Drafter|MTP Drafter]]
- [[../wiki/concepts/Multi-Token Prediction|Multi-Token Prediction]]
- [[../wiki/concepts/Speculative Decoding|Speculative Decoding]]
- [[../wiki/concepts/并行投机解码|并行投机解码]]

### 训练与 Scaling（6）

- [[../wiki/concepts/Chinchilla Scaling|Chinchilla Scaling]]
- [[../wiki/concepts/Critical Batch Size|Critical Batch Size]]
- [[../wiki/concepts/Scaling Laws|Scaling Laws]]
- [[../wiki/concepts/数据缩放定律|数据缩放定律]]
- [[../wiki/concepts/混合精度训练与推理|混合精度训练与推理]]
- [[../wiki/concepts/重计算|重计算]]

### 位置编码（3）

- [[../wiki/concepts/Dual RoPE|Dual RoPE]]
- [[../wiki/concepts/M-RoPE|M-RoPE]]
- [[../wiki/concepts/RoPE|RoPE]]


## 实体页面（按类型）

### 项目（9）

- [[../wiki/entities/CAKE KDA|CAKE KDA]]
- [[../wiki/entities/DeepEP|DeepEP]]
- [[../wiki/entities/Engram|Engram]]
- [[../wiki/entities/FlashKDA|FlashKDA]]
- [[../wiki/entities/Megakernels|Megakernels]]
- [[../wiki/entities/MoonEP|MoonEP]]
- [[../wiki/entities/NCCL Extensions|NCCL Extensions]]
- [[../wiki/entities/vLLM AFD Plugin|vLLM AFD Plugin]]
- [[../wiki/entities/verifiers|verifiers]]

### 框架（8）

- [[../wiki/entities/FlashInfer|FlashInfer]]
- [[../wiki/entities/NCCL|NCCL]]
- [[../wiki/entities/Nvidia Dynamo|Nvidia Dynamo]]
- [[../wiki/entities/RTP-LLM|RTP-LLM]]
- [[../wiki/entities/SGLang|SGLang]]
- [[../wiki/entities/TensorRT-LLM|TensorRT-LLM]]
- [[../wiki/entities/TileRT|TileRT]]
- [[../wiki/entities/vLLM|vLLM]]

### 模型（4）

- [[../wiki/entities/DeepSeek V4|DeepSeek V4]]
- [[../wiki/entities/Gemma 4|Gemma 4]]
- [[../wiki/entities/Kimi K3|Kimi K3]]
- [[../wiki/entities/Qwen VL|Qwen VL]]

### 公司（5）

- [[../wiki/entities/DeepSeek-AI|DeepSeek-AI]]
- [[../wiki/entities/Google DeepMind|Google DeepMind]]
- [[../wiki/entities/Moonshot AI|Moonshot AI]]
- [[../wiki/entities/Prime Intellect|Prime Intellect]]
- [[../wiki/entities/阿里巴巴|阿里巴巴]]

### 组织（4）

- [[../wiki/entities/Colfax Research|Colfax Research]]
- [[../wiki/entities/HazyResearch|HazyResearch]]
- [[../wiki/entities/vLLM Team|vLLM Team]]
- [[../wiki/entities/阿里云 PAI 团队|阿里云 PAI 团队]]

### 人物（6）

- [[../wiki/entities/kaiyuan|kaiyuan]]
- [[../wiki/entities/kason_zhang|kason_zhang]]
- [[../wiki/entities/方佳瑞|方佳瑞]]
- [[../wiki/entities/梦初AI Infra|梦初AI Infra]]
- [[../wiki/entities/特里斯丹井底之娃 往上爬|特里斯丹井底之娃 往上爬]]
- [[../wiki/entities/陈巍|陈巍]]

### 课程（1）

- [[../wiki/entities/Stanford CS336|Stanford CS336]]

### 硬件（4）

- [[../wiki/entities/NVIDIA Ampere|NVIDIA Ampere]]
- [[../wiki/entities/NVIDIA Blackwell|NVIDIA Blackwell]]
- [[../wiki/entities/NVIDIA Hopper|NVIDIA Hopper]]
- [[../wiki/entities/NVIDIA Rubin|NVIDIA Rubin]]


## 来源摘要（按类型）

### 文章（36）

- [[../wiki/sources/2026 年MoE 架构正在发生一次关键变化|2026 年MoE 架构正在发生一次关键变化]]
- [[../wiki/sources/A Preview of Production-Scale Kimi K3 Support on vLLM|A Preview of Production-Scale Kimi K3 Support on vLLM]]
- [[../wiki/sources/CUDA优化维度框架|CUDA优化维度框架]]
- [[../wiki/sources/DeepSeekV4中RoPE设计解析|DeepSeekV4中RoPE设计解析]]
- [[../wiki/sources/Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs|Dynamic persistent tile scheduling with Cluster Launch Control (CLC) on NVIDIA Blackwell GPUs]]
- [[../wiki/sources/Flash Attention 详细解释推演与Pytorch代码实现|Flash Attention 详细解释推演与Pytorch代码实现]]
- [[../wiki/sources/Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构|Gemma 4 核心技术深度解析：PLE、Shared KV Cache 与全模态架构]]
- [[../wiki/sources/Gemma 4：Drafter 详解|Gemma 4：Drafter 详解]]
- [[../wiki/sources/Kimi新作《Attention Residuals》：对Transformer中残差结构的调整|Kimi新作《Attention Residuals》：对Transformer中残差结构的调整]]
- [[../wiki/sources/LLM推理优化核心技术|LLM推理优化核心技术]]
- [[../wiki/sources/LLM提速利器：投机推理的原理与常见方案|LLM提速利器：投机推理的原理与常见方案]]
- [[../wiki/sources/MegaMoE — 让 all-to-all 消失|MegaMoE — 让 all-to-all 消失]]
- [[../wiki/sources/Model Runner V2 A Modular and Faster Core for vLLM|Model Runner V2 A Modular and Faster Core for vLLM]]
- [[../wiki/sources/NVIDIA 开源 NCCL Extensions：把 MoE 专家路由与跨 Mesh 权重重分片推进到 GPU 设备侧|NVIDIA 开源 NCCL Extensions：把 MoE 专家路由与跨 Mesh 权重重分片推进到 GPU 设备侧]]
- [[../wiki/sources/Nvidia Rubin架构分析预览|Nvidia Rubin架构分析预览]]
- [[../wiki/sources/PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化|PAI-FA｜突破 TMEM 瓶颈：FlashAttention-4 大 Head Dimension (256) 高性能算子实现与优化]]
- [[../wiki/sources/PageAttention代码走读|PageAttention代码走读]]
- [[../wiki/sources/REMINDER FF-KDA & CAKE KDA Highlights|REMINDER FF-KDA & CAKE KDA Highlights]]
- [[../wiki/sources/RTP-LLM|RTP-LLM]]
- [[../wiki/sources/Recursive Language Models the paradigm of 2026|Recursive Language Models the paradigm of 2026]]
- [[../wiki/sources/SGLang的KDA管理与Prefix Cache难题|SGLang的KDA管理与Prefix Cache难题]]
- [[../wiki/sources/SGLang：LLM推理引擎发展新方向|SGLang：LLM推理引擎发展新方向]]
- [[../wiki/sources/vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署|vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署]]
- [[../wiki/sources/vLLM Large Scale Serving DeepSeek @ 2.2k toksH200 with Wide-EP|vLLM Large Scale Serving DeepSeek @ 2.2k toksH200 with Wide-EP]]
- [[../wiki/sources/vLLM x TileRT Specialized Decode for Latency-Critical Serving|vLLM x TileRT Specialized Decode for Latency-Critical Serving]]
- [[../wiki/sources/vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现|vLLM皇冠上的明珠：深入浅出理解PagedAttention CUDA实现]]
- [[../wiki/sources/vllm PCP 与 DCP 深度解析|vllm PCP 与 DCP 深度解析]]
- [[../wiki/sources/vllm并行策略之DCP(Decode Context Parallel)|vllm并行策略之DCP(Decode Context Parallel)]]
- [[../wiki/sources/你一定要知道：CUDA优化六要|你一定要知道：CUDA优化六要]]
- [[../wiki/sources/十分钟读懂旋转编码（RoPE）|十分钟读懂旋转编码（RoPE）]]
- [[../wiki/sources/并行投机解码(DFlashDSpark)的快速理解与vLLM实测|并行投机解码(DFlashDSpark)的快速理解与vLLM实测]]
- [[../wiki/sources/彻底搞懂RoPE计算原理：从1D到3D|彻底搞懂RoPE计算原理：从1D到3D]]
- [[../wiki/sources/推理的非确定性运算及vLLMSGLang控制方式|推理的非确定性运算及vLLMSGLang控制方式]]
- [[../wiki/sources/译 NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell|译 NVIDIA’s GPUs - 从 Ampere, Hopper 到 Blackwell]]
- [[../wiki/sources/还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速|还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速]]
- [[../wiki/sources/陈巍：DeepSeek 开源Day（1）-FlashMLA 深入分析（收录于：DeepSeek技术详解系列）|陈巍：DeepSeek 开源Day（1）-FlashMLA 深入分析（收录于：DeepSeek技术详解系列）]]

### 论文（4）

- [[../wiki/sources/Attention Residuals|Attention Residuals]]
- [[../wiki/sources/Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models|Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models]]
- [[../wiki/sources/Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B|Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B]]
- [[../wiki/sources/mHC: Manifold-Constrained Hyper-Connections|mHC: Manifold-Constrained Hyper-Connections]]

### 课程（6）

- [[../wiki/sources/斯坦福CS336 Lecture 10 - Inference systems and optimization|斯坦福CS336 Lecture 10 - Inference systems and optimization]]
- [[../wiki/sources/斯坦福CS336 Lecture 5 - GPUs|斯坦福CS336 Lecture 5 - GPUs]]
- [[../wiki/sources/斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing|斯坦福CS336 Lecture 6 - Benchmarking, Profiling, and Kernel Writing]]
- [[../wiki/sources/斯坦福CS336 Lecture 7 - Parallelism basics|斯坦福CS336 Lecture 7 - Parallelism basics]]
- [[../wiki/sources/斯坦福CS336 Lecture 8 - Distributed communication and training code|斯坦福CS336 Lecture 8 - Distributed communication and training code]]
- [[../wiki/sources/斯坦福CS336 Lecture 9 - Scaling laws basics|斯坦福CS336 Lecture 9 - Scaling laws basics]]

### 代码（1）

- [[../wiki/sources/engram_demo_v1|engram_demo_v1]]

### 面试整理（6）

- [[../wiki/sources/CUDA内存层次与动态共享内存问答整理|CUDA内存层次与动态共享内存问答整理]]
- [[../wiki/sources/MLA与DP Attention面试整理|MLA与DP Attention面试整理]]
- [[../wiki/sources/多卡GPU监控与SM执行模型面试整理|多卡GPU监控与SM执行模型面试整理]]
- [[../wiki/sources/秋招CUDA手撕题复盘（附代码）|秋招CUDA手撕题复盘（附代码）]]
- [[../wiki/sources/美团一面：请介绍 vLLM PageAttention|美团一面：请介绍 vLLM PageAttention]]
- [[../wiki/sources/量化剪枝推理瓶颈Nsight与异构集群面试整理|量化剪枝推理瓶颈Nsight与异构集群面试整理]]

### 截图整理（2）

- [[../wiki/sources/SGLang 与 vLLM 区别截图整理|SGLang 与 vLLM 区别截图整理]]
- [[../wiki/sources/vLLM v0 与 vLLM v1 调度架构差异截图整理|vLLM v0 与 vLLM v1 调度架构差异截图整理]]


## 最近日志

- [2026-06-22] query | Split-KV 在现代 decode attention 中是否常见
- [2026-06-22] query | DP Attention 概念复查
- [2026-06-22] query | PD 分离与 Chunked Prefill 对 TPOT 的作用
- [2026-06-22] query | MTP 推理验证规则
- [2026-06-22] query | DeepSeek-V3 MTP 实现口径
- [2026-06-22] query | DCP 和 Flash Decoding 区别复查
- [2026-06-16] query | MHA/GQA/MLA prefill 计算量对比
- [2026-06-15] query | CuTe DSL 概念补页

## 报告

- [[../output/reports/Blackwell相对Hopper的新特性|Blackwell相对Hopper的新特性]]
- [[../output/reports/DCP是什么|DCP是什么]]
- [DFlash与DSpark投机解码详解](../output/reports/DFlash与DSpark投机解码详解.html)
- [[../output/reports/DeepSpeed Ulysses适用场景与DeepSeek关系|DeepSpeed Ulysses适用场景与DeepSeek关系]]
- [[../output/reports/FlashKDA为什么能并行|FlashKDA为什么能并行]]
- [[../output/reports/FlashKDA优化方法与GDN迁移指南|FlashKDA优化方法与GDN迁移指南]]
- [[../output/reports/Fused MoE NVFP4 v1-v5优化复盘|Fused MoE NVFP4 v1-v5优化复盘]]
- [[../output/reports/GDN公式与逐步计算|GDN公式与逐步计算]]
- [[../output/reports/Gemma 开源代码结构导读|Gemma 开源代码结构导读]]
- [[../output/reports/Hopper架构变化与Persistent Kernel|Hopper架构变化与Persistent Kernel]]
- [[../output/reports/KDA伪代码与输入输出|KDA伪代码与输入输出]]
- [[../output/reports/KDA投影融合优化|KDA投影融合优化]]
- [[../output/reports/KDA最小Decode伪代码|KDA最小Decode伪代码]]
- [[../output/reports/KDA相对GDN的改进|KDA相对GDN的改进]]
- [[../output/reports/Kimi K3为何采用TP8部署|Kimi K3为何采用TP8部署]]
- [[../output/reports/Kimi K3技术报告后续阅读重点|Kimi K3技术报告后续阅读重点]]
- [[../output/reports/Kimi K3权重构成与TP8切分|Kimi K3权重构成与TP8切分]]
- [[../output/reports/Kimi K3的KDA部署与Prefix Cache|Kimi K3的KDA部署与Prefix Cache]]
- [[../output/reports/LPT在Causal Attention中的调度优化|LPT在Causal Attention中的调度优化]]
- [[../output/reports/MLA模型常见部署拓扑|MLA模型常见部署拓扑]]
- [MoE计算流程与TP-EP实现](../output/reports/MoE计算流程与TP-EP实现.html)
- [[../output/reports/MoE通信-计算重叠与MegaKernel技术谱系|MoE通信-计算重叠与MegaKernel技术谱系]]
- [[../output/reports/MoonEP动态冗余Expert机制|MoonEP动态冗余Expert机制]]
- [[../output/reports/PCP是什么|PCP是什么]]
- [[../output/reports/Prefill Attention 的 CUDA 并行映射|Prefill Attention 的 CUDA 并行映射]]
- [[../output/reports/Recursive Language Models 中文导读|Recursive Language Models 中文导读]]
- [[../output/reports/Triton在Ascend上的支持|Triton在Ascend上的支持]]
- [[../output/reports/Triton跨芯片支持|Triton跨芯片支持]]
- [[../output/reports/vLLM CUDA Graph Capture Size为何是两倍max_num_seqs|vLLM CUDA Graph Capture Size为何是两倍max_num_seqs]]
- [[../output/reports/vLLM CUDA Graph Piecewise 与 Full Decode Only|vLLM CUDA Graph Piecewise 与 Full Decode Only]]
- [[../output/reports/现代推理框架中的Torch Compile作用|现代推理框架中的Torch Compile作用]]
- [[../output/reports/算子融合与Torch Compile、CUDA Graph的分层关系|算子融合与Torch Compile、CUDA Graph的分层关系]]

## HTML 导出

- 暂无

## 面试备考

- [[../output/interview/AI Infra面试题全答（二）|AI Infra面试题全答（二）]]
- [[../output/interview/多卡GPU监控与SM执行模型面试整理|多卡GPU监控与SM执行模型面试整理]]
- [[../output/interview/多卡与推理系统面试梳理|多卡与推理系统面试梳理]]
- [[../output/interview/大模型系统面试题全答|大模型系统面试题全答]]
- [[../output/interview/大模型系统面试题全答补充|大模型系统面试题全答补充]]
- [[../output/interview/大模型系统面试题地图|大模型系统面试题地图]]
- [[../output/interview/字节二面高压题拆解|字节二面高压题拆解]]
- [[../output/interview/推理系统专题面试稿|推理系统专题面试稿]]
- [[../output/interview/算子与GPU优化、推理优化补充|算子与GPU优化、推理优化补充]]
- [[../output/interview/量化剪枝推理瓶颈Nsight与异构集群面试整理|量化剪枝推理瓶颈Nsight与异构集群面试整理]]
- [[../output/interview/面试经验|面试经验]]

## 复习卡片

- 暂无

## 幻灯片

- 暂无
