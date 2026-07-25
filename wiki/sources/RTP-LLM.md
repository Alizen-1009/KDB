---
type: source
source_kind: 文章
topic: 推理服务
updated: 2026-06-12
---

# RTP-LLM：阿里开源工业级 LLM 推理引擎，模型加载提速 6.3 倍、TTFT 降低 37%，吞吐量领先 vLLM 与 SGLang！

## 来源信息

- 标题：RTP-LLM：阿里开源工业级 LLM 推理引擎，模型加载提速 6.3 倍、TTFT 降低 37%，吞吐量领先 vLLM 与 SGLang！
- 作者：阿里，北大，浙大（原文署名）
- 发布时间：2026-05-31
- 原始文件：`raw/articles/RTP-LLM：阿里开源工业级 LLM 推理引擎，模型加载提速 6.3 倍、TTFT 降低 37%，吞吐量领先 vLLM 与 SGLang！.md`
- 外部线索：微信原文、arXiv `2605.29639`、GitHub `alibaba/rtp-llm`、官方文档 `rtp-llm.ai`

## 2-3 条核心摘要

- 这篇来源把 [[../entities/RTP-LLM]] 描述为一个生产级 LLM/VLM 推理 serving 引擎，而不是单点 kernel 优化：系统同时覆盖模型加载、中心化调度、[[../concepts/分层 KV Cache]]、[[../concepts/PD分离]]、推测解码、量化和多模态解耦。
- RTP-LLM 的架构主线是“中心化调度 + 分布式执行”：Master 维护 worker 状态、KV cache 分布和负载信息，Prefill/Decode 节点按不同资源画像拆开，DP 控制器负责本地批处理和 GPU 资源管理。
- 原文给出多组性能声称，包括模型加载 4.7x-6.3x、生产流量 TTFT P95 降低 35%-37%、KV 复用长度提升 215%、推测解码吞吐提升 1.12x-2.48x、多模态 EPD 吞吐提升 1.86x-2.52x；这些数字应带着模型、硬件、流量和版本边界引用。

## 值得关注的论断

- RTP-LLM 的模型加载优化不是简单并行读文件，而是把加载范式从模型结构驱动改为文件顺序驱动，再结合 `fastsafetensors`、共享内存复用和 I/O-通信重叠，减少 FUSE 云存储上的随机访问和重复读取。
- 其 KV cache 设计更偏生产系统层：统一哈希映射做跨 worker 前缀匹配，缓存键增量同步，多级缓存覆盖 GPU、本地 CPU、远程 CPU 和分布式存储，并让调度器在负载均衡和缓存命中之间折中。
- RTP-LLM 对 [[../entities/vLLM]] 和 [[../entities/SGLang]] 的对比应理解为特定配置下的来源声称，不能泛化为“所有负载下全面领先”；尤其要区分模型加载、TTFT、吞吐、多模态、推测解码等不同指标。

## 关键概念

- [[../concepts/PD分离]]
- [[../concepts/分层 KV Cache]]
- [[../concepts/Prefix Caching]]
- [[../concepts/缓存感知路由]]
- [[../concepts/Speculative Decoding]]
- [[../concepts/混合精度训练与推理]]
- [[../concepts/Continuous Batching]]

## 相关实体

- [[../entities/RTP-LLM]]
- [[../entities/阿里巴巴]]
- [[../entities/vLLM]]
- [[../entities/SGLang]]
- [[../entities/Qwen VL]]

## 与现有 wiki 的关系

- 会创建哪些实体页：`RTP-LLM`、`阿里巴巴`
- 会创建哪些概念页：`分层 KV Cache`
- 会更新哪些概念页：`PD分离`、`KV Cache`、`Prefix Caching`、`缓存感知路由`、`Speculative Decoding`、`混合精度训练与推理`
- 会更新哪些实体页：`vLLM`、`SGLang`、`Qwen VL`
- 是否存在冲突：未发现与现有 wiki 的直接事实冲突；本来源主要把既有推理系统概念推进到“跨节点、生产流量、缓存层级和运维可靠性”视角。但它关于性能领先的表述更强，需要保留为来源声称并按论文和开源代码复核。

## 待确认

- 原始剪藏中部分公式被图片/OCR 丢失，Prefill 预测调度和缓存复用评分的精确定义需要回到论文 PDF 或源码核对。
- 文章中的生产部署、用户规模、`Carbon` 服务、3FS/远程 KV cache 等细节可能包含阿里内部系统语境；开源仓库是否完整覆盖这些能力需要单独核实。
- Benchmark 数字需要记录模型、硬件、并行配置、输入输出长度、并发、cache hit、量化方式和框架版本后再用于严格横评。
