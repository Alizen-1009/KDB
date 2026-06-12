# RTP-LLM

## 一句话说明

`RTP-LLM` 是 [[阿里巴巴]] 开源的生产级 LLM/VLM 推理 serving 引擎，强调模型加载、调度、KV cache、PD 分离、推测解码、量化和多模态推理的全栈集成。

## 类型

- 项目 / 推理框架

## 核心信息

- 架构上采用“中心化调度 + 分布式执行”：Master 维护全局 worker 状态、KV cache 分布和负载信息，Prefill 节点、Decode 节点、多级缓存与 DP 控制器分别承担不同职责。
- 支持 `PD-Fusion` 与 `PD-Disaggregation` 两种部署模式；来源强调阿里生产环境主要采用 Prefill/Decode 物理分离，并通过跨节点 KV cache 传输衔接两个阶段。
- 模型加载路径围绕文件顺序驱动加载、混合分布式读取、共享内存复用和 I/O-通信重叠优化，目标是减少 FUSE 云存储下的随机访问和多进程重复读取。
- 调度层同时考虑 Prefill 侧长度分组/预测完成时间、Decode 侧 KV cache 亲和性、worker 负载和缓存复用分数。
- KV cache 设计扩展到 [[../concepts/分层 KV Cache]]：缓存可位于 GPU、本地 CPU、远程 CPU 和分布式存储，Master 使用统一哈希映射和增量同步做跨节点前缀匹配。
- 推测解码框架采用 C++ 模块化设计，把提议生成、目标模型评分、接受采样和状态更新拆成独立组件，支持朴素 draft、Prompt Lookup、Eagle、MTP 等路线。
- 多模态支持强调 EPD 解耦：将 ViT/视觉编码与 LLM 文本生成分开部署，让视觉和语言计算流可以独立伸缩和重叠执行。

## 相关概念

- [[../concepts/PD分离]]
- [[../concepts/分层 KV Cache]]
- [[../concepts/KV Cache]]
- [[../concepts/Prefix Caching]]
- [[../concepts/缓存感知路由]]
- [[../concepts/Speculative Decoding]]
- [[../concepts/混合精度训练与推理]]
- [[../concepts/Continuous Batching]]

## 相关来源

- [[../sources/RTP-LLM]]

## 冲突与备注

- 来源中的性能数字应保留为特定 benchmark/生产流量下的声称，不能直接泛化到所有模型、硬件、并发和上下文长度。
- 开源 RTP-LLM 与阿里内部生产部署能力可能不完全等价；涉及 `Carbon`、内部 FUSE 云存储、3FS 和业务流量的细节需要按公开论文、文档和代码再核实。
- 与 [[vLLM]]、[[SGLang]] 对比时，应按指标拆开：模型加载、TTFT、吞吐、cache hit、多模态和推测解码的瓶颈并不相同。
