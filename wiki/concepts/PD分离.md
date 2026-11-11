---
type: concept
topic: 推理服务
sources: 9
updated: 2026-07-08
---

# PD分离

## 定义

将 Prefill 和 Decode 拆分到不同引擎或不同硬件资源上的推理架构，也常称为 `Prefill Decode Disaggregation`。

## 它解决什么问题

- 减少 prefill 与 decode 在同一套资源上互相争用
- 让不同阶段分别采用更适合自身负载的优化与并行策略
- 更准确地说，它主要保护 decode 阶段的 ITL / TPOT 和 tail latency，而不是让单个 decode token 的模型计算本身变少；代价可能转移到 prefill 排队、KV Cache 传输和首 token 延迟。

## 核心机制

- Prefill 引擎负责长输入处理、KV 构建和首 token
- Decode 引擎负责后续自回归生成
- 通过 NVLink 或 InfiniBand 等互联传递 KV Cache
- 在 [[../entities/RTP-LLM]] 中，PD 分离被组织成 `PD-Fusion` 与 `PD-Disaggregation` 两种部署模式；后者将 Prefill/Decode 放到专用节点上，并通过跨节点 KV cache 传输衔接两个阶段
- [[../sources/KV Cache 研习社 · 深潜  LMCache、Mooncake、FlexKV——三条路线谁在改写 KV 的命运？]] 将 [[../entities/Mooncake]] 作为 PD 分离与跨节点 KV 传输的例子；RDMA 和 GPU 直达的可用性依赖配置。

## vLLM 交接、传输与多轮复用

[[../sources/Taking vLLM Apart A Practical Guide to Disaggregated Serving]] 描述的 NIXL 示例中，代理先请求 Prefill 计算并保留 KV blocks，再把返回的 `kv_transfer_params` 交给 Decode 拉取；示例代理仅适合开发，生产部署还需编排、超时与失败处理。原文列出 `fail` 和回退重算策略，具体参数及默认值应按部署版本核实。

传输速度直接影响首 token 路径，也可能限制 goodput。原文在双 L40S、Qwen2.5-7B、约 8k 输入且 GPU P2P 不可用的实验中观察到：PD 的 ITL 尾延迟更低，但 TTFT 比混部更高；混部基线是单个 DP=2 服务，不能将该结果外推到其他拓扑。应同时测量实际 KV 传输延迟、TTFT、p99 ITL 和 SLO goodput，而不是只比较链路理论带宽。

多轮请求中，单向传输可能让下一轮 Prefill 重算 Decode 已生成的历史；文章提出双向 KV 传输，并由代理跟踪会话。复用前必须保证新 prompt 与缓存前缀逐 token 一致，尤其留意 chat template 是否移除 reasoning 内容；文章称相关实现存在正确性风险，影响版本与修复状态待核实。

## 同构与异构 PD

PD 分离不仅允许 Prefill/Decode 独立扩缩，也使 Decode backend 成为可选择组件：

- 同构 PD：Prefill 与 Decode 使用同一种引擎，状态 layout 和兼容路径相对简单。
- 异构 PD：例如 vLLM Prefill + [[../entities/TileRT]] Decode；共享 vLLM API、调度和 Prefix Cache，但将延迟敏感流量交给专用 Decode runtime。

异构交接通常不只传普通 KV，还可能包含 sparse attention index、MTP draft-layer KV 与元数据，并需在目标端转换为原生 layout，详见 [[可插拔 Decode 引擎]]。

## 与 AFD 的区别与组合

[[Attention-FFN 分离]] 与 PD 分离作用于不同切分轴，可以叠加：PD 在请求生命周期上拆 Prefill/Decode，AFD 在每个切分层内部拆 Attention/FFN。组合部署可以分别设置 Prefill A/F 资源池和 Decode A/F 资源池。

两者的状态传输不同：PD 主要在 Prefill 完成后把各层 KV Cache 与请求状态交给 Decode，属于阶段性交接；AFD 则在每个 forward step 的每个切分层发送当前 hidden states，并把 FFN output 返回 Attention 侧，频率更高但中间激活生命周期更短。vLLM AFD Plugin 的异步 CAM 路径就是面向 PD 分离部署中的 prefill 阶段。

## Wide-EP 下的额外价值

[[Wide Expert Parallelism]] 的 ranks 需要在每个 MoE 层协调 dispatch/combine；一个长而 compute-bound 的 prefill 请求可能拖慢整个 EP group。PD 分离可把这种干扰隔离到 Prefill pool，并让 Prefill/Decode 分别选择偏高吞吐或低延迟的 [[../entities/DeepEP]] 路径。

[[DWDP]] 论文在 PD 分离中仅优化 context 侧：单位 GPU 吞吐可能提高，但减少 context GPU 后排队可能增加，论文的端到端配对实验中位 TTFT 均上升。因此应同时评估 GPU 配比、吞吐与 TTFT SLO。

## 关键权衡

- 获得更好的资源专用化和调优空间
- 代价是更高的系统复杂度、缓存传输成本和运维投入
- 与 [[Chunked Prefill]] 不是简单替代关系：PD 分离削弱了 chunked prefill 在同一 GPU 上保护 decode ITL 的价值，但 prefill pool 内部仍可能需要 chunking 来控制长 prompt 对 TTFT、公平性、峰值显存和 KV 传输流水的影响。

## 阶段交接与持久缓存写回

[[../sources/谈谈 DeepInfer]] 描述的路径是：Prefill 计算 KV 并交给 Decode，随后释放相应 GPU KV，由 Decode 写入分布式存储。这是作者对 DeepSeek 系统的描述，释放的同步条件、写回粒度与生产实现待核实，不是所有 PD 架构的固定协议。

LLM 归纳：PD 阶段交接与跨请求持久缓存是两件事；释放 Prefill 副本不意味着 Decode 不需要执行期 KV。“计算节点无状态”应限定在持久缓存驻留与调度依赖语境，不能推导出无需请求元数据和故障恢复。原文还提到 DualPath 借 Decode Relay 缓解 Prefill 存储入口瓶颈；400Gbps 的配置和适用版本待核实，“V4.1 可能无需 DualPath”仍是作者猜测。

## 相关实体

- [[../entities/DeepInfer]]

- [[../entities/RTP-LLM]]
- [[../entities/Nvidia Dynamo]]
- [[../entities/TensorRT-LLM]]
- [[../entities/vLLM AFD Plugin]]
- [[../entities/vLLM]]
- [[../entities/DeepEP]]
- [[../entities/TileRT]]
- [[../entities/Mooncake]]

## 相关来源

- [[../sources/Taking vLLM Apart A Practical Guide to Disaggregated Serving]]

- [[../sources/谈谈 DeepInfer]]
- [[../sources/LLM推理优化核心技术]]
- [[../sources/RTP-LLM]]
- [[../sources/vLLM AFD Plugin 发布：为 MoE 推理拆分 Attention 与 FFN，实现灵活部署]]
- [[../sources/vLLM Large Scale Serving DeepSeek @ 2.2k toksH200 with Wide-EP]]
- [[../sources/vLLM x TileRT Specialized Decode for Latency-Critical Serving]]
- [[../sources/KV Cache 研习社 · 深潜  LMCache、Mooncake、FlexKV——三条路线谁在改写 KV 的命运？]]
- [[../sources/DWDP Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72]]

## 相关概念

- [[KV Cache]]
- [[分层 KV Cache]]
- [[缓存感知路由]]
- [[Tensor Parallelism]]
- [[Chunked Prefill]]
- [[Attention-FFN 分离]]
- [[Wide Expert Parallelism]]
- [[可插拔 Decode 引擎]]
- [[DWDP]]

## 研究备注

- 条件聚合比全量解耦更实用，后续值得单独拆成子概念页
- 面试里应避免说“有了 PD 分离就不需要 chunked prefill”。更准确的说法是：严格 PD 分离能更可靠地隔离 decode tail latency，但 chunked prefill 仍是 prefill 侧和混部路径的调度粒度工具。
- RTP-LLM 的案例提醒：PD 分离不只是两类节点拆开，还要配套 KV 传输、缓存亲和路由、故障恢复和独立扩缩容；否则容易只把瓶颈从 GPU 计算转移到网络和调度层。
