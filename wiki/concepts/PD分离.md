---
type: concept
topic: 推理服务
sources: 2
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

## 关键权衡

- 获得更好的资源专用化和调优空间
- 代价是更高的系统复杂度、缓存传输成本和运维投入
- 与 [[Chunked Prefill]] 不是简单替代关系：PD 分离削弱了 chunked prefill 在同一 GPU 上保护 decode ITL 的价值，但 prefill pool 内部仍可能需要 chunking 来控制长 prompt 对 TTFT、公平性、峰值显存和 KV 传输流水的影响。

## 相关实体

- [[../entities/RTP-LLM]]
- [[../entities/Nvidia Dynamo]]
- [[../entities/TensorRT-LLM]]

## 相关来源

- [[../sources/LLM推理优化核心技术]]
- [[../sources/RTP-LLM]]

## 相关概念

- [[KV Cache]]
- [[分层 KV Cache]]
- [[缓存感知路由]]
- [[Tensor Parallelism]]
- [[Chunked Prefill]]

## 研究备注

- 条件聚合比全量解耦更实用，后续值得单独拆成子概念页
- 面试里应避免说“有了 PD 分离就不需要 chunked prefill”。更准确的说法是：严格 PD 分离能更可靠地隔离 decode tail latency，但 chunked prefill 仍是 prefill 侧和混部路径的调度粒度工具。
- RTP-LLM 的案例提醒：PD 分离不只是两类节点拆开，还要配套 KV 传输、缓存亲和路由、故障恢复和独立扩缩容；否则容易只把瓶颈从 GPU 计算转移到网络和调度层。
