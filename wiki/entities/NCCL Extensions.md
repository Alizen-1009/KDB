---
type: entity
entity_type: 项目
topic: 并行与分布式
sources: 1
updated: 2026-07-25
---

# NCCL Extensions

## 一句话说明

`NCCL Extensions` 是 NVIDIA 官方 GitHub 组织下、建立在 NCCL Host/Device API 之上的开源项目，为 MoE 专家路由和跨 GPU Mesh 张量重分片提供带 AI 工作负载语义的通信组件。

## 类型

- 项目 / 通信库扩展

## 核心信息

- 仓库：`github.com/NVIDIA/nccl-extensions`，以 NCCL git submodule 为基础构建；它是独立扩展项目，不等于相关能力已全部进入 NCCL core。
- `nccl_ep`：提供 MoE dispatch/combine，包含面向小 batch、低延迟推理的 LL 路径，以及面向训练和 prefill 的 HT 路径。
- `nccl_m2n`：描述源/目标 Mesh 和 distributed tensor，通过全局 shard 交集生成 TransferPlan，并使用 NCCL Window 执行跨 Mesh reshard。
- 系统分工是“Host 规划、Device 执行”：CPU 仍负责拓扑和计划，GPU 侧承担 signal、等待、one-sided put 与通信热路径，而不是完全消除 CPU。
- 文章所述版本要求与支持范围较新且严格，使用时应以具体 commit 的 README、测试和 benchmark 为准。

## 相关概念

- [[Expert Parallelism]]
- [[跨 Mesh 权重重分片]]
- [[通信-计算重叠]]
- [[集合通信]]
- [[MoE]]

## 相关来源

- [[../sources/NVIDIA 开源 NCCL Extensions：把 MoE 专家路由与跨 Mesh 权重重分片推进到 GPU 设备侧]]

## 冲突与备注

- 应区分 [[NCCL]] 与 `NCCL Extensions`：前者是通用 GPU 集合通信基础库，后者在其 Host/Device API 上实现面向特定 AI 模式的扩展组件。
- `nccl_ep` 不是完整 MoE runtime，不负责请求调度、KV Cache 或专家 GEMM；`nccl_m2n` 也不负责 RL 轨迹、奖励和策略版本切换，只处理张量数据通路与重分片。
- “零拷贝”是对 Host staging 和中间复制路径的描述，不是零网络流量。
