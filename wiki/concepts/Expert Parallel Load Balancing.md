---
type: concept
topic: 并行与分布式
sources: 0
updated:
---

# Expert Parallel Load Balancing

## 定义

`Expert Parallel Load Balancing`（EPLB）是在 MoE 推理时根据真实 token 路由负载，动态调整逻辑 expert 到物理 rank 的 placement，以减少 Wide-EP 中部分 ranks 空闲、部分 ranks 过载的尾部效应。

## 核心机制

- 每次 MoE forward 记录各逻辑 expert 的 token load。
- 用滑动窗口跨 EP ranks 聚合统计。
- 到达 rebalance interval 后计算新的 logical-to-physical mapping。
- 执行 expert 权重 shuffle，让新 placement 生效。
- 可配置窗口、重平衡周期、冗余 expert 和日志选项。

需要区分：Router 仍决定 token 选哪个逻辑 expert；EPLB 决定该 expert 当前位于哪个物理 rank。冗余 expert 则可为热点逻辑 expert 提供多个物理副本。

## 与训练负载均衡的区别

训练时的 auxiliary loss、expert bias 或 quantile balancing 会改变 Router 学到的选择分布；EPLB 是 serving 部署层策略，不改变已训练模型的逻辑 expert 语义，而是调整物理放置。

## 关键权衡

- 可降低热点 expert 导致的 rank tail，但需支付负载统计、mapping 计算和权重迁移成本。
- 窗口太短容易追逐噪声，太长则对流量变化反应迟缓。
- 重平衡期间如何保证请求一致性、CUDA Graph 和正在执行 batch 的安全，需要具体 runtime 支持。
- 冗余 experts 会消耗额外显存，但可能降低热点排队。

## 相关实体

- [[../entities/vLLM]]

## 相关来源

- [[../sources/vLLM Large Scale Serving DeepSeek @ 2.2k toksH200 with Wide-EP]]

## 相关概念

- [[Wide Expert Parallelism]]
- [[Expert Parallelism]]
- [[Tail Effect]]
- [[MoE]]

## 研究备注

- 后续需结合 DeepSeek EPLB repo 核实 hierarchical/global policy、冗余 expert 分配和在线 weight shuffle 的实现边界。
