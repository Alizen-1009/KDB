---
type: entity
entity_type: 项目
topic: 并行与分布式
updated: 2026-08-02
sources: 0
---

# MoonEP

## 一句话说明

[[Kimi K3]] 3T级预训练系统使用的perfectly balanced Expert Parallel方案：根据每个micro-batch、每层的真实Router输出动态复制热点experts，把同一逻辑expert的部分token assignments迁移到空闲ranks执行。

## 核心机制

MoonEP不改变Router选中的expert ID，只改变该expert在哪个rank的home或redundant copy上执行。Forward前在线规划并prefetch副本权重；Backward将副本梯度Reduce回home rank的正式gradient buffer。

设`E`个experts、`R`个EP ranks，每rank预留`E/R`个redundant-expert slots即可保证任意Router输出下存在perfectly balanced plan，使每rank严格执行`S×K`个token–expert assignments。该上界在最坏情况下基本紧致，但不代表每一步都会复制`E/R`个experts。

## 系统收益

- 消除rank级MoE straggler；
- 令rank级buffer与compute shape静态；
- fused permute/unpermute可把token直接发送到远端expert-grouped位置；
- 通信buffer固定为`S×K`，而不是为最坏不均衡预留`S×K×R`；
- 避免每层host等待device返回动态总负载shape。

Rank总工作量完全均衡后，rank内部不同experts的token数仍可不均，因此还需要workload-aware expert-GEMM scheduler。

## 与Quantile Balancing的区别

Quantile Balancing通过expert-specific router bias改变训练中的Top-K选择倾向；MoonEP不改变选择结果，只复制执行位置。前者控制长期训练负载和dying experts，后者消除当前micro-batch的瞬时rank不均衡。

## 与EPLB的关系

MoonEP与DeepSeek/vLLM EPLB同属“logical expert映射到多个physical replicas”的负载均衡家族。区别是：vLLM EPLB按历史window周期性重排expert mapping，优化预计/平均负载；MoonEP读取当前layer、当前micro-batch真实routes，按assignment在线规划并要求当前step每rank严格`S×K`工作量。MoonEP的`E/R`冗余槽给出任意路由下的可行性保证，并包含训练Backward副本梯度回收；EPLB的冗余数由配置决定，在线Serving没有Backward。

## 适用边界

K3报告将MoonEP放在预训练基础设施章节，明确包含Backward gradient reduction。不能据此断言在线Decode也使用相同动态副本机制；Serving需权衡副本权重prefetch、预留显存和小batch weight-bandwidth成本。

## 详细报告

- [[../../output/reports/MoonEP动态冗余Expert机制|MoonEP动态冗余Expert机制]]

## 相关概念

- [[../concepts/Expert Parallelism|Expert Parallelism]]
- [[../concepts/Wide Expert Parallelism|Wide Expert Parallelism]]
- [[../concepts/LatentMoE|LatentMoE]]

## 官方资料

- [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)，§5.2.1与Appendix E
- [MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP)

## 待核实

- GPU planner的near-optimal启发式、P2P weight prefetch与计算重叠细节需绑定MoonEP源码版本。
- 在线Serving是否使用MoonEP动态冗余需要具体backend证据。
