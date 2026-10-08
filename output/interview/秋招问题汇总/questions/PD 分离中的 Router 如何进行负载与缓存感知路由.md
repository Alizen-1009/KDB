# PD 分离中的 Router 如何进行负载与缓存感知路由？

- 整理状态：已整理
- 题目出处：用户提供的“千问 C 端 AI Infra 一面”面经

## 其他问法

- PD 分离中还有哪些先进的 Router 实现？
- Router 如何同时考虑队列负载、KV Cache 命中与传输成本？

## 问题背景

该题承接 PD 分离与 P:D 实例配比，需要区分请求入口路由、Prefill/Decode 阶段调度、缓存亲和性和跨实例 KV 迁移；具体实现与性能结论待绑定项目和版本。

## 30 秒回答

先为请求选择可用的 prefill/decode 资源池，再综合队列长度、可用 KV 容量、前缀缓存命中、KV 传输成本与目标 TTFT/ITL 分配实例。缓存命中可省 prefill，但若热点实例排队过长，负载均衡可能更快；需要定期更新状态并避免路由抖动。

## 深入解释

SGLang Model Gateway 的 cache-aware 策略是一个实现例子，利用前缀索引近似估算缓存命中，并结合负载阈值；具体 PD 跨池配对与版本细节须看部署配置。

若大量请求共享同一个 System Prompt，按最大前缀命中路由会持续选择同一副本：该副本的 prefill 节省可能被排队时间抵消，形成 TTFT 长尾。面试时可用“预计完成时间”统一比较候选副本：当前排队时间 + 未命中部分的 prefill 时间 + 必要的 KV 传输/同步时间；缓存命中提供计算节省，但过载时应让其他副本接手。若实现采用加权命中率与负载分数，先归一化量纲，再按实际 TTFT/SLO 标定权重并加入过载阈值或滞回，避免路由震荡。

评估应固定请求 trace，并按数据集的前缀重复率、输入/输出长度及到达率分组，报告总体与分组的 P50/P95/P99 TTFT、ITL、吞吐、SLO 违约率和每副本队列/缓存分布；与纯负载、纯缓存和原策略比较。这里只给验证方法，不代表某项目已经取得这些收益。

## 关联知识

- [[../../../../wiki/concepts/缓存感知路由|缓存感知路由]]
- [[../../../../wiki/concepts/Prefix Caching|Prefix Caching]]
- [[PD 分离多副本仿真如何建模与校验延迟数据|PD 分离多副本仿真]]

## 参考来源与待核实

- [SGLang cache-aware 路由源码](https://github.com/sgl-project/sglang/blob/main/sgl-model-gateway/src/policies/cache_aware.rs)
- 重复 System Prompt 的热点、加权公式和跨数据集收益来自本次用户提供的项目追问；具体实现、权重和性能数字待项目代码与实验记录核实。这里的预计完成时间是候选设计，不归因于特定 SGLang 版本。

## 所属题单

- [[../sets/千问C端AI Infra一面|千问 C 端 AI Infra 一面]]
- [[../sets/小鹏汽车端侧AI Infra三面|小鹏汽车端侧 AI Infra 三面]]
- [[../sets/推理服务|推理服务]]
- [[../README|秋招问题汇总]]
