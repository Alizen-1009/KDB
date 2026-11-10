# 吞吐提升但 TTFT P95 和 P99 恶化，如何分析？

- 整理状态：已整理
- 题目出处：用户本次提供

## 30 秒回答

吞吐和尾延迟不矛盾：更大 batch、等待组批、提高利用率或接近饱和均可能增加排队，长 prompt、热点前缀和 prefill 抢占也可放大尾部。固定到达率、输入长度和缓存命中分布，分解网关、队列、prefill、首 token 及 PD 传输时间，按请求组与副本看 P95/P99 和 SLO 违约；对照 batch、token budget、准入与 P/D 配比，优先满足 TTFT SLO 而不是只看 tokens/s。需区分开环与闭环压测及首 token 定义。

依据：[[../../../../wiki/concepts/Continuous Batching|Continuous Batching]]、[[TTFT 由哪些环节构成，如何制定首 token 延迟优化方案|TTFT 分解]]。

## 所属题单

- [[../sets/长上下文与全局缓存调度面试题|长上下文与全局缓存调度面试题]]
