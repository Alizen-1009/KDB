# Memory-bound kernel 如何判断优化空间与端到端收益？

- 整理状态：已整理
- 题目出处：用户提供的百度 Infra 面试复盘

## 30 秒回答

我先按 shape 和 dtype 算必要读写字节，用 kernel 时间估算有效带宽，并与同设备、相近访问模式的可持续带宽基线比较；再看 profiler 的 DRAM/L2 流量，确认数据实际来自哪里。若仍有明显差距，依次查合并访问、无效或重复流量、在途内存请求、寄存器溢出、可调度 warp 和尾块。若带宽已接近可达到的上限，就转向减少必需字节数或融合中间结果。最后看该 kernel 占端到端时间的比例，按 Amdahl 定律估计继续优化的收益。

## 深入解释

- 仅看 `bytes / time` 不够：分清算法最少字节数、L1/L2 流量和实际 DRAM 字节数。缓存命中高时，不能把有效带宽直接拿去与 HBM 峰值比较。
- Occupancy 低是线索，不自动等于瓶颈；应与 eligible warps、stall 原因、寄存器和实际吞吐一起看。
- 若 kernel 占总耗时比例为 `f`，即使该 kernel 提速 `s` 倍，端到端上限也只是 `1 / ((1-f)+f/s)`。还要把 launch、同步与邻接算子纳入计时。

## 关联知识

- [[../../../../wiki/concepts/Roofline 模型|Roofline 模型]]
- [[../../../../wiki/concepts/Profiling|Profiling]]
- [[../../../../wiki/concepts/Occupancy|Occupancy]]
- [[如何用 Roofline 模型区分计算、访存和其他瓶颈|Roofline 面试题]]

## 参考来源与待核实

- [CUDA Best Practices：Effective Bandwidth](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)；具体余量取决于目标 GPU、访问模式和测量条件，不存在通用带宽利用率阈值。

## 所属题单

- [[../sets/百度Infra面试|百度 Infra 面试]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
