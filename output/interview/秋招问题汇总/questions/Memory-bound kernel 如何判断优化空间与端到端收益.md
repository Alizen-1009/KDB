# Memory-bound kernel 如何判断优化空间与端到端收益？

- 整理状态：已整理
- 题目出处：用户提供的百度 Infra 面试复盘

## 30 秒回答

我先按 shape 和 dtype 算必要读写字节，用 kernel 时间估算有效带宽，并与同设备、相近访问模式的可持续带宽基线比较；再看 profiler 的 DRAM/L2 流量，确认数据实际来自哪里。若仍有明显差距，依次查合并访问、无效或重复流量、在途内存请求、寄存器溢出、可调度 warp 和尾块。若带宽已接近可达到的上限，就转向减少必需字节数或融合中间结果。最后看该 kernel 占端到端时间的比例，按 Amdahl 定律估计继续优化的收益。

## 深入解释

- 仅看 `bytes / time` 不够：分清算法最少字节数、L1/L2 流量和实际 DRAM 字节数。缓存命中高时，不能把有效带宽直接拿去与 HBM 峰值比较。
- Occupancy 低是线索，不自动等于瓶颈；应与 eligible warps、stall 原因、寄存器和实际吞吐一起看。
- 若 kernel 占总耗时比例为 `f`，即使该 kernel 提速 `s` 倍，端到端上限也只是 `1 / ((1-f)+f/s)`。还要把 launch、同步与邻接算子纳入计时。

## 10 月 10 日追问：增加很多 SM 后仍然 memory-bound，为什么？

- 追问出处：用户提供的 2026-10-10 面试回忆。
- 口述答案：先区分“仍然 memory-bound”和“没有加速”。更多 SM 可能已经降低耗时，但瓶颈仍在内存；若算术强度和共享带宽没改变，增加计算资源不会自动使它 compute-bound。若 HBM 已饱和，应减少搬运或增加复用；若 HBM 没打满，查访存依赖、独立在途请求、访问效率、L2/LSU 限制及新增 SM 是否获得有效工作。

Roofline 上界：

$$P\le\min(P_{compute},BW\times AI)$$

| 原因 | 更多 SM 为什么不够 | 验证方向 |
|---|---|---|
| HBM 带宽饱和 | 争用同一共享带宽 | 实际 DRAM 流量与可持续带宽基线；减少必要字节数 |
| 访存延迟/依赖限制 | 地址依赖或独立请求不足 | Eligible warps、stall、ILP、预取和在途请求 |
| 访问或内存管线效率差 | 非合并、重复流量，L2/L1/LSU 或 shared bank 限制 | 分层流量、事务效率、管线利用率和 bank conflict |
| 有效并行工作不足 | CTA 太少、资源占用、不可拆任务或尾块 | Grid、活跃 SM、寄存器/shared 占用和尾部时间 |

“增加 SM”需说明是同卡提高参与度/资源份额，还是换更多 SM 的 GPU；更换 GPU 时带宽、缓存与频率也可能改变，不能只按 SM 数解释。Memory-bound 不等于 HBM 已打满，也不能仅凭某个 stall 指标判定唯一原因。

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：区分 Mem Busy、Max Bandwidth、Mem Pipes Busy 等限制。
- 具体原因待目标 kernel profiling 确认；本次没有硬件、shape 或实测数据。

## 关联知识

- [[../../../../wiki/concepts/Roofline 模型|Roofline 模型]]
- [[../../../../wiki/concepts/Profiling|Profiling]]
- [[../../../../wiki/concepts/Occupancy|Occupancy]]
- [[如何用 Roofline 模型区分计算、访存和其他瓶颈|Roofline 面试题]]

## 参考来源与待核实

- [CUDA Best Practices：Effective Bandwidth](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)；具体余量取决于目标 GPU、访问模式和测量条件，不存在通用带宽利用率阈值。

## 所属题单

- [[../sets/10月10号面试遇到的问题|10月10号面试遇到的问题]]

- [[../sets/百度Infra面试|百度 Infra 面试]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
