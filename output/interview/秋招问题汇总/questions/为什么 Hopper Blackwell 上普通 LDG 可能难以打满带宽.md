# 为什么 Hopper、Blackwell 上普通 LDG 可能难以打满带宽？

- 整理状态：已整理
- 题目出处：用户提供的百度 Infra 面试复盘

## 30 秒回答

高带宽需要足够多并发的在途内存请求。若 kernel 的加载形成串行依赖，或可调度 warp、每 warp 的独立加载太少，即使访问合并，内存流水线也会有空档。我会先检查访问模式、并发 CTA/warp、eligible warps 和内存 stall，再尝试增加 ILP、独立累加链或流水深度。规则的大块 tile 可考虑 TMA 异步搬运，减少逐线程地址生成，并与计算重叠。但普通 LDG 并非一定打不满带宽；TMA 的描述符、shared memory 和同步开销也需要由目标 shape 实测证明划算。

## 深入解释

“Hopper/Blackwell 必须用 TMA”是错误推广。TMA 主要是 global↔shared 的规则 tile 搬运路径；一次性、细粒度或不规则访问仍可能适合普通 load。增加 stage 还会消耗 shared memory，可能降低驻留 CTA 数。

## 关联知识

- [[../../../../wiki/concepts/Occupancy|Occupancy]]
- [[../../../../wiki/concepts/Tiling|Tiling]]
- [[TMA 与 cp.async 如何选择|TMA 与 cp.async]]

## 参考来源与待核实

- [CUDA Programming Guide：Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)；[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/)。

## 所属题单

- [[../sets/百度Infra面试|百度 Infra 面试]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
