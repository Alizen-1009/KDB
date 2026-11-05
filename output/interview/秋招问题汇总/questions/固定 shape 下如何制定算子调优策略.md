# 固定 shape 下如何制定算子调优策略？

- 整理状态：整理中
- 题目出处：既有面试原稿（见文末具体章节）

> [!note] 整理边界
> 本页从历史原稿提炼；本轮未逐题重新核验技术事实或运行代码，不表示答案已通过事实复核。

## 其他问法

- 针对固定 shape，如何制定超越官方库的算子优化策略？
- 针对特定 shape 的算子调优，如何制定优化策略以超越官方库的性能？

## 30 秒回答

超越官方库通常靠更窄、更稳定的 workload，而非单纯写得更底层。先固定基线并判断计算或访存瓶颈，再利用固定 shape、布局和算子链做融合、tile 特化、离线调优与数据流重排。最后用 profiler 解释收益，并权衡兼容性、数值风险和维护成本。

## 深入解释

沿文末原稿章节阅读完整机制、推导或代码；本页保留短答，不复制长篇正文。

## 关联知识

- [[../../../../wiki/concepts/CUDA Kernel|CUDA Kernel]]

## 参考来源与待核实

- [[../../算子与GPU优化、推理优化补充#5. 针对特定 shape 的算子调优，如何制定优化策略以超越官方库的性能？|算子与GPU优化、推理优化补充]]

- 待核实 / 原稿边界：原稿强调比较对象必须区分 PyTorch eager、torch.compile 与高度特化的官方 kernel；硬件、驱动、CUDA、shape、warmup 和计时口径必须一致。没有具体实验数字或超越官方库的个人经历证据。步骤与方向小标题均不另建题。
- 原稿是派生备考资料，不是一手技术证据；涉及具体版本、硬件、性能数字的结论，复习时应沿原稿和 wiki 继续核对一手来源。

## 面试追问：怎么证明优化有效，测过核利用率吗？

1. 和参考实现对比正确性，覆盖常用 shape、dtype、layout、尾块和边界值。
2. 固定输入和运行环境，预热后重复测 kernel latency，覆盖常见 shape 的回归。报告加速比，按瓶颈看有效带宽或吞吐。
3. 用模型 timeline 检查新路径命中、调用次数、总耗时和是否落在关键路径。最后测端到端收益。
4. NVIDIA 用 Nsight Compute 看 SM、DRAM/L2 吞吐和 achieved occupancy。昇腾用 msprof op 或 msopprof 看 Cube、Vector、MTE 流水与各核耗时。Occupancy 表示 warp 驻留情况，计算吞吐还需独立测量。
5. Double buffer 的验证可以在相同 shape 做开关 A/B，看总耗时和搬运与计算的重叠变化。个人实测数值待提供。

- [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/ComputeTriage/)
- [CANN 通算流水图](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/devaids/opdev/atlasopdev_16_0139.html)

## 所属题单

- [[../sets/推理并行、昇腾调优与 MLIR 面试追问|推理并行、昇腾调优与 MLIR 面试追问]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
