# 如何用 Roofline 推导 GEMM 与 MoE 的临界 token 数？

- 整理状态：已整理
- 题目出处：用户提供的《英伟达 AI Infra 五面面经》截图，AI Infra 与推理系统第 1 题；2026-11-05 收录，实际面试日期与具体轮次未提供。

## 原题与条件

根据硬件计算和访存带宽，计算 Matmul 算子的最佳输入数据大小，再根据这个计算 MoE TopK 输入的最佳 token 数（Roofline 模型）。

## 30 秒回答

先求硬件算力带宽比 F/W，再把 GEMM 的 FLOPs 除以理想搬运字节数，求进入算力受限区间的临界 M。对 MoE，均匀路由时每个 expert 平均收到 Tk/E 个 tokens，因此把目标 expert M 反推为全局 T≈EM/k。临界规模不是最佳 batch；延迟、显存、路由和通信仍需实测。

## 深入解释

### GEMM 的临界规模

设 A 为 M×K、B 为 K×N，输出 C 为 M×N；F 是目标 dtype/计算路径的 FLOP/s，W 是 byte/s，b 是每个元素的字节数。令 ρ=F/W。

按 β=0、输入和权重各读一次、输出写一次、无其他中间张量的理想模型：

$$I=\frac{2MNK}{b(MK+KN+MN)}$$

令 I≥ρ，得到：

$$M\ge\frac{\rho bKN}{2KN-\rho b(K+N)}$$

分母必须为正；若为零或负，该流量模型下没有有限 M 能达到目标 ρ。若权重读取 KN 占主导，可近似：

$$I\approx\frac{2M}{b},\qquad M_{critical}\approx\frac{bF}{2W}$$

例如仅假设 F/W=200 FLOP/byte、BF16 的 b=2，近似临界 M 为 200。这个数字是演示，不是具体 GPU 的性能数据；KN 不占主导时应使用完整公式。

### 从 GEMM 到 MoE

设全局本步 T 个 tokens、E 个 routed experts、每 token 选择 k 个 experts。在负载均衡且不计丢弃/padding 时：

$$M_e\approx\frac{Tk}{E},\qquad T_{critical}\approx\frac{E M_{critical}}{k}$$

SwiGLU expert 有多个 GEMM，维度、dtype 可能不同，应分别检查。Grouped GEMM 改善调度和硬件利用率，但不同 expert 仍使用不同权重，不能把所有 expert 的 M 简单相加成共享权重的大 GEMM。

### 适用边界

- 算力必须对应实际精度、是否稀疏及实际指令路径；不能拿不适用的峰值代入。
- 理想 traffic 不是 kernel 实测 traffic；tiling、L2、重复加载、混合精度、旧 C 读取都会改账本。
- 达到 ridge point 只是可能转向算力受限，不代表达到峰值或端到端最优。
- 若要求延迟约束下最佳 batch，还要量测通信、launch、资源占用和尾块效率。

## 关联知识

- [[../../../../wiki/concepts/Roofline 模型|Roofline 模型]]
- [[../../../../wiki/concepts/Expert Parallelism|Expert Parallelism]]
- [[../../../../wiki/concepts/Tiling|Tiling]]

## 参考来源与待核实

- [NVIDIA：Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。
- 公式为上述假设下的机制推导；本批未运行 GPU benchmark。

## 所属题单

- [[../sets/英伟达AI Infra五面面经|英伟达 AI Infra 五面面经]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
