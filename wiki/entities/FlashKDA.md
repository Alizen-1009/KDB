---
type: entity
entity_type: 项目
topic: GPU 编程
updated: 2026-08-02
sources: 2
---

# FlashKDA

## 一句话说明

Moonshot为KDA训练与Prefill开发的CUTLASS chunkwise kernel：利用KDA状态更新的仿射可组合性和chunk内lower-triangular矩阵化，将逐token recurrence转成Tensor Core GEMM，并重叠块内token计算与块间状态传播。

## 为什么能并行

KDA可写为：

```text
S_t = M_t S_{t-1} + B_t
```

仿射transition满足结合律，因此segments可以用prefix scan组合；chunk内部则通过UT transform改写为：

```text
O = (Gamma * Q) @ S_in + Tril((Gamma * Q) @ (K/Gamma)^T) @ V_tilde
```

使多个tokens通过causal GEMM并行。K3将log-decay下界设为-5，使16-token secondary tile的reciprocal decay留在BF16范围内，对角与非对角tiles均可使用dense Tensor Core GEMM。

## 适用范围

- 训练；
- 长序列Prefill；
- 不主要用于`T=1` Decode，后者走fused recurrent kernel。

FlashKDA是专用backend；Flash Linear Attention（FLA）是提供多种线性注意力算子与backend接口的库。vLLM的K3 Preview确认Prefill同时集成FlashKDA与Triton路径，并优化Merged Input Projection、Causal Conv与Initial State Gather。当前检查源码中Prefill仍可见Q/K/V Conv逻辑调用，因此“Projection + Conv Fusion”的精确kernel边界需绑定Release Branch；截至文章发布时最终Backend Selection和数值验证仍在进行。

## 与 FF-KDA 图示和 CAKE KDA 的关系

`REMINDER FF-KDA & CAKE KDA Highlights` 来源把三种思路放在同一优化谱系中：

- FlashKDA 公开两阶段设计：K1 沿 chunks×heads 准备 workspace，K2 按 sequence/head 顺序推进 recurrence；优势是解耦自然并行度，代价是 workspace/HBM 往返。
- 来源的 FF-KDA 图片仍保留 K1/K2 边界，但把 swizzled SMEM 作为 opaque byte image 经 raw `cp.async.bulk` S2G/G2S 搬运，避免 TMA unswizzle/reswizzle；图中称小 segment 从约 833/CTA 变为每方向 6 个 contiguous payloads。该图片未给出 PR/commit，`FF-KDA` 身份待核实。
- [[CAKE KDA]] 进一步把 preparation 与 recurrence 融入单 kernel，让 chunk-local 中间量留在 register/SMEM/TMEM；代价是 grid 受 batch×heads 约束，小并行度 shape 可能不如两阶段。

## 详细报告

- [[../../output/reports/FlashKDA为什么能并行|FlashKDA为什么能并行]]

## 相关概念

- [[../concepts/KDA|KDA]]
- [[../concepts/Chunked Gated Delta Rule|Chunked Gated Delta Rule]]
- [[../concepts/线性注意力递归状态|线性注意力递归状态]]
- [[../concepts/Tensor Memory|Tensor Memory]]

## 相关来源

- [[../sources/A Preview of Production-Scale Kimi K3 Support on vLLM]]
- [[../sources/REMINDER FF-KDA & CAKE KDA Highlights]]

## 官方资料

- [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA)
- [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)，§2.1.1与§5.1

## 待核实

- CUTLASS tile、pipeline、workspace和支持shape需绑定具体commit与GPU架构。
