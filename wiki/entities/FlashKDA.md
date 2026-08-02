---
type: entity
entity_type: 项目
topic: GPU 编程
updated: 2026-08-02
sources: 0
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

FlashKDA是专用backend；Flash Linear Attention（FLA）是提供多种线性注意力算子与backend接口的库。

## 详细报告

- [[../../output/reports/FlashKDA为什么能并行|FlashKDA为什么能并行]]

## 相关概念

- [[../concepts/KDA|KDA]]
- [[../concepts/Chunked Gated Delta Rule|Chunked Gated Delta Rule]]
- [[../concepts/线性注意力递归状态|线性注意力递归状态]]

## 官方资料

- [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA)
- [Kimi K3 Technical Report](../../raw/papers/k3_tech_report.pdf)，§2.1.1与§5.1

## 待核实

- CUTLASS tile、pipeline、workspace和支持shape需绑定具体commit与GPU架构。
