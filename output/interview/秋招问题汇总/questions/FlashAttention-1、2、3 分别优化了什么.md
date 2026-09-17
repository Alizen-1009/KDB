# FlashAttention-1、2、3 分别优化了什么？

- 整理状态：已整理
- 题目出处：用户在本次对话中提供的面试题截图

## 其他问法

- FlashAttention-1/2/3 如何利用 SRAM 分块和 Online Softmax 减少 HBM 读写？

## 30 秒回答

三代都保持 exact attention，并用 Q/K/V tiling 与 Online Softmax 避免把完整 score/probability 矩阵写回 HBM。FA1 建立 IO-aware 主算法；FA2 进一步减少非 matmul FLOPs，沿序列增加 block 并行并重做 warp work partition；FA3 面向 Hopper，用 TMA、异步 WGMMA 和 warp specialization 重叠搬运、GEMM 与 softmax，并加入 FP8 block quantization。准确说法是显著减少 HBM 往返，而不是完全不访问 HBM。

## 关联知识

- [[../../../../wiki/concepts/FlashAttention|FlashAttention]]
- [[../../../../wiki/concepts/Online Softmax|Online Softmax]]

## 参考来源与待核实

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- exact 指数学重排而非浮点逐 bit 相同；具体 tile、loop order 和流水实现依赖版本与 GPU。

## 所属题单

- [[../sets/近期对话面试问题回顾|近期对话面试问题回顾]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
