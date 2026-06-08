# RMSNorm

## 定义

一种只基于均方根而不减去均值的归一化方法，常见形式是先计算输入向量的平方均值，再乘上其倒数平方根完成缩放。

## 它解决什么问题

- 以更低的统计开销完成与 `LayerNorm` 类似的稳定化作用
- 减少归一化 kernel 中对均值和方差两套统计量的需求
- 在现代 LLM 中作为高频基础算子，需要高效的 CUDA 实现

## 核心机制

- 对一行或一个 hidden vector 计算平方和
- 用 [[Block Reduce]] 或 [[Warp Shuffle Reduce]] 汇总得到均方根统计量
- 计算 `rsqrtf(sum / D + eps)` 得到缩放因子
- 再对原向量执行逐元素缩放，并乘可学习参数
- [[CODA]] 讨论了 `GEMM-RMSNorm-GEMM` 的代数重写：由于 RMSNorm 的缩放因子是每行共享标量，可将其应用延后到后续 GEMM 的 epilogue；前序 GEMM epilogue 只计算 partial RMS，再由轻量规约合并。

## 关键权衡

- 相比 `LayerNorm`，实现更简单、访存和计算也更轻
- 是否带偏置项、参数命名以及具体融合形式会因框架和模型实现而异
- 在手写 kernel 时，通常要同时兼顾归约效率和最后一遍逐元素写回的访存模式
- 将 RMSNorm 融入 GEMM epilogue 可以减少中间张量落 HBM，但必须保持数值等价边界，并核实重排后的误差、dtype 和反向传播实现。

## 相关来源

- [[../sources/秋招CUDA手撕题复盘（附代码）]]
- [[../sources/还在手写CUDA内核？CODA来了！LLM和新手也能让Transformer跑出光速]]

## 相关概念

- [[CUDA Kernel]]
- [[Block Reduce]]
- [[Warp Shuffle Reduce]]
- [[CODA]]

## 研究备注

- 这篇来源把 `RMSNorm` 放在面试高频题语境里；后续可补它在 `LLaMA`、`Gemma`、`Qwen` 等模型里的工程位置与融合实现
- CODA 来源中的 RMSNorm 重写是二手文章转述，后续应按原论文公式和代码确认 partial RMS 的合并方式与 backward kernel。
