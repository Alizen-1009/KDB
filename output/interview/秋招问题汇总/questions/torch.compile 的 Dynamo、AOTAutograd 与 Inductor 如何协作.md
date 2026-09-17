# torch.compile 的 Dynamo、AOTAutograd 与 Inductor 如何协作？

- 整理状态：已整理
- 题目出处：用户提供的一面面经（PyTorch 图编译与推理系统专题）

## 其他问法

- `torch.compile` 的原理是什么？
- `torch.compile`、算子融合与 CUDA Graph 有什么区别？

## 30 秒回答

`torch.compile` 是一个编译入口而不是单一编译器。TorchDynamo 在 Python frame/bytecode 层捕获可表示的 PyTorch 运算并生成带 guard 的 FX 图；训练时 AOTAutograd 进一步提前构造 forward/backward 图；Inductor 对 ATen 图做融合、调度、内存规划和代码生成，GPU 上常生成 Triton kernel，也会调用 cuBLAS/cuDNN 或 custom op。动态 Python、数据依赖控制流和 guard 失效会造成 graph break 或重新编译。

## 深入解释

```text
Python/PyTorch
  → TorchDynamo：捕获图、建立 guards、处理 graph break
  → AOTAutograd：函数化并生成 forward/backward 图
  → Inductor：融合、schedule、代码生成、外部 kernel 选择
  → Triton/C++/cuBLAS/cuDNN/custom op
```

CUDA Graph 位于更靠后的执行提交层：它重放稳定的 kernel 序列以减少 launch overhead，但不负责把 ATen 图变成 fused kernel。成熟推理框架还会把 Attention、MoE 等重型热点保留为 custom op，让 compile 负责图分区、轻量融合、shape specialization 和 piecewise CUDA Graph 组织。

## 常见误区

- `torch.compile` 不保证全图无 graph break。
- 编译成功不代表一定更快；首次编译、动态 shape 重编译和次优融合可能抵消收益。
- 它与 CUDA Graph、Triton、CUTLASS 不是互斥关系。

## 关联知识

- [[../../../../wiki/concepts/Torch Compile|Torch Compile]]
- [[../../../../wiki/concepts/算子融合|算子融合]]
- [[../../../../wiki/concepts/CUDA Graph 执行模式|CUDA Graph 执行模式]]

## 参考来源与待核实

- [[../../../../wiki/concepts/Torch Compile|Torch Compile 概念页与官方实现核对]]
- 具体 graph-break 规则、dynamic-shape 支持和 backend 行为需绑定 PyTorch 版本。

## 所属题单

- [[../sets/PyTorch图编译与推理系统一面|PyTorch 图编译与推理系统一面]]
- [[../sets/GPU与算子|GPU与算子]]
- [[../README|秋招问题汇总]]
